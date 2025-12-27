#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified evaluation of:
  • Base model (LLaMA 3.2 1B)
  • Fine-tuned model
  • Retomaton retrieval-augmented model

Per example metrics:
  • Self-PPL (generated text only)
  • Conditional PPL (prompt → reference)
  • Conditional PPL (prompt → generation)

Outputs:
  • Per-model CSVs
  • summary.csv with mean perplexities
"""

import os, sys, json, math, re
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# ============================================================
# ARGUMENT PARSER
# ============================================================
parser = argparse.ArgumentParser()

parser.add_argument("--repo_path", required=True)
parser.add_argument("--model_name", required=True)
parser.add_argument("--finetuned_dir", required=True)
parser.add_argument("--jsonl", required=True)
parser.add_argument("--output_dir", required=True)

parser.add_argument("--dstore_dir", required=True)
parser.add_argument("--dstore_size", type=int, required=True)
parser.add_argument("--dimension", type=int, required=True)

parser.add_argument("--k", type=int, required=True)
parser.add_argument("--lambda_val", type=float, required=True)
parser.add_argument("--knn_temp", type=float, required=True)
parser.add_argument("--min_knn", type=int, required=True)
parser.add_argument("--max_knn", type=int, required=True)

parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--limit", type=int, default=10)

args = parser.parse_args()
OUTPUT_DIR = Path(args.output_dir)

# ============================================================
# PATHS + IMPORTS
# ============================================================
if args.repo_path not in sys.path:
    sys.path.append(args.repo_path)

from retomaton import RetomatonWrapper
from knnlm import KEY_TYPE, DIST

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# MODELS
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_model(path):
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device).eval()

base_model = load_model(args.model_name)
ft_model = load_model(args.finetuned_dir)

# ============================================================
# LOAD DATA
# ============================================================
records = [json.loads(l) for l in open(args.jsonl)]
if len(records) < 3:
    raise ValueError("JSONL requires at least 3 records.")

def extract_triplet(msgs):
    return msgs[0]["content"], msgs[1]["content"], msgs[2]["content"]

sys_a, usr_a, asst_a = extract_triplet(records[0]["messages"])
sys_b, usr_b, asst_b = extract_triplet(records[1]["messages"])
SYSTEM_PROMPT = sys_a

eval_records = records[2:2 + args.limit]

def is_invalid_ref(t):
    if not t or not t.strip():
        return True
    cleaned = re.sub(r"\s+", " ", t.strip().lower()).strip(":;.,")
    return cleaned == "impression"

eval_records = [
    r for r in eval_records
    if not is_invalid_ref(r["messages"][2]["content"])
]

print(f"Loaded {len(eval_records)} records for evaluation.")

# ============================================================
# PROMPT + PPL HELPERS
# ============================================================
def build_prompt(user_text):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": usr_a},
        {"role": "assistant", "content": asst_a},
        {"role": "user", "content": usr_b},
        {"role": "assistant", "content": asst_b},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def self_ppl(model, text):
    enc = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    out = model(**enc, labels=enc["input_ids"])
    loss = out.loss.item()
    return loss, math.exp(loss)

def cond_ppl_prompt_ref(model, prompt, ref):
    full = prompt + ref
    enc = tokenizer(full, return_tensors="pt", truncation=True).to(device)
    labels = enc["input_ids"].clone()
    n_prompt = len(tokenizer(prompt)["input_ids"])
    labels[:, :n_prompt] = -100
    out = model(**enc, labels=labels)
    loss = out.loss.item()
    return loss, math.exp(loss)

def cond_ppl_prompt_gen(model, prompt, gen):
    full = prompt + gen
    enc = tokenizer(full, return_tensors="pt", truncation=True).to(device)
    labels = enc["input_ids"].clone()
    n_prompt = len(tokenizer(prompt)["input_ids"])
    labels[:, :n_prompt] = -100
    out = model(**enc, labels=labels)
    loss = out.loss.item()
    return loss, math.exp(loss)

def extract_continuation(full_ids, prompt_ids):
    full = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    pr = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
    return full[len(pr):].strip()

# ============================================================
# MAIN LOOP
# ============================================================
MAX_NEW_TOKENS = 180

def run_model(model, name, wrapper=None):
    if wrapper:
        wrapper.break_into(model)

    rows = []
    print(f"\n==== Running {name} ====")

    for rec in tqdm(eval_records, desc=name):
        user = rec["messages"][1]["content"]
        ref = rec["messages"][2]["content"]

        prompt = build_prompt(user)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        gen = extract_continuation(out_ids, inputs["input_ids"])

        sl, sp = self_ppl(model, gen)
        clr, cpr = cond_ppl_prompt_ref(model, prompt, ref)
        clg, cpg = cond_ppl_prompt_gen(model, prompt, gen)

        rows.append({
            "prompt": user,
            "reference": ref,
            "generated": gen,

            "self_loss": sl,
            "self_ppl": sp,

            "cond_loss_ref": clr,
            "cond_ppl_ref": cpr,

            "cond_loss_gen": clg,
            "cond_ppl_gen": cpg,
        })

    if wrapper:
        wrapper.break_out()

    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")
    return df

# ============================================================
# RETOMATON
# ============================================================
retomaton = RetomatonWrapper(
    dstore_size=args.dstore_size,
    dstore_dir=args.dstore_dir,
    dimension=args.dimension,
    knn_keytype=KEY_TYPE.last_ffn_input,
    knn_sim_func=DIST.l2,
    knn_gpu=True,
    probe=32,
    k=args.k,
    lmbda=args.lambda_val,
    knn_temp=args.knn_temp,
    min_knns=args.min_knn,
    max_knns=args.max_knn,
    move_dstore_to_mem=True
)

# ============================================================
# RUN
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_base = run_model(base_model, "Base")
df_ft = run_model(ft_model, "FineTuned")
df_reto = run_model(base_model, "Retomaton", wrapper=retomaton)

summary = pd.DataFrame([
    {
        "Model": "Base",
        "SelfPPL": df_base["self_ppl"].mean(),
        "CondPPL_Ref": df_base["cond_ppl_ref"].mean(),
        "CondPPL_Gen": df_base["cond_ppl_gen"].mean(),
    },
    {
        "Model": "FineTuned",
        "SelfPPL": df_ft["self_ppl"].mean(),
        "CondPPL_Ref": df_ft["cond_ppl_ref"].mean(),
        "CondPPL_Gen": df_ft["cond_ppl_gen"].mean(),
    },
    {
        "Model": "Retomaton",
        "SelfPPL": df_reto["self_ppl"].mean(),
        "CondPPL_Ref": df_reto["cond_ppl_ref"].mean(),
        "CondPPL_Gen": df_reto["cond_ppl_gen"].mean(),
    },
])

summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
print("\n==== Summary ====")
print(summary.round(3))
