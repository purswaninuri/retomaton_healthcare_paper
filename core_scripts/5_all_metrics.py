#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive evaluation script for clinical text generation.

Computes:
- ROUGE (1/2/L/Lsum)
- BLEU
- BERTScore (P/R/F1)
- FOUR NLI faithfulness metrics (DeBERTa MNLI)
- Biomedical NER entity overlap + hallucination metrics (SciSpaCy + UMLS)
- Composite "safe factuality" score
- Merges everything with existing self/conditional perplexity metrics.

Skips records where reference impressions are blank or contain only 'impression:'.

RUN:
    python evaluate.py --input_dir /path/to/output_dir

NLI SCORES (all are: entailment_prob - contradiction_prob, range ≈ [-1, +1])
---------------------------------------------------------------------------
> +0.50   = Very faithful, low hallucination risk
+0.20–0.50 = Mostly faithful
-0.20–+0.20 = Neutral / paraphrase / possible omission
< -0.20  = Contradiction risk
< -0.50  = Strong hallucination or fabrication

ENTITY METRICS (all in [0, 1])
---------------------------------------------------------------------------
entity_recall_ref_gen:
    - "How many reference entities are preserved in the generation?"
    - >0.80 good, 0.5–0.8 moderate, <0.5 missing important findings.

entity_precision_ref_gen:
    - "How many generated entities are supported by the reference?"
    - >0.80 good (few fabricated entities), <0.5 many invented findings.

entity_jaccard_*:
    - Overall overlap between sets (prompt vs gen, ref vs gen).
    - >0.6 strong overlap, <0.3 weak overlap.

COMPOSITE SAFE FACTUALITY SCORE (rough heuristic scale)
---------------------------------------------------------------------------
composite_safe_score ≈ weighted mix of:
    + nli_ref_gen
    + nli_prompt_gen
    + entity_recall_ref_gen
    + entity_precision_ref_gen
    + bertscore_f1
    + rougeL

Interpretation (empirical guideline):
    > 0.60   → high-quality + faithful
    0.40–0.60 → acceptable but may need review
    < 0.40   → likely hallucinations or omissions
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re
import evaluate
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------
# Optional SciSpaCy imports for biomedical NER + UMLS
# ------------------------------------------------------------
NER_AVAILABLE = True
try:
    import spacy
    import scispacy  # noqa: F401 (import needed to register components)
    from scispacy.linking import EntityLinker
except ImportError:
    print("⚠️ SciSpaCy or its models are not installed. "
          "Entity-based metrics will be disabled.")
    NER_AVAILABLE = False

# ------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Evaluate LLM generations with ROUGE, BLEU, BERTScore, NLI, and biomedical entity metrics."
)
parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help="Directory containing Base.csv / FineTuned.csv / Retomaton.csv files."
)
args = parser.parse_args()

BASE_DIR = Path(args.input_dir)
OUTPUT_DIR = BASE_DIR
print(f"📂 Input directory: {BASE_DIR}")

# ------------------------------------------------------------
# Load text overlap metrics
# ------------------------------------------------------------
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# ------------------------------------------------------------
# Load DeBERTa NLI Faithfulness Model
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Loading NLI DeBERTa cross-encoder on {device} ...")

nli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base")
nli_model = AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/nli-deberta-v3-base"
).to(device)
nli_model.eval()

def compute_nli_faithfulness(premise, hypothesis):
    """
    Computes an NLI-based faithfulness score in [-1, +1].

    premise    : assumed ground-truth statement (e.g., reference or prompt)
    hypothesis : statement being evaluated (e.g., generated text)

    We compute:
        entailment_prob - contradiction_prob

    Interpretation (heuristic):
        > +0.5   → strong support (highly faithful)
        0–0.5    → mild/partial support
        ~ 0      → neutral or ambiguous
        < 0      → contradiction (hallucination / fabrication)
        < -0.5   → strong contradiction
    """
    if not premise or not hypothesis:
        return 0.0

    enc = nli_tokenizer(
        premise, hypothesis,
        truncation=True, padding=True,
        max_length=512, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = nli_model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    # cross-encoder/nli-deberta-v3-base label order: ['contradiction', 'entailment', 'neutral']
    contradiction, entailment, neutral = probs
    return float(entailment - contradiction)


# ------------------------------------------------------------
# SciSpaCy biomedical NER + UMLS linker
# ------------------------------------------------------------
if NER_AVAILABLE:
    print("🧬 Loading SciSpaCy model en_ner_bc5cdr_md with UMLS linker (this may take a moment)...")
    # Disease + chemical NER model
    nlp = spacy.load("en_ner_bc5cdr_md")

    # Add UMLS linker (maps entities to CUIs, MeSH, ICD, etc.)
    if "scispacy_linker" not in nlp.pipe_names:
        linker = EntityLinker(
            resolve_abbreviations=True,
            name="umls"  # UMLS KB; contains MeSH, SNOMED, ICD mappings
        )
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    else:
        linker = nlp.get_pipe("scispacy_linker")
else:
    nlp = None
    linker = None

def extract_biomed_entities(text):
    """
    Extracts biomedical entities from text using SciSpaCy + UMLS.

    Returns:
        surface_ents: set of lowercased surface strings (e.g., {"pneumonia", "aspiration"})
        cui_ents    : set of UMLS CUI identifiers (e.g., {"C0032285", ...})

    If SciSpaCy is not available or text is empty, returns (empty, empty).
    """
    if not NER_AVAILABLE or not isinstance(text, str) or not text.strip():
        return set(), set()

    doc = nlp(text)
    surface_ents = set()
    cui_ents = set()
    for ent in doc.ents:
        surf = ent.text.strip().lower()
        if surf:
            surface_ents.add(surf)
        # Use top-ranked linked UMLS concept if present
        if ent._.kb_ents:
            best_cui, score = ent._.kb_ents[0]
            cui_ents.add(best_cui)
    return surface_ents, cui_ents


def safe_div(num, den):
    """Avoid division by zero; if den == 0, return 1.0 (perfect by convention for empty sets)."""
    if den == 0:
        return 1.0
    return float(num) / float(den)


# ------------------------------------------------------------
# Expected model output CSVs
# ------------------------------------------------------------
files = {
    "base": BASE_DIR / "Base.csv",
    "finetuned": BASE_DIR / "FineTuned.csv",
    "retomaton": BASE_DIR / "Retomaton.csv",
}
files = {k: v for k, v in files.items() if v.exists()}
if not files:
    raise FileNotFoundError(f"No Base.csv / FineTuned.csv / Retomaton.csv files found in {BASE_DIR}")

# ------------------------------------------------------------
# Utility: detect empty/meaningless reference
# ------------------------------------------------------------
def is_invalid_reference(text: str) -> bool:
    if not text or text.strip() == "":
        return True
    cleaned = re.sub(r"[\s\n\r\t]+", " ", text.strip().lower()).strip(" :;.,")
    # Cases like just "impression:" or "impression"
    return cleaned == "impression"


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
summary_rows = []

for model_name, file_path in files.items():
    print(f"\n🔍 Evaluating {model_name.upper()}  →  {file_path}")
    df = pd.read_csv(file_path)

    df = df.rename(columns={
        "reference_impression": "reference",
        "generated_impression": "generated"
    })

    df["reference"] = df["reference"].fillna("").astype(str)
    df["generated"] = df["generated"].fillna("").astype(str)

    valid_mask = ~df["reference"].apply(is_invalid_reference)
    skipped = len(df) - valid_mask.sum()
    df = df[valid_mask].reset_index(drop=True)

    if len(df) == 0:
        print(f"⚠ No valid references — skipping {model_name}")
        continue

    print(f"✅ {len(df)} usable rows ({skipped} skipped)")

    refs = df["reference"].tolist()
    preds = df["generated"].tolist()

    metric_rows = []
    for i, (p, r) in tqdm(enumerate(zip(preds, refs)), total=len(preds), desc=model_name):
        prompt = df.loc[i, "prompt_user_input"] if "prompt_user_input" in df.columns else ""

        # ------------------------------------------------------
        # Text overlap metrics (per-example)
        # ------------------------------------------------------
        rouge_res = rouge.compute(predictions=[p], references=[r], use_stemmer=True)
        bleu_res = bleu.compute(predictions=[p], references=[[r]])
        bert_res = bertscore.compute(predictions=[p], references=[r], lang="en")
        bert_f1 = float(bert_res["f1"][0])

        # ------------------------------------------------------
        # FOUR–WAY NLI METRICS
        # ------------------------------------------------------
        # 1️⃣ nli_ref_gen: hallucination vs reference (primary factual metric)
        nli_ref_gen = compute_nli_faithfulness(r, p)

        # 2️⃣ nli_prompt_gen: hallucination vs original prompt (prompt drift)
        nli_prompt_gen = compute_nli_faithfulness(prompt, p) if prompt else 0.0

        # 3️⃣ nli_prompt_ref: dataset quality (does reference follow prompt?)
        nli_prompt_ref = compute_nli_faithfulness(prompt, r) if prompt else 0.0

        # 4️⃣ nli_gen_ref: missing info (does generated text entail reference?)
        nli_gen_ref = compute_nli_faithfulness(p, r)

        # ------------------------------------------------------
        # Biomedical ENTITY METRICS (NER + UMLS)
        # ------------------------------------------------------
        surf_prompt, cui_prompt = extract_biomed_entities(prompt)
        surf_ref,    cui_ref    = extract_biomed_entities(r)
        surf_gen,    cui_gen    = extract_biomed_entities(p)

        # Overlap between REFERENCE and GENERATED
        inter_ref_gen = len(surf_ref & surf_gen)
        union_ref_gen = len(surf_ref | surf_gen)
        entity_jaccard_ref_gen = safe_div(inter_ref_gen, union_ref_gen)
        entity_recall_ref_gen = safe_div(inter_ref_gen, len(surf_ref))   # % of reference entities preserved
        entity_precision_ref_gen = safe_div(inter_ref_gen, len(surf_gen))# % of generated entities supported by ref

        # Overlap between PROMPT and GENERATED (input consistency / prompt drift)
        inter_prompt_gen = len(surf_prompt & surf_gen)
        union_prompt_gen = len(surf_prompt | surf_gen)
        entity_jaccard_prompt_gen = safe_div(inter_prompt_gen, union_prompt_gen)

        # Surface-level hallucinated entities: present in GEN but not in PROMPT or REF
        new_surf_from_prompt = surf_gen - surf_prompt
        new_surf_from_ref = surf_gen - surf_ref
        hallucinated_surface_ents = surf_gen - (surf_prompt | surf_ref)
        entity_neg_hallucination_flag = 1 if len(hallucinated_surface_ents) > 0 else 0

        # UMLS concept-level hallucinations (new CUIs never in prompt or reference)
        new_cui_from_union_pr = cui_gen - (cui_prompt | cui_ref)
        umls_new_concepts_count = len(new_cui_from_union_pr)
        umls_hallucination_flag = 1 if umls_new_concepts_count > 0 else 0

        # ------------------------------------------------------
        # Composite SAFE factuality score
        # ------------------------------------------------------
        # Weighted mix of: NLI faithfulness + entity recall/precision + semantic and surface similarity.
        composite_safe_score = (
            0.40 * nli_ref_gen +
            0.20 * entity_recall_ref_gen +
            0.15 * entity_precision_ref_gen +
            0.10 * nli_prompt_gen +
            0.10 * bert_f1 +
            0.05 * float(rouge_res["rougeL"])
        )

        metric_rows.append({
            # ROUGE
            "rouge1": float(rouge_res["rouge1"]),
            "rouge2": float(rouge_res["rouge2"]),
            "rougeL": float(rouge_res["rougeL"]),
            "rougeLsum": float(rouge_res["rougeLsum"]),
            # BLEU
            "bleu": float(bleu_res["bleu"]),
            # BERTScore
            "bertscore_precision": float(bert_res["precision"][0]),
            "bertscore_recall": float(bert_res["recall"][0]),
            "bertscore_f1": bert_f1,
            # NLI
            "nli_ref_gen": nli_ref_gen,
            "nli_prompt_gen": nli_prompt_gen,
            "nli_prompt_ref": nli_prompt_ref,
            "nli_gen_ref": nli_gen_ref,
            # Entity metrics (surface-level)
            "entity_jaccard_ref_gen": entity_jaccard_ref_gen,
            "entity_recall_ref_gen": entity_recall_ref_gen,
            "entity_precision_ref_gen": entity_precision_ref_gen,
            "entity_jaccard_prompt_gen": entity_jaccard_prompt_gen,
            "entity_neg_hallucination_flag": entity_neg_hallucination_flag,
            # UMLS concept-level hallucination metrics
            "umls_new_concepts_count": umls_new_concepts_count,
            "umls_hallucination_flag": umls_hallucination_flag,
            # Composite score
            "composite_safe_score": composite_safe_score,
        })

    metric_df = pd.DataFrame(metric_rows)
    merged = pd.concat([df.reset_index(drop=True), metric_df], axis=1)

    out_csv = OUTPUT_DIR / f"generated_{model_name}_with_allmetrics.csv"
    merged.to_csv(out_csv, index=False)
    print(f"💾 Saved detailed metrics → {out_csv}")

    # Averages across all valid rows
    avg = merged[[
        "self_loss", "self_ppl", "cond_loss", "cond_ppl",
        "rouge1", "rouge2", "rougeL", "rougeLsum",
        "bleu", "bertscore_precision", "bertscore_recall", "bertscore_f1",
        "nli_ref_gen", "nli_prompt_gen", "nli_prompt_ref", "nli_gen_ref",
        "entity_jaccard_ref_gen", "entity_recall_ref_gen", "entity_precision_ref_gen",
        "entity_jaccard_prompt_gen",
        "entity_neg_hallucination_flag",
        "umls_new_concepts_count", "umls_hallucination_flag",
        "composite_safe_score",
    ]].mean(numeric_only=True).to_dict()

    avg["model"] = model_name
    summary_rows.append(avg)

# ------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------
if summary_rows:
    summary_df = pd.DataFrame(summary_rows).set_index("model")

    csv_path = OUTPUT_DIR / "all_metrics_summary.csv"
    summary_df.to_csv(csv_path)

    print("\n=== SUMMARY TABLE (mean over valid rows) ===")
    print(summary_df.round(4))
    print(f"\n📊 Saved → {csv_path}")
else:
    print("⚠ No valid data to summarize.")
