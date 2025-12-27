# ============================================================
# 🔥 GPU-SAFE, BF16-OPTIMIZED KNN-LM + RETOMATON WRAPPER
# ============================================================

import os
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import faiss
import faiss.contrib.torch_utils

logger = logging.getLogger(__name__)
logger.setLevel(20)


# ============================================================
# ENUMS
# ============================================================

class DIST(torch.nn.Module):
    l2 = 0
    dot = 1


class KEY_TYPE:
    last_ffn_input = 0
    last_ffn_output = 1


# ============================================================
# 🔥 BF16-SAFE DISTANCE FUNCS
# ============================================================

def dist_l2_bf16(q, k):
    return torch.sum((q.unsqueeze(1) - k).pow(2), dim=-1)


def dist_dot_bf16(q, k):
    return torch.sum(q.unsqueeze(1) * k, dim=-1)


DIST_FUNCS = {
    DIST.l2: dist_l2_bf16,
    DIST.dot: dist_dot_bf16,
}


# ============================================================
# 🔥 GPU-SAFE FAISS LOADER (fp16 index, bf16 queries)
# ============================================================

def load_faiss_gpu_safe(index_path, probe, gpu_id=0):
    """Loads an IVFPQ index on GPU using safe fp16 options."""
    logger.info(f"[FAISS] Loading index: {index_path}")
    cpu_index = faiss.read_index(str(index_path))

    # safe IVF probe
    try:
        ivf = faiss.extract_index_ivf(cpu_index)
        cpu_index.nprobe = min(probe, ivf.nlist)
    except Exception:
        cpu_index.nprobe = probe

    # ---- GPU options ----
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    co.useFloat16LookupTables = True
    co.indicesOptions = faiss.INDICES_32_BIT
    co.storeTransposed = False

    logger.info("[FAISS] Moving IVFPQ -> GPU (fp16 codes)")
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index, co)
    gpu_index.nprobe = cpu_index.nprobe

    return cpu_index, gpu_index


# ============================================================
# 🔥 BF16 Activation Capturer
# ============================================================

class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.capture_input = capture_input
        self.captured = None

    def forward(self, module, input, output):
        if self.capture_input:
            self.captured = input[0].detach().to(torch.bfloat16)
        else:
            self.captured = output.detach().to(torch.bfloat16)


# ============================================================
# 🔥 Main KNN Wrapper (bf16 version)
# ============================================================

class KNNWrapper:
    def __init__(
        self,
        dstore_size,
        dstore_dir,
        dimension,
        knn_sim_func=DIST.l2,
        knn_keytype=KEY_TYPE.last_ffn_input,
        k=1024,
        lmbda=0.25,
        knn_temp=1.0,
        probe=32,
        no_load_keys=False,
        move_dstore_to_mem=False,
        knn_gpu=True,
        recompute_dists=False,
    ):
        self.dstore_size = dstore_size
        self.dstore_dir = Path(dstore_dir)
        self.dimension = dimension
        self.knn_sim_func = knn_sim_func
        self.knn_keytype = knn_keytype
        self.k = k
        self.lmbda = lmbda
        self.knn_temperature = knn_temp
        self.probe = probe

        self.no_load_keys = no_load_keys
        self.move_dstore_to_mem = move_dstore_to_mem
        self.recompute_dists = recompute_dists

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.knn_gpu = knn_gpu and torch.cuda.is_available()

        self.model = None
        self.index = None
        self.reconstruct_index = None
        self.keys = None
        self.vals = None
        self.hook_handles = []
        self.activation_capturer = None
        self.vocab_size = None
        self.labels = None
        self.is_encoder_decoder = None

        self.dist_func = DIST_FUNCS[self.knn_sim_func]

    # ------------------------------------------------------------
    # 🔥 FAISS Setup (bf16 → fp16 boundary)
    # ------------------------------------------------------------
    def setup_faiss(self):
        index_path = f"{self.dstore_dir}/index_{self.model.config.model_type}_{self.dstore_size}_{self.dimension}.indexed"

        # Load index on GPU (fp16)
        cpu_index, gpu_index = load_faiss_gpu_safe(index_path, self.probe)

        # Load value memmap
        kv_prefix = f"{self.dstore_dir}/dstore_{self.model.config.model_type}_{self.dstore_size}_{self.dimension}"
        self.vals = np.memmap(f"{kv_prefix}_vals.npy", dtype=np.int32, mode="r", shape=(self.dstore_size, 1))

        if not self.no_load_keys:
            self.keys = np.memmap(f"{kv_prefix}_keys.npy", dtype=np.float16, mode="r",
                                  shape=(self.dstore_size, self.dimension))

        # Optionally move keys to GPU (bf16)
        if self.move_dstore_to_mem:
            self.keys = torch.as_tensor(self.keys[:], dtype=torch.float16, device=self.device)
            self.keys = self.keys.to(torch.bfloat16)

            self.vals = torch.as_tensor(self.vals[:], dtype=torch.int32, device=self.device)

        return cpu_index, gpu_index

    # ------------------------------------------------------------
    # 🔥 Hook-In to Model
    # ------------------------------------------------------------
    def break_into(self, model):
        self.model = model
        model.broken_into = True

        # FAISS setup
        self.reconstruct_index, self.index = self.setup_faiss()

        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        # Capture activations
        layer_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer = layer_fn(model)
        self.activation_capturer = ActivationCapturer(layer, capture_input=capture_input)
        self.register_hook(layer, self.activation_capturer)

        # Hook final lm_head
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features

    # ------------------------------------------------------------
    # 🔥 KNN Search (bf16 inside Torch → fp16 for FAISS)
    # ------------------------------------------------------------
    def get_knns(self, queries_bf16):
        # Convert for FAISS (must be fp16 or fp32)
        q = queries_bf16.to(torch.float16)

        dists, knns = self.index.search(q, self.k)
        return dists.to(self.device), knns.to(self.device)

    # ------------------------------------------------------------
    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    # ------------------------------------------------------------
    # 🔥 MAIN RETRIEVAL INTERPOLATION (bf16)
    # ------------------------------------------------------------
    def post_forward_hook(self, module, input, output):
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1

        # bf16 logits
        lm_logits = torch.nn.functional.log_softmax(output.to(torch.bfloat16), dim=-1)

        queries = self.activation_capturer.captured  # (B, T, D) bf16

        # mask
        if self.labels is None:
            mask = torch.zeros([batch, time_dim-1], dtype=torch.bool, device=self.device)
            mask = torch.cat([mask, torch.ones([batch,1], dtype=torch.bool, device=self.device)], dim=-1)
        else:
            mask = torch.cat([
                self.labels[:, shift:] != -100,
                torch.zeros([self.labels.shape[0], shift], dtype=torch.bool, device=self.device)
            ], dim=-1)

        lm_logits = lm_logits[mask]
        queries = queries[mask]

        # ---------------------------
        # 🔥 KNN Search
        # ---------------------------
        dists, knns = self.get_knns(queries)

        # recompute distances in bf16 (optional)
        if self.recompute_dists:
            if isinstance(self.keys, np.memmap):
                keys_fp16 = torch.from_numpy(self.keys[knns]).to(self.device)
            else:
                keys_fp16 = self.keys[knns]

            knn_vecs = keys_fp16.to(torch.bfloat16)
            dists = self.dist_func(queries, knn_vecs)

        # ---------------------------
        # 🔥 Build knn distribution
        # ---------------------------
        neg_d = -dists.to(torch.bfloat16)
        probs = torch.nn.functional.softmax(neg_d / self.knn_temperature, dim=-1)

        vals = torch.as_tensor(self.vals[knns], device=self.device).squeeze(-1)

        knn_log_probs = torch.full(
            (vals.shape[0], self.vocab_size),
            fill_value=-10000.0,
            dtype=torch.bfloat16,
            device=self.device
        )

        knn_log_probs.scatter_add_(1, vals, probs)
        knn_log_probs = knn_log_probs.clamp(min=-10000).log()

        # ---------------------------
        # 🔥 Interpolate (all bf16)
        # ---------------------------
        out = torch.logaddexp(
            lm_logits + torch.log(torch.tensor(1.0 - self.lmbda, device=self.device, dtype=torch.bfloat16)),
            knn_log_probs + torch.log(torch.tensor(self.lmbda, device=self.device, dtype=torch.bfloat16))
        )

        # write back
        output[mask] = out.to(output.dtype)
        return output

    # ------------------------------------------------------------
    def register_hook(self, layer, func, pre=False):
        h = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(h)

    # ------------------------------------------------------------
    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    # ------------------------------------------------------------
    def get_metrics(self):
        return {}

    # ------------------------------------------------------------
    @staticmethod
    def get_model_last_layer(model_type):
        return lambda m: m.lm_head

    # ------------------------------------------------------------
    model_layer_to_capture = {
        "llama": {
            KEY_TYPE.last_ffn_input: (lambda m: m.model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda m: m.model.layers[-1], False),
        },
        "gpt2": {
            KEY_TYPE.last_ffn_input: (lambda m: m.transformer.h[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda m: m.transformer.h[-1], False),
        },
        "olmo": {
            KEY_TYPE.last_ffn_input: (lambda m: m.model.transformer.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda m: m.model.transformer.layers[-1], False),
        },
        "olmo2": {
            KEY_TYPE.last_ffn_input: (lambda m: m.model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda m: m.model.layers[-1], False),
        },
    }

# ============================================================
# END OF FILE
# ============================================================
# ============================================================
# 🔥 GPU-SAFE, BF16-OPTIMIZED KNN DATASTORE SAVER
# ============================================================

import os
import logging
import numpy as np
import torch
from pathlib import Path
from torch import nn

logger = logging.getLogger(__name__)
logger.setLevel(20)


# ============================================================
# 🔥 BF16 Activation Capturer (shared with wrapper)
# ============================================================

class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.capture_input = capture_input
        self.captured = None

    def forward(self, module, input, output):
        if self.capture_input:
            self.captured = input[0].detach().to(torch.bfloat16)
        else:
            self.captured = output.detach().to(torch.bfloat16)


# ============================================================
# 🔥 KNNSaver — fully bf16-safe, GPU-safe, memmap-safe
# ============================================================

class KNNSaver(object):
    def __init__(self, dstore_size, dstore_dir, dimension, knn_keytype):
        self.dstore_size = dstore_size
        self.dstore_dir = Path(dstore_dir)
        self.dimension = dimension
        self.knn_keytype = knn_keytype

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.labels = None
        self.dstore_idx = 0
        self.hook_handles = []

        logger.info(f"[KNNSaver] Key type: {self.knn_keytype}")
        logger.info("[KNNSaver] Saving keys in fp16 (required for FAISS)")

    # ------------------------------------------------------------
    # 🔥 Connect hooks to model
    # ------------------------------------------------------------
    def break_into(self, model):
        """
        Attaches hooks to capture bf16 activations + int labels,
        but writes datastore on disk in fp16.
        """

        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # ---- Create datastore folder ----
        kv_prefix = get_dstore_path(self.dstore_dir, model.config.model_type, self.dstore_size, self.dimension)
        keys_file = f"{kv_prefix}_keys.npy"
        vals_file = f"{kv_prefix}_vals.npy"

        Path(keys_file).parent.mkdir(parents=True, exist_ok=True)

        # If datastore exists, we reopen it; otherwise create new
        mode = "r+" if os.path.exists(keys_file) and os.path.exists(vals_file) else "w+"

        # Memmaps (fp16 keys, int32 vals)
        self.dstore_keys = np.memmap(
            keys_file,
            dtype=np.float16,
            mode=mode,
            shape=(self.dstore_size, self.dimension)
        )

        self.dstore_vals = np.memmap(
            vals_file,
            dtype=np.int32,
            mode=mode,
            shape=(self.dstore_size, 1)
        )

        # ---- Capture activations ----
        layer_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        capture_layer = layer_fn(model)
        self.activation_capturer = ActivationCapturer(capture_layer, capture_input)
        self.register_hook(capture_layer, self.activation_capturer)

        # ---- Capture labels ----
        self.original_forward = model.forward
        model.forward = self.pre_forward_hook

        # ---- Hook final layer to save keys/vals ----
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)

        logger.info("[KNNSaver] Hooks registered. Ready to save datastore.")

    # ------------------------------------------------------------
    # 🔥 Hook forward to capture labels
    # ------------------------------------------------------------
    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            raise ValueError(
                "\n❌ labels must be provided when saving the datastore.\n"
                "You may be using --predict_with_generate incorrectly."
            )
        self.labels = labels
        return self.original_forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    # ------------------------------------------------------------
    # 🔥 Final layer hook: save bf16 → fp16 keys + integer values
    # ------------------------------------------------------------
    def post_forward_hook(self, module, inputs, outputs):
        """
        Extract activations (bf16) and save them as fp16 to memmap.
        """

        shift = 0 if self.is_encoder_decoder else 1

        # captured activations: (batch, time, dim) bf16
        keys_bf16 = self.activation_capturer.captured
        labels = self.labels

        if shift == 1:
            keys_bf16 = keys_bf16[:, :-shift]

        keys_bf16 = keys_bf16.flatten(0, 1)            # (B*T, dim) bf16
        values = labels[:, shift:].flatten(0, 1)       # (B*T)

        # mask out padding
        mask = values != -100
        keys_bf16 = keys_bf16[mask]
        values = values[mask]

        batch_count = keys_bf16.shape[0]

        # Prevent overflow
        if self.dstore_idx >= self.dstore_size:
            return outputs

        if self.dstore_idx + batch_count > self.dstore_size:
            batch_count = self.dstore_size - self.dstore_idx
            keys_bf16 = keys_bf16[:batch_count]
            values = values[:batch_count]

        # -------- SAVE (fp16 keys, int32 vals) --------
        keys_fp16 = keys_bf16.to(torch.float16).cpu().numpy()
        vals_i32 = values.to(torch.int32).cpu().numpy().reshape(-1, 1)

        self.dstore_keys[self.dstore_idx:self.dstore_idx + batch_count] = keys_fp16
        self.dstore_vals[self.dstore_idx:self.dstore_idx + batch_count] = vals_i32

        self.dstore_idx += batch_count

        return outputs

    # ------------------------------------------------------------
    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    # ------------------------------------------------------------
    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None:
            self.model.forward = self.original_forward
            self.model.broken_into = None


# ============================================================
# 🔥 Path helpers
# ============================================================

def get_dstore_path(dstore_dir, model_type, dstore_size, dimension):
    return f"{dstore_dir}/dstore_{model_type}_{dstore_size}_{dimension}"


def get_index_path(dstore_dir, model_type, dstore_size, dimension):
    return f"{dstore_dir}/index_{model_type}_{dstore_size}_{dimension}.indexed"
