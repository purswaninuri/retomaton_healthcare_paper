import os

import logging
import time
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from pathlib import Path

import faiss
import faiss.contrib.torch_utils

logger = logging.getLogger(__name__)
logger.setLevel(20)

# -----------------------------------------------------
# Unified logit extractor for all HuggingFace models
# -----------------------------------------------------
# -----------------------------------------------------
# Unified logit extractor for all HuggingFace models
# -----------------------------------------------------
def extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    if torch.is_tensor(output):
        return output
    raise ValueError(f"[KNNLM ERROR] Cannot extract logits from type {type(output)}")


class DIST(Enum):
    l2 = auto()
    dot = auto()

    @staticmethod
    def from_string(s):
        try:
            return DIST[s.lower()]
        except KeyError:
            raise ValueError()


class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()

class KNNWrapper(object):
    def __init__(self, dstore_size, dstore_dir, dimension, 
            knn_sim_func=None, knn_keytype=None,
            no_load_keys=False, move_dstore_to_mem=False, knn_gpu=True,
            recompute_dists = False,
            k=1024, lmbda=0.25, knn_temp=1.0, probe=32):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.lmbda = lmbda
        self.k = k
        self.knn_temperature = knn_temp
        self.probe = probe
        self.knn_sim_func = DIST.l2 if knn_sim_func is None else knn_sim_func
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype
        self.no_load_keys = no_load_keys
        self.recompute_dists = recompute_dists
        self.move_dstore_to_mem = move_dstore_to_mem
        self.knn_gpu = knn_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt_input_ids = None
        self.keys = None
        self.values = None
        self.prompt_attention_mask = None
        self.model = None
        self.vocab_size = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.hook_handles = []

        dist_type_to_dist_func = {
            DIST.l2: KNNWrapper.l2,
            DIST.dot: KNNWrapper.dotprod,
        }
        self.dist_func = dist_type_to_dist_func[knn_sim_func] # l2 or dot product function


    def setup_faiss(self):
        if not self.dstore_dir:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension) 
        cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
        logger.info(f'Reading datastore took {time.time() - start} s')
        cpu_index.nprobe = self.probe

        if self.knn_gpu:
            start = time.time()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
            logger.info(f'Moving index to GPU took {time.time() - start} s')
        else:
            gpu_index = cpu_index

        # make_direct_map() allows calling reconstruct(n), 
        # and reconstructing key vectors given their ids
        # currently, this is implemented only for CPU indexes:
        # https://github.com/facebookresearch/faiss/issues/2181
        cpu_index.make_direct_map()

        keys_vals_prefix = get_dstore_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension)
        if not self.no_load_keys:
            self.keys = np.memmap(f'{keys_vals_prefix}_keys.npy', dtype=np.float16, mode='r',
                                  shape=(self.dstore_size, self.dimension))
        self.vals = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r',
                              shape=(self.dstore_size, 1))
        # self.vals = torch.from_numpy(self.vals).to(self.device)

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if self.move_dstore_to_mem:
            logger.info('Loading to memory...')
            start = time.time()

            if not self.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(f'{keys_vals_prefix}_keys.npy', 
                    dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = self.keys_from_memmap[:].astype(np.float16)

            del self.vals
            vals_from_memmap = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r', shape=(self.dstore_size, 1))
            self.vals = torch.from_numpy(vals_from_memmap[:]).long().to(self.device)
            del vals_from_memmap
            logger.info('Loading to memory took {} s'.format(time.time() - start))

        return cpu_index, gpu_index

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.reconstruct_index, self.index = self.setup_faiss()
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook
        
        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features

    def get_knns(self, queries):
        if not self.knn_gpu:
            queries = queries.cpu()
        dists, knns = self.index.search(queries, self.k)
        dists, knns = dists.to(self.device), knns.to(self.device)
        return dists, knns

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):

        # -----------------------------------------------------------
        # 1. Extract logits robustly for GPT/LLaMA/OLMo/Gemma/Phi-3
        # -----------------------------------------------------------
        logits = extract_logits(output)
        

        # -----------------------------------------------------------
        # 2. FIX FOR PHI-3 EXTRA DIMENSION DURING GENERATE()
        # logits shape may be: [B, 1, T, V]
        # -----------------------------------------------------------
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)    # → [B, T, V]

        # shape check
        if logits.ndim != 3:
            raise ValueError(
                f"[KNNLM ERROR] Expected [batch, seq, vocab], got {logits.shape}"
            )

        batch, time_dim, vocab = logits.shape

        # -----------------------------------------------------------
        # 🔥 2. LOG-SOFTMAX FOR LM BASELINE
        # -----------------------------------------------------------
        lm_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # -----------------------------------------------------------
        # 🔥 3. EXTRACT HIDDEN ACTIVATIONS (queries)
        # -----------------------------------------------------------
        queries = self.activation_capturer.captured   # [B, T, D]
        if queries.ndim != 3:
            raise ValueError(
                f"[KNNLM ERROR] queries must be [batch, seq, dim], got {queries.shape}"
            )

        # -----------------------------------------------------------
        # 🔥 4. SELECT POSITIONS WHERE WE INTERPOLATE
        # (Phi is causal LM => shift=1)
        # -----------------------------------------------------------
        shift = 0 if self.is_encoder_decoder else 1

        if self.labels is None:
            # generation mode: only last token
            nonpad_mask = torch.zeros_like(lm_log_probs[..., 0], dtype=torch.bool)
            nonpad_mask[:, -1] = True
        else:
            # training/eval mode: valid labels only
            nonpad_mask = torch.cat(
                [
                    self.labels[:, shift:] != -100,
                    torch.zeros([batch, shift], dtype=torch.bool).to(self.device),
                ],
                dim=-1,
            )

        # flatten to (N, D)
        flat_lm = lm_log_probs[nonpad_mask]
        flat_queries = queries[nonpad_mask]     # (N, D)

        # -----------------------------------------------------------
        # 🔥 5. KNN RETRIEVAL
        # -----------------------------------------------------------
        dists, knns = self.get_knns(flat_queries)

        if self.recompute_dists:
            # recompute true L2 on GPU
            knn_vecs = torch.from_numpy(self.keys[knns]).to(self.device)
            dists = self.dist_func(flat_queries, knn_vecs)

        # ------------------------------------
        # 🔥 6. Convert distances → log-probs
        # ------------------------------------
        neg_d = -dists
        knn_log_probs, _ = self.knns_to_log_prob(knns, neg_d)

        # -----------------------------------------------------------
        # 🔥 7. INTERPOLATION
        # -----------------------------------------------------------
        blended = KNNWrapper.interpolate(knn_log_probs, flat_lm, self.lmbda)

        # -----------------------------------------------------------
        # 🔥 8. WRITE BACK LOGITS (in-place)
        # -----------------------------------------------------------
        logits[nonpad_mask] = blended

        # -----------------------------------------------------------
        # 🔥 9. Return a *modified* ModelOutput if needed
        # -----------------------------------------------------------
        if hasattr(output, "logits"):
            output.logits = logits
            return output
        else:
            return logits
 

    def knns_to_log_prob(self, knns, neg_dists):
        probs = torch.nn.functional.softmax(neg_dists / self.knn_temperature, dim=-1)
        vals_at_knns = self.vals[knns].squeeze(-1) # (nonpad batch * time, k)
        knn_log_probs = torch.full(size=(vals_at_knns.shape[:-1] + (self.vocab_size,)), fill_value=0.0).to(self.device) \
            .scatter_add(dim=-1, index=vals_at_knns, src=probs).log() # (nonpad_batch * time, vocab)
        knn_log_probs = torch.nan_to_num(knn_log_probs, nan=None, neginf=-10000.0)
        return knn_log_probs, vals_at_knns
        
    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None
    
    def get_metrics(self):
        return {}
    
    @staticmethod
    def l2(query, keys):
        # query: (batch*time, dim)
        # keys:  (batch*time, k, dim)
        # returns: (batch*time, k)
        return torch.sum((query.unsqueeze(-2) - keys)**2, dim=-1)

    @staticmethod
    def dotprod(query, keys):
        # query: (batch, beams, dim)
        # keys:  (batch, 1, time, dim)
        # returns: (batch, beams, time)
        return torch.sum((query.unsqueeze(-2) * keys), dim=-1)


    @staticmethod
    def interpolate(knn_log_probs, lm_log_probs, lmbda):
        interpolated = torch.logaddexp(
            lm_log_probs + np.log(1 - lmbda), 
            knn_log_probs + np.log(lmbda))

        return interpolated

    @staticmethod
    def get_model_last_layer(model_type):
        # works for gpt2, marian, t5. If a model does not have an ".lm_head" layer, 
        # add an "if model_type is ..." statement here, and return the output embedding layer
        return lambda model: model.lm_head

    @staticmethod
    def get_model_embedding_layer(model_type):
        if model_type.startswith('gpt2'):
            return lambda model: model.transformer.wte

    # For every model name and key type, returns a lambda that returns the relevant layer in the model, 
    # and whether the input of that layer should be captured (True) or the output (False)
    model_layer_to_capture = {
        'bart': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        'gpt2': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.h[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.h[-1], False),
        },
        'marian': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        't5': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.block[-1].layer[2].DenseReluDense, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.block[-1].layer[2], False),
        },
        'llama': {
            KEY_TYPE.last_ffn_input: (lambda model: model.model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.model.layers[-1], False),
        },
        'olmo': {
            KEY_TYPE.last_ffn_input: (lambda model: model.model.transformer.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.model.transformer.layers[-1], False),
        },
        'olmo2': {
            KEY_TYPE.last_ffn_input: (lambda model: model.model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.model.layers[-1], False),
        },

        # -------------------------
        # PHI MODELS
        # -------------------------
        "phi": {
            KEY_TYPE.last_ffn_input: (lambda model: model.model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.model.layers[-1], False),
        },
        "phi3": {
            KEY_TYPE.last_ffn_input: (lambda model: model.model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.model.layers[-1], False),
        },

        # -------------------------
        # GEMMA MODELS
        # -------------------------
        "gemma": {
            KEY_TYPE.last_ffn_input: (lambda model: model.model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.model.layers[-1], False),
        },
        "gemma2": {
            KEY_TYPE.last_ffn_input: (lambda model: model.model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.model.layers[-1], False),
        },
        "gemma3": {
            KEY_TYPE.last_ffn_input: (lambda model: model.model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.model.layers[-1], False),
        },
    }

    

class KNNSaver(object):
    def __init__(self, dstore_size, dstore_dir, dimension, knn_keytype=None):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.dstore_idx = 0
        self.dstore_keys = None
        self.dstore_vals = None
        self.labels = None
        self.hook_handles = []

        logger.info(f'keytype being saved: {self.knn_keytype}')
        logger.info('Saving fp16')

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder
        
        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook
        
        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)

        keys_vals_prefix = get_dstore_path(self.dstore_dir, model.config.model_type, self.dstore_size, self.dimension)
        keys_filename = f'{keys_vals_prefix}_keys.npy'
        vals_filename = f'{keys_vals_prefix}_vals.npy'
        if os.path.exists(keys_filename) and os.path.exists(vals_filename):
            mode = 'r'
        else:
            mode = 'w+'
            Path(keys_filename).parent.mkdir(parents=True, exist_ok=True)
        
        self.dstore_keys = np.memmap(keys_filename, dtype=np.float16, mode=mode, shape=(self.dstore_size, self.dimension))
        self.dstore_vals = np.memmap(vals_filename, dtype=np.int32, mode=mode, shape=(self.dstore_size, 1))

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            raise ValueError('labels must be provided when saving a datastore. Are you using --predict_with_generate by mistake? If so, disable it')
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        shift = 0 if self.is_encoder_decoder else 1
        captured_keys = self.activation_capturer.captured
        if shift == 1:
            captured_keys = captured_keys[:, :-shift]
        captured_keys = captured_keys.flatten(0, 1) # (batch * time, dim)
        captured_values = self.labels[:, shift:].flatten(0, 1) # (batch * time)

        nonpad_mask = captured_values != -100
        keys = captured_keys[nonpad_mask]
        values = captured_values[nonpad_mask]

        batch_time_size = keys.shape[0]
        # if shape[0] == args.tokens_per_sample:
        if self.dstore_idx + batch_time_size > self.dstore_size:
            batch_time_size = max(self.dstore_size - self.dstore_idx, 0)
            keys = keys[:batch_time_size]
            values = values[:batch_time_size]
        try:
            self.dstore_keys[self.dstore_idx:(batch_time_size + self.dstore_idx)] = keys.cpu().numpy().astype(np.float16)
            self.dstore_vals[self.dstore_idx:(batch_time_size + self.dstore_idx)] = values.unsqueeze(-1).cpu().numpy().astype(np.int32)
        except ValueError as ex:
            logger.error(f'Error saving datastore with mode {self.dstore_keys.mode}, did you try to save an already existing datastore?')
            logger.error(f'Delete the files {self.dstore_keys.filename} and {self.dstore_vals.filename} and try again')
            raise ex

        self.dstore_idx += batch_time_size
        
        return output

    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)
    
    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    def build_index(self, num_keys_to_add_at_a_time=1000000, 
            ncentroids=4096, seed=1, code_size=64, probe=32):
        logger.info('Building index')
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension) 
        
        # Initialize faiss index
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFPQ(quantizer, self.dimension,
            ncentroids, code_size, 8)
        index.nprobe = probe

        logger.info('Training Index')
        np.random.seed(seed)
        random_sample = np.random.choice(np.arange(self.dstore_vals.shape[0]), size=[min(1000000, self.dstore_vals.shape[0])], replace=False)
        start = time.time()
        # Faiss does not handle adding keys in fp16 as of writing this.
        index.train(self.dstore_keys[random_sample].astype(np.float32))
        logger.info(f'Training took {time.time() - start} s')

        logger.info('Adding Keys')
        # index = faiss.read_index(f'{index_name}.trained')
        start = 0
        start_time = time.time()
        while start < self.dstore_size:
            end = min(self.dstore_size, start + num_keys_to_add_at_a_time)
            to_add = self.dstore_keys[start:end].copy()
            index.add_with_ids(torch.tensor(to_add.astype(np.float32)), torch.arange(start, end))
            start += num_keys_to_add_at_a_time

            if (start % 1000000) == 0:
                logger.info(f'Added {start} tokens so far')
                logger.info(f'Writing Index {start}')
                faiss.write_index(index, f'{index_name}')

        logger.info(f'Adding total {start} keys')
        logger.info(f'Adding took {time.time() - start_time} s')
        logger.info(f'Writing Index to {index_name}')
        start = time.time()
        faiss.write_index(index, f'{index_name}')
        logger.info(f'Writing index took {time.time() - start} s')
        
    def get_metrics(self):
        return {}

class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None

    
    def forward(self, module, input, output):
        if self.capture_input:
            self.captured = input[0].detach()
        else:
            self.captured = output.detach()


def get_dstore_path(dstore_dir, model_type, dstore_size, dimension):
    return f'{dstore_dir}/dstore_{model_type}_{dstore_size}_{dimension}'

def get_index_path(dstore_dir, model_type, dstore_size, dimension):
    return f'{dstore_dir}/index_{model_type}_{dstore_size}_{dimension}.indexed'