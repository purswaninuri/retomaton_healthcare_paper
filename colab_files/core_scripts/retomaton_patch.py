# ------------------------------------------------------------
# Retomaton (Compatible with patched KNNWrapper for HF models)
# ------------------------------------------------------------
from collections import defaultdict
import os
import logging
import pickle
import time
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from pathlib import Path
import glob
from itertools import zip_longest

from tqdm import tqdm

from knnlm_patch import (
    KNNWrapper,
    get_dstore_path,
    extract_logits,     # <-- unified safe logit extractor
)

import faiss
import faiss.contrib.torch_utils
from faiss import IndexFlatL2
import scipy.sparse as sp

logger = logging.getLogger(__name__)
logger.setLevel(20)


# ============================================================
#                    RETOMATON WRAPPER
# ============================================================
class RetomatonWrapper(KNNWrapper):
    def __init__(self, no_pointer=False, min_knns=1, max_knns=1024, members=None, **kwargs):
        super().__init__(**kwargs)

        self.no_pointer = no_pointer
        self.min_knns = min_knns
        self.max_knns = max_knns

        # -------------------------------
        # Load cluster membership matrix
        # -------------------------------
        if members is None:
            available_member_files = glob.glob(f"{self.dstore_dir}/members*")
            if len(available_member_files) == 0:
                logger.info(f"No cluster member files found in {self.dstore_dir}. RetoMaton will run in pointer-only mode.")
                self.extend_pointers_using_clusters = lambda pointers: pointers
            else:
                members = available_member_files[0]
                logger.info(f"Using cluster member file: {members}")

        if members is not None:
            with open(members, "rb") as f:
                self.members = pickle.load(f)

            # Build reverse-lookup structure
            members_for_indices = np.nonzero(self.members[np.arange(self.members.shape[0])])
            self.cluster = torch.zeros((self.dstore_size,), dtype=torch.int32).to(self.device)
            self.cluster[members_for_indices[1]] = torch.from_numpy(members_for_indices[0]).to(self.device)

        self.generate_cur_knns = torch.tensor([], dtype=torch.int64)
        self.generate_cur_dists = torch.tensor([], dtype=torch.float32)
        self.no_lookup_counter_history = []

    # ============================================================
    #               MAIN HOOK: RETOMATON OPERATION
    # ============================================================
    def post_forward_hook(self, module, input, output):
        shift = 0 if self.is_encoder_decoder else 1

        # -----------------------------------------------------------
        # GENERATE MODE (beam search) fallback to KNN-LM (no pointers)
        # -----------------------------------------------------------
        if self.labels is None:
            return super().post_forward_hook(module, input, output)

        # -----------------------------------------------------------
        # Extract logits safely (works on all ModelOutput types)
        # -----------------------------------------------------------
        lm_logits = extract_logits(output)
        lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1)

        # -----------------------------------------------------------
        # Extract hidden states (queries)
        # -----------------------------------------------------------
        queries = self.activation_capturer.captured
        if isinstance(queries, (tuple, list)):
            queries = queries[0]

        # -----------------------------------------------------------
        # Build nonpad mask (same as KNNWrapper)
        # -----------------------------------------------------------
        shifted_labels = self.labels[:, shift:]
        nonpad_mask = torch.cat(
            [
                shifted_labels != -100,
                torch.zeros([self.labels.size(0), shift], dtype=torch.bool).to(self.device),
            ],
            dim=-1,
        )

        # filter
        captured_labels = shifted_labels[shifted_labels != -100]
        queries = queries[nonpad_mask]        # (N, dim)
        lm_logits = lm_logits[nonpad_mask]    # (N, vocab)

        # -----------------------------------------------------------
        #               RETOMATON POINTER UPDATE LOOP
        # -----------------------------------------------------------
        all_knn_probs = []

        cur_knns = torch.tensor([], dtype=torch.int64).to(self.device)
        cur_dists = torch.tensor([], dtype=torch.float32).to(self.device)
        no_lookup_counter = 0

        for timestep_query, label in zip_longest(queries, captured_labels):
            perform_search = False
            pointers = cur_knns + 1
            extended_pointers = None

            # Determine whether lookup is performed
            if self.no_pointer or cur_knns.numel() < self.min_knns:
                perform_search = True
                self.no_lookup_counter_history.append(no_lookup_counter)
                no_lookup_counter = 0
            else:
                no_lookup_counter += 1

            # Expand pointer list via cluster membership
            if not self.no_pointer:
                if pointers.numel() >= self.max_knns:
                    extended_pointers = pointers[: self.max_knns]
                else:
                    extended_pointers = self.extend_pointers_using_clusters(pointers)

            # ---------------------------------------------------
            # Query FAISS / compute pointers / compute log-probs
            # ---------------------------------------------------
            cur_knn_log_prob, knns, neg_dists, vals_at_knns = self.get_knn_log_prob(
                timestep_query,
                pointers=extended_pointers,
                perform_search=perform_search,
            )

            all_knn_probs.append(cur_knn_log_prob)

            # ---------------------------------------------------
            # Update pointers IF labels available
            # ---------------------------------------------------
            if (not self.no_pointer) and (label is not None):
                correct = (vals_at_knns == label) & (knns < self.dstore_size - 1)
                cur_knns = knns[correct]
                cur_dists = neg_dists[correct]
                # Sort by highest similarity
                cur_knns = cur_knns[cur_dists.argsort(descending=True)]

        # ============================================================
        #              INTERPOLATE & WRITE BACK LOGITS
        # ============================================================
        interpolated_scores = KNNWrapper.interpolate(
            torch.stack(all_knn_probs),
            lm_logits,
            self.lmbda,
        )

        # -----------------------------------------------------------
        # PATCH LOGITS BACK INTO ModelOutput (Phi-3/Gemma safe)
        # -----------------------------------------------------------
        if hasattr(output, "logits"):
            output.logits[nonpad_mask] = interpolated_scores
            return output
        else:
            output[nonpad_mask] = interpolated_scores
            return output

    # ============================================================
    # KNN / POINTER RETRIEVAL LOGIC
    # ============================================================
    def get_knn_log_prob(self, query, pointers, perform_search):
        pointer_dists = torch.tensor([], dtype=torch.float32).to(self.device)

        if pointers is not None and pointers.numel() > 0 and not self.recompute_dists:
            pointer_vectors = self.reconstruct_ids(pointers)
            pointer_dists = self.dist_func(query, pointer_vectors)

        if perform_search:
            dists, knns = self.get_knns(query.unsqueeze(0))
            dists, knns = dists.squeeze(0), knns.squeeze(0)
            if pointers is not None and pointers.numel() > 0:
                knns = torch.cat([knns, pointers])
                dists = torch.cat([dists, pointer_dists])
        else:
            knns = pointers
            dists = pointer_dists

        if self.recompute_dists:
            knn_vecs = torch.from_numpy(self.keys[knns]).to(self.device)
            dists = self.dist_func(query, knn_vecs)

        neg_dists = -dists
        knn_log_probs, vals_at_knns = self.knns_to_log_prob(knns, neg_dists)

        return knn_log_probs, knns, neg_dists, vals_at_knns

    # ============================================================
    def extend_pointers_using_clusters(self, pointers):
        if pointers is None or pointers.numel() == 0:
            return pointers

        clusters, cluster_counts = torch.unique(self.cluster[pointers], return_counts=True)
        clusters = clusters[torch.argsort(-cluster_counts)]

        members = torch.from_numpy(np.nonzero(self.members[clusters.cpu().numpy()])[1]).to(self.device)
        extended = torch.cat([pointers, members])

        if extended.numel() > self.max_knns:
            extended = extended[: self.max_knns]

        return extended

    # ============================================================
    def reconstruct_ids(self, ids):
        ids = ids.cpu().numpy()

        reconstruct_func = np.vectorize(lambda x: self.reconstruct_index.reconstruct(int(x)), otypes=[object])
        vectors = reconstruct_func(ids)
        vectors = np.stack(vectors).reshape(ids.shape + (self.dimension,))
        return torch.from_numpy(vectors).to(self.device)

    # ============================================================
    def get_metrics(self):
        return {
            "lookups_saved": np.sum(self.no_lookup_counter_history)
            / (np.sum(self.no_lookup_counter_history) + len(self.no_lookup_counter_history)),
        }

    def break_out(self):
        super().break_out()
        self.print_stats()

    def print_stats(self):
        if len(self.no_lookup_counter_history) > 0:
            saved = np.sum(self.no_lookup_counter_history) / (
                np.sum(self.no_lookup_counter_history) + len(self.no_lookup_counter_history)
            )
            logger.info(f"Lookups saved: {saved*100:.2f}%")

    # ============================================================
    # CLUSTER CONSTRUCTION (optional preprocessing step)
    # ============================================================
    def cluster_dstore(self, num_clusters, sample_size, model, batch_size=500000):
        keys_vals_prefix = get_dstore_path(self.dstore_dir, model.config.model_type, self.dstore_size, self.dimension)
        keys = np.memmap(
            f"{keys_vals_prefix}_keys.npy",
            dtype=np.float16,
            mode="r",
            shape=(self.dstore_size, self.dimension),
        )

        if sample_size > self.dstore_size:
            to_cluster = keys[:].astype(np.float32)
        else:
            idx = np.random.RandomState(1).choice(np.arange(self.dstore_size), size=sample_size, replace=False)
            to_cluster = keys[idx].astype(np.float32)

        logger.info(f"Clustering {sample_size} samples into {num_clusters} clusters.")
        kmeans = faiss.Kmeans(self.dimension, num_clusters, niter=20, verbose=True, gpu=True, seed=1)
        kmeans.train(to_cluster)

        logger.info("Assigning all datastore entries to clusters...")
        index = IndexFlatL2(self.dimension)
        index.add(kmeans.centroids)

        if torch.cuda.is_available():
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index, co)

        start = 0
        centroid_ids = []
        while start < self.dstore_size:
            end = min(self.dstore_size, start + batch_size)
            _, key_i = index.search(torch.from_numpy(keys[start:end].astype(np.float32)), 1)
            centroid_ids.append(key_i.squeeze())
            start += batch_size

        centroid_ids = np.concatenate(centroid_ids)
        cluster_to_members = defaultdict(set)

        for key_i, cluster in tqdm(enumerate(centroid_ids), total=self.dstore_size):
            cluster_to_members[cluster.item()].add(key_i)

        row_ind = [k for k, v in cluster_to_members.items() for _ in range(len(v))]
        col_ind = [i for ids in cluster_to_members.values() for i in ids]
        members_sp = sp.csr_matrix(([1] * len(row_ind), (row_ind, col_ind)))

        members_filename = get_members_path(
            self.dstore_dir,
            model.config.model_type,
            self.dstore_size,
            self.dimension,
            sample_size,
            num_clusters,
        )
        with open(members_filename, "wb") as f:
            pickle.dump(members_sp, f)

        logger.info(f"Cluster file written to {members_filename}")


def get_members_path(dstore_dir, model_type, dstore_size, dimension, sample_size, num_clusters):
    return f"{dstore_dir}/members_{model_type}_{dstore_size}_{dimension}_{sample_size}_{num_clusters}.pkl"
