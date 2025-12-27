# retomaton_wrapper.py
# ============================================================
# GPU-SAFE BF16 RETOMATON WRAPPER
# Separate module depending only on knnlm.KNNWrapper
# ============================================================

import os
import glob
import pickle
import logging
import numpy as np
import torch
from torch import nn
from pathlib import Path
from itertools import zip_longest
from collections import defaultdict

from tqdm import tqdm

from knnlm_gpu import KNNWrapper, get_dstore_path
import faiss
from faiss import IndexFlatL2
import scipy.sparse as sp

logger = logging.getLogger(__name__)
logger.setLevel(20)


# ============================================================
# Retomaton Wrapper (BF16 + GPU optimized)
# ============================================================

class RetomatonWrapper(KNNWrapper):
    """
    GPU-safe bf16 Retomaton implementation.
    Fully compatible with your patched KNNWrapper.
    """

    def __init__(self, no_pointer=False, min_knns=1, max_knns=1024, members=None, **kwargs):
        super().__init__(**kwargs)

        self.no_pointer = no_pointer
        self.min_knns = min_knns
        self.max_knns = max_knns

        # ---- Load clustering if available ----
        if members is None:
            files = glob.glob(f"{self.dstore_dir}/members*")
            if len(files) == 0:
                logger.info(f"[Retomaton] No members file found — clustering disabled.")
                self.members = None
                self.cluster = None
            else:
                members = files[0]
                logger.info(f"[Retomaton] Using cluster file: {members}")
                with open(members, "rb") as f:
                    self.members = pickle.load(f)
                idxs = np.nonzero(self.members[np.arange(self.members.shape[0])])
                self.cluster = torch.zeros((self.dstore_size,), dtype=torch.int32, device=self.device)
                self.cluster[idxs[1]] = torch.from_numpy(idxs[0]).to(self.device)
        else:
            logger.info(f"[Retomaton] Using explicit cluster file: {members}")
            with open(members, "rb") as f:
                self.members = pickle.load(f)
            idxs = np.nonzero(self.members[np.arange(self.members.shape[0])])
            self.cluster = torch.zeros((self.dstore_size,), dtype=torch.int32, device=self.device)
            self.cluster[idxs[1]] = torch.from_numpy(idxs[0]).to(self.device)

        # Internal tracking for training-only pointer logic
        self.generate_cur_knns = torch.empty(0, dtype=torch.int64, device=self.device)
        self.generate_cur_dists = torch.empty(0, dtype=torch.bfloat16, device=self.device)
        self.no_lookup_counter_history = []


    # ========================================================
    # Post-forward (training mode only)
    # ========================================================
    def post_forward_hook(self, module, input, output):
        shift = 0 if self.is_encoder_decoder else 1

        # ---------- GENERATION MODE ---------- #
        if self.labels is None:
            # fallback to KNN-LM behaviour during generation
            return super().post_forward_hook(module, input, output)

        # ---------- TRAINING MODE ---------- #
        lm_logits = torch.nn.functional.log_softmax(output.to(torch.bfloat16), dim=-1)
        queries = self.activation_capturer.captured  # (B,T,D)

        shifted_labels = self.labels[:, shift:]
        nonpad_mask = torch.cat(
            [
                shifted_labels != -100,
                torch.zeros([self.labels.shape[0], shift], dtype=torch.bool, device=self.device),
            ],
            dim=-1,
        )

        captured_labels = shifted_labels[shifted_labels != -100]  # (N,)
        lm_logits = lm_logits[nonpad_mask]                        # (N,V)
        queries = queries[nonpad_mask]                            # (N,D)

        all_knn_probs = []
        cur_knns = torch.empty(0, dtype=torch.int64, device=self.device)
        cur_dists = torch.empty(0, dtype=torch.bfloat16, device=self.device)
        no_lookup_counter = 0

        for q, label in zip_longest(queries, captured_labels):
            perform_search = False
            pointers = cur_knns + 1
            extended_pointers = None

            # ---------- Pointer heuristic ---------- #
            if self.no_pointer or cur_knns.numel() < self.min_knns:
                perform_search = True
                self.no_lookup_counter_history.append(no_lookup_counter)
                no_lookup_counter = 0
            else:
                no_lookup_counter += 1

            # ---------- Pointer extension ---------- #
            if not self.no_pointer:
                if pointers.numel() >= self.max_knns:
                    extended_pointers = pointers[: self.max_knns]
                else:
                    extended_pointers = self.extend_pointers_using_clusters(pointers)

            # ---------- Get KNN probs ---------- #
            cur_knn_log_prob, knns, neg_dists, vals_at_knns = self.get_knn_log_prob(
                q,
                pointers=extended_pointers,
                perform_search=perform_search,
            )
            all_knn_probs.append(cur_knn_log_prob)

            # ---------- Update pointers ---------- #
            if not self.no_pointer and label is not None and knns.numel() > 0:
                correct = (vals_at_knns == label) & (knns < self.dstore_size - 1)
                cur_knns = knns[correct]
                cur_dists = (-neg_dists)[correct]
                if cur_knns.numel() > 0:
                    order = cur_dists.argsort(descending=True)
                    cur_knns = cur_knns[order]
                    cur_dists = cur_dists[order]

        if len(all_knn_probs) == 0:
            return output

        stacked_knn_probs = torch.stack(all_knn_probs, dim=0)
        interp = KNNWrapper.interpolate(stacked_knn_probs, lm_logits, self.lmbda, device=self.device)
        output[nonpad_mask] = interp.to(output.dtype)
        return output


    # ========================================================
    # Core KNN-lookup logic (bf16-safe)
    # ========================================================
    def get_knn_log_prob(self, query, pointers, perform_search):
        pointer_dists = torch.empty(0, dtype=torch.bfloat16, device=self.device)

        # ---- Reuse pointers ----
        if pointers is not None and pointers.numel() > 0 and not self.recompute_dists:
            pointer_vectors = self.reconstruct_ids(pointers)
            pointer_dists = self.dist_func(query.to(torch.bfloat16), pointer_vectors).view(-1)

        # ---- FAISS search if needed ----
        if perform_search:
            dists, knns = self.get_knns(query.unsqueeze(0).to(torch.bfloat16))
            dists, knns = dists.squeeze(0), knns.squeeze(0)
            if pointers is not None and pointers.numel() > 0:
                knns = torch.cat([knns, pointers], dim=-1)
                dists = torch.cat([dists.to(self.device), pointer_dists], dim=-1)
        else:
            knns = pointers if pointers is not None else torch.empty(0, dtype=torch.int64, device=self.device)
            dists = pointer_dists

        # ---- Optional distance recomputation ----
        if self.recompute_dists and knns.numel() > 0:
            if isinstance(self.keys, np.memmap):
                keys_fp16 = torch.from_numpy(self.keys[knns.cpu().numpy()]).to(self.device)
            else:
                keys_fp16 = self.keys[knns]
            knn_vecs = keys_fp16.to(torch.bfloat16)
            dists = self.dist_func(query.to(torch.bfloat16), knn_vecs).view(-1)

        if knns.numel() == 0:
            blank = torch.full((self.vocab_size,), -10000.0, dtype=torch.bfloat16, device=self.device)
            return blank, knns, dists, torch.empty(0, dtype=torch.int64, device=self.device)

        neg_dists = -dists.to(torch.bfloat16)
        knn_log_probs, vals_at_knns = self.knns_to_log_prob(knns, neg_dists)
        return knn_log_probs, knns, neg_dists, vals_at_knns


    # ========================================================
    # Pointer expansion via clusters
    # ========================================================
    def extend_pointers_using_clusters(self, pointers):
        if self.members is None or self.cluster is None or pointers.numel() == 0:
            return pointers

        clusters, counts = torch.unique(self.cluster[pointers], return_counts=True)
        clusters = clusters[torch.argsort(-counts)]

        members = torch.from_numpy(
            np.nonzero(self.members[clusters.cpu().numpy()])[1]
        ).to(self.device)

        ext = torch.cat([pointers, members])
        return ext[: self.max_knns]


    # ========================================================
    # CPU reconstruct for BF16 distance recomputation
    # ========================================================
    def reconstruct_ids(self, ids):
        ids_np = ids.detach().cpu().numpy()
        vecs = [self.reconstruct_index.reconstruct(int(i)) for i in ids_np]
        arr = np.stack(vecs, axis=0)
        return torch.from_numpy(arr).to(self.device).to(torch.bfloat16)


    # ========================================================
    # Metrics
    # ========================================================
    def get_metrics(self):
        if len(self.no_lookup_counter_history) == 0:
            return {"lookups_saved": 0.0}
        total_skip = float(np.sum(self.no_lookup_counter_history))
        denom = total_skip + len(self.no_lookup_counter_history)
        return {"lookups_saved": total_skip / denom}


    def break_out(self):
        super().break_out()
        self.print_stats()


    def print_stats(self):
        if len(self.no_lookup_counter_history) == 0:
            return
        total_skip = float(np.sum(self.no_lookup_counter_history))
        denom = total_skip + len(self.no_lookup_counter_history)
        logger.info(f"[Retomaton] Lookups saved: {100 * total_skip / denom:.2f}%")


    # ========================================================
    # Datastore Clustering — GPU-safe version
    # ========================================================
    def cluster_dstore(self, num_clusters, sample_size, model, batch_size=500_000):

        prefix = get_dstore_path(self.dstore_dir, model.config.model_type, self.dstore_size, self.dimension)
        keys = np.memmap(
            f"{prefix}_keys.npy",
            dtype=np.float16,
            mode="r",
            shape=(self.dstore_size, self.dimension),
        )

        if sample_size > self.dstore_size:
            logger.info(f"[Retomaton] Using full datastore for clustering")
            to_cluster = keys[:]
        else:
            idxs = np.random.RandomState(1).choice(
                np.arange(self.dstore_size),
                size=sample_size,
                replace=False,
            )
            to_cluster = keys[idxs]

        to_cluster = to_cluster.astype(np.float32)

        logger.info(f"[Retomaton] Training KMeans ({num_clusters} clusters)...")
        kmeans = faiss.Kmeans(
            self.dimension,
            num_clusters,
            niter=20,
            verbose=True,
            gpu=torch.cuda.is_available(),
            seed=1,
        )
        kmeans.train(to_cluster)

        logger.info(f"[Retomaton] Assigning all keys to clusters...")
        index = IndexFlatL2(self.dimension)
        index.add(kmeans.centroids)

        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, 0, index, co)

        centroid_ids = []
        start = 0
        while start < self.dstore_size:
            end = min(self.dstore_size, start + batch_size)
            chunk = keys[start:end].astype(np.float32)
            _, key_i = index.search(torch.from_numpy(chunk), 1)
            centroid_ids.append(key_i.squeeze())
            start += batch_size

            if start % 1_000_000 == 0:
                logger.info(f"[Retomaton] Assigned {start} keys...")

        centroid_ids = np.concatenate(centroid_ids)

        logger.info(f"[Retomaton] Building sparse cluster->member matrix...")
        cluster_to_members = defaultdict(set)
        for key_i, c in tqdm(enumerate(centroid_ids), total=self.dstore_size):
            cluster_to_members[int(c)].add(key_i)

        row_ind = [k for k, v in cluster_to_members.items() for _ in range(len(v))]
        col_ind = [i for v in cluster_to_members.values() for i in v]
        members_sp = sp.csr_matrix(([1] * len(row_ind), (row_ind, col_ind)))

        members_path = f"{self.dstore_dir}/members_{model.config.model_type}_{self.dstore_size}_{self.dimension}_{sample_size}_{num_clusters}.pkl"
        with open(members_path, "wb") as f:
            pickle.dump(members_sp, f)

        logger.info(f"[Retomaton] Saved cluster members → {members_path}")
