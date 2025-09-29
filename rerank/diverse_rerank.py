import os
import numpy as np
import logging
import time
import torch
from sklearn.metrics.pairwise import cosine_similarity
from types import SimpleNamespace
from rerank.exact_rerank import embed_passages
from normalize_text import normalize as nm

os.environ["TOKENIZERS_PARALLELISM"] = "true"
CACHE_DIR = "./embedding_cache"

def diverse_rerank_topk(raw_passages, query_encoder, lambda_val):
    total_start = time.time()
    logging.info(f"Doing diverse reranking for {len(raw_passages)} total evaluation samples using lambda {lambda_val}...")

    step_start = time.time()
    raw_query = raw_passages[0]['raw_query']
    ctxs = [nm(p["text"].lower()) for p in raw_passages]
    ctx_ids = [p["passage_id"] for p in raw_passages]
    logging.info(f"Step 1: Normalization done in {time.time() - step_start:.2f}s")

    args = SimpleNamespace(
        per_gpu_batch_size=128,
        get=lambda k, default=None: default 
    )

    step_start = time.time()
    text_to_embeddings = embed_passages(
        args=args,
        raw_query=raw_query,
        texts=ctxs,
        passage_ids=ctx_ids,
        model=query_encoder
    )
    logging.info(f"Step 2: Embedding done in {time.time() - step_start:.2f}s")

    step_start = time.time()
    ctx_embeddings = [text_to_embeddings[t] for t in ctxs]
    ctx_embeddings = np.vstack(ctx_embeddings)
    logging.info(f"Step 3: Embedding vector preparation done in {time.time() - step_start:.2f}s")

    # === Classic MMR (no |S| normalization; max redundancy; λ/(1−λ) weighting) ===
    step_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx_embeddings = torch.tensor(ctx_embeddings, dtype=torch.float32, device=device)  # (N, D)
    N = ctx_embeddings.shape[0]

    # Candidate-to-candidate cosine (redundancy)
    norms = torch.norm(ctx_embeddings, dim=1, keepdim=True).clamp_min(1e-12)
    cosine_sim_matrix = (ctx_embeddings @ ctx_embeddings.T) / (norms @ norms.T)

    # Determine relevance vector:
    # - Prefer precomputed exact cosine scores if present
    # - Else use original retrieval score if available
    # - Else fall back to cosine similarity to the query (computed on-the-fly)
    if "exact score" in raw_passages[0]:
        rel_scores = torch.tensor(
            [p["exact score"] for p in raw_passages], dtype=torch.float32, device=device
        )
        relevance_source = "exact"
    elif "retrieval score" in raw_passages[0]:
        rel_scores = torch.tensor(
            [p.get("retrieval score", 0.0) for p in raw_passages], dtype=torch.float32, device=device
        )
        relevance_source = "retrieval"

    # Clamp λ to [0, 1]
    lambda_mult = float(lambda_val)
    lambda_mult = max(0.0, min(1.0, lambda_mult))

    selected = []
    selected_idxs = []
    candidate_idxs = list(range(N))

    if N > 0:
        first_idx = torch.argmax(rel_scores).item()
        selected.append(raw_passages[first_idx])
        selected_idxs.append(first_idx)
        candidate_idxs.remove(first_idx)

    # Greedy MMR selection
    for _ in range(N - 1):
        if not candidate_idxs:
            break

        # Max similarity to already-selected items (redundancy term)
        if selected_idxs:
            pen_mat = cosine_sim_matrix[candidate_idxs][:, selected_idxs]  
            redundant = pen_mat.max(dim=1).values                           
        else:
            redundant = torch.zeros(len(candidate_idxs), device=device)

        # MMR score: λ * relevance - (1 - λ) * max redundancy
        scores = lambda_mult * rel_scores[candidate_idxs] - (1.0 - lambda_mult) * redundant

        best_idx_local = torch.argmax(scores).item()
        best_idx = candidate_idxs[best_idx_local]

        selected.append(raw_passages[best_idx])
        selected_idxs.append(best_idx)
        candidate_idxs.remove(best_idx)

    logging.info(f"Step 4: Greedy selection loop done in {time.time() - step_start:.2f}s")
    logging.info(f"Total rerank time: {time.time() - total_start:.2f}s")

    return selected