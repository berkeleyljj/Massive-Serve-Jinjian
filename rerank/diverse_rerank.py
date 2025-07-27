import os
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from types import SimpleNamespace
from rerank.exact_rerank import embed_passages
from normalize_text import normalize as nm

os.environ["TOKENIZERS_PARALLELISM"] = "true"
CACHE_DIR = "./embedding_cache"

def diverse_rerank_topk(raw_passages, query_encoder, lambda_val=0.5):
    logging.info(f"Doing diverse reranking for {len(raw_passages)} total evaluation samples...")
    raw_query = raw_passages[0]['raw_query']
    ctxs = [nm(p["text"].lower()) for p in raw_passages]
    ctx_ids = [p["passage_id"] for p in raw_passages]

    args = SimpleNamespace(
        per_gpu_batch_size=128,
        get=lambda k, default=None: default 
    )

    text_to_embeddings = embed_passages(
        args=args,
        raw_query=raw_query,
        texts=ctxs,
        passage_ids=ctx_ids,
        model=query_encoder
    )

    query_emb = text_to_embeddings[raw_query].reshape(1, -1)
    ctx_embeddings = [text_to_embeddings[t] for t in ctxs]
    ctx_embeddings = np.vstack(ctx_embeddings)

    selected = []
    selected_idxs = []

    candidate_idxs = list(range(len(ctxs)))

    for _ in range(len(ctxs)):
        best_score = -np.inf
        best_idx = -1
        for idx in candidate_idxs:
            p_i = ctx_embeddings[idx].reshape(1, -1)
            rel_score = float(np.dot(query_emb, p_i.T))

            div_score = 0.0
            for j in selected_idxs:
                p_j = ctx_embeddings[j].reshape(1, -1)
                div_score += float(np.dot(p_i, p_j.T))

            score = rel_score - lambda_val * div_score

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1:
            break

        selected.append(raw_passages[best_idx])
        selected_idxs.append(best_idx)
        candidate_idxs.remove(best_idx)

    return selected
