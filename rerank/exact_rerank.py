import os
import json
import pickle as pkl
import logging
from tqdm import tqdm
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from rerank import contriever
from normalize_text import normalize as nm
os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = 'cuda' if torch.cuda.is_available()  else 'cpu'


def embed_queries(args, queries, model, cached_embeddings={}):
    logging.info(f"Embedding {len(queries)} text after caching...")
    logging.info(f"queries length is {len(queries)}\n")
    queries = [q for q in queries if q not in cached_embeddings]

    # if "GritLM" in model_name_or_path:
    # Defaults to GritLM
    all_question = []
    for k, q in enumerate(queries):
        if args.lowercase:
            q = q.lower()
        if args.normalize_text:
            q = nm(q)
        all_question.append(q)
    embeddings = model.encode(all_question, batch_size=min(128, args.per_gpu_batch_size), instruction="<|embed|>\n")  # sentence-transformer has extra memory overhead and can only support a smaller batch size
    
    logging.info(f"Questions embeddings shape: {embeddings.shape}")
    for query, embedding in zip(queries, embeddings):
        if np.isnan(embedding).any():
            logging.info(f"[WARNING] Skipping query due to NaN in embedding: {q}")
            continue
        cached_embeddings[query] = embedding
    return cached_embeddings


# def get_search_output_path(cfg, index_shard_ids=None):
#     eval_args = cfg.evaluation
#     if index_shard_ids:
#         shards_postfix = '_'.join([str(shard_id) for shard_id in index_shard_ids])
#         output_dir = os.path.join(eval_args.eval_output_dir, shards_postfix)
#     else:
#         output_dir = eval_args.eval_output_dir
    
#     index_type = cfg.datastore.index.index_type
#     if "IVF" in index_type:
#         postfix = f"_{index_type}.{cfg.datastore.index.ncentroids}"
#         if "PQ" in index_type:
#             postfix = f"{postfix}.{cfg.datastore.index.n_subquantizers}"
#         postfix = f"{postfix}.{cfg.datastore.index.probe}"
#     else:
#         postfix = ""

#     output_path = os.path.join(output_dir + postfix, os.path.basename(eval_args.data.eval_data).replace('.jsonl', '_retrieved_results.jsonl'))
#     return output_path


# Defaults to memory reranking with live passages
def exact_rerank_topk(raw_passages):
    from gritlm import GritLM
    query_tokenizer  = None
    query_encoder = GritLM("GritLM/GritLM-7B", torch_dtype="auto", mode="embedding")
    query_encoder.eval()
    query_encoder = query_encoder.to(device)

    logging.info(f"Doing exact reranking for {len(raw_passages)} total evaluation samples...")
    
    # ctxs = []
    # queries = []
    # for psg_group in raw_passages:
    #     for passage in psg_group:
    #         queries.append(raw_query)
    #         ctxs.extend([passage['text']])

    raw_query = raw_passages[0]['raw_query']

    ctxs = [p["text"] for p in raw_passages]

    from types import SimpleNamespace

    args = SimpleNamespace(
        lowercase=True,
        normalize_text=True,
        per_gpu_batch_size=64,
        get=lambda k, default=None: default 
    )
    text_to_embeddings = embed_queries(
        args=args,
        queries=[raw_query] + ctxs,
        model=query_encoder,
        cached_embeddings={}
    )
    
    print("Calculating the exact similarity scores...")


    query_embedding = text_to_embeddings[raw_query].reshape(1, -1)
    ctx_embeddings = [text_to_embeddings[text] for text in ctxs]
    ctxs_embedding = np.vstack(ctx_embeddings)

    print("Query embedding shape:", query_embedding.shape)
    print("Ctxs embedding shape:", ctxs_embedding.shape)
    print("Query dtype:", query_embedding.dtype)
    print("Ctxs dtype:", ctxs_embedding.dtype)



    if np.isnan(query_embedding).any():
        print("[ERROR] query_embedding contains NaN")
    if np.isnan(ctxs_embedding).any():
        print("[ERROR] ctxs_embedding contains NaN")

    similarities = cosine_similarity(query_embedding, ctxs_embedding)
    print("Similarity calculation finished successfully!")

    for i, p in enumerate(raw_passages):
        p["exact score"] = float(similarities[0][i]) if not np.isnan(similarities[0][i]) else -1.0

    raw_passages.sort(key=lambda p: p["exact score"], reverse=True)
        

    return raw_passages
    


def normalize_text(text):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(lower(text)))

# def safe_write_jsonl(data, output_file):
#     success = False
#     try:
#         with open(output_file, 'w') as fout:
#             for ex in data:
#                 fout.write(json.dumps(ex) + "\n")
#             success = True
#         logging.info(f"Saved results to {output_file}")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#     # If an error was raised, and success is still False, delete the file
#         if not success and os.path.exists(output_file):
#             os.remove(output_file)
#             print(f"File '{output_file}' has been deleted due to an error.")


def search_topk(cfg):
    exact_rerank_topk(cfg)
