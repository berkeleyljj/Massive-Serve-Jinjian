## Introduction

This repository contains the Compact‑DS Dive Public API and server code. It exposes a production‑ready Flask service for retrieval‑augmented generation (RAG) backed by a billion‑scale FAISS IVFPQ index. The server provides adjustable settings and search modes at low-latency. A small CLI helps download/prepare indices and start the server, and an example domain (`index_dev/`) shows the expected data layout. Please refer to the Quickstart below to set up.

### What this repo provides

- API server: a Flask service exposing `POST /search` and `POST /vote` with optional exact/diverse reranking and on‑disk vote storage.
  - Server entry: `massive_serve/api/serve.py`
  - Endpoints: `/search`, `/vote`, plus small utilities (`/current_search`, `/queue_size`)
- Retrieval backend: FAISS IVFPQ index with passage lookup via position arrays.
  - Core code: `massive_serve/src/indicies/ivf_pq.py`, `massive_serve/src/search.py`
- CLI tooling: start the server and prepare/download large index files.
  - CLI entry: `python -m massive_serve.cli`
  - Commands: `serve`, `upload-data` (HF), helpers in `massive_serve/data_utils.py`
- Example domain for local dev: `index_dev/` with `config.json`, `index/`, and `passages/` layout used in the Quickstart below.

### Expected data layout (under DATASTORE_PATH)

```
<DATASTORE_PATH>/
  <domain_name>/
    config.json              # loader config (encoder, nprobe, index filename, etc.)
    index/                   # a single merged FAISS file (*.faiss)
    passages/                # *.jsonl shards for text lookup
```

## Quickstart (Local Dev)

### 1) Download the dataset/index from Hugging Face

- Choose a local data root (DATASTORE_PATH). Example:
```bash
export DATASTORE_PATH=/home/ubuntu/massive-serve-dev
```

- Download the dataset into `$DATASTORE_PATH/<domain_name>` (example uses `index_dev`):
```bash
huggingface-cli download <ORG_OR_USER>/<DATASET_REPO> \
  --repo-type dataset \
  --local-dir $DATASTORE_PATH/index_dev
```

Notes:
- The directory should include an `index/` directory with a FAISS index and a `passages/` directory with `.jsonl` files.
- If your index is uploaded in split/chunked form, see Step 3 to combine shards.

### 2) Build the position mapping arrays (passage offsets)

The server looks up passage text by FAISS index id using position mapping arrays. Generate them once from your `passages/` directory:

- Open `build_arr.py` and set `INPUT_DIR` to your passages folder, e.g.:
```python
INPUT_DIR = "/home/ubuntu/massive-serve-dev/index_dev/passages"
```

- Then run from the repo root:
```bash
python build_arr.py
```

This writes three files next to the script (and the server expects them under `index_dev/` as configured by the code):
- `index_dev/position_array.npy`
- `index_dev/filename_index_array.npy`
- `index_dev/filename_list.npy`

### 3) Combine FAISS index shards (if present)

If your `index/` folder contains split parts, combine them into a single `.faiss` file before serving.

- Simple shard set (concatenate all `...faiss_**` parts in order; do NOT include the `.meta` file):
```bash
cd $DATASTORE_PATH/index_dev/index
# Example names: index_IVFPQ.100000000.768.65536.64.faiss_aa, ..._ab, ..._ac, ...
cat $(ls index_IVFPQ.100000000.768.65536.64.faiss_* | sort) > index_IVFPQ.100000000.768.65536.64.faiss
```

- If your split format uses a `.split/` directory with `.chunk_###` files and a JSON metadata (e.g., `index_metadata.json`), you can also combine via the helper (optional alternative):
```bash
python -m massive_serve.data_utils --mode combine --data_dir $DATASTORE_PATH/index_dev
```

After combining, ensure there is exactly one `.faiss` file in `index/` (e.g., `index/index_full.faiss`).

### 4) Launch the API server

Use `index_dev` as the domain name (provided by `index_dev/config.json`):
```bash
DATASTORE_PATH=/home/ubuntu/massive-serve-dev \
python -m massive_serve.cli serve --domain_name index_dev
```

By default the server starts at port `30888` and exposes `/search` and `/vote` endpoints.

### 5) Test the API

Basic single query:
```bash
curl -X POST http://compactds.duckdns.org:30888/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me more about Albert Einstein", "n_docs": 5, "nprobe": 32}'
```

Batch queries:
```bash
curl -X POST http://compactds.duckdns.org:30888/search \
  -H "Content-Type: application/json" \
  -d '{"queries": ["quantum computing", "Who is Nikola Tesla", "AI ethics"], "n_docs": 2}'
```

## API

For the full reference and examples, see `API_DOCUMENTATION.md`. Below is a brief, readability-first summary.

- Endpoint: `POST /search` (JSON)
- Request: use either `query: string` or `queries: string[]`
- Common options: `n_docs`, `nprobe`, `exact_search`, `diverse_search`, `lambda`

Minimal example:
```bash
curl -X POST http://compactds.duckdns.org:30888/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "n_docs": 5, "nprobe": 32}'
```

Votes: `POST /vote` to record relevance feedback. See `VOTES_DOCUMENTATION.md` for storage details and schema.

## Statistics
| Source                        | # Documents (M) | # Words (B) | # Passages (M) | Size of Raw Data (G) | Size of Index (G) |
|------------------------------|----------------:|------------:|---------------:|---------------:|---------------:|
| Math                         |             6.4 |         7.8 |           33.7 |           15.8 |            8.3 |
| High-quality CC              |           407.3 |       171.9 |          895.1 |          413.1 |          220.1 |
| Academic Papers (PeS2o)      |             7.8 |        49.7 |          198.1 |          104.4 |           49.2 |
| Wikipedia (Redpajama)        |            29.8 |        10.8 |           60.5 |           27.1 |           14.9 |
| Reddit                       |            47.3 |         7.5 |           54.1 |           21.7 |           13.3 |
| Wikipedia (DPR)              |            21.0 |         2.2 |           21.0 |            4.6 |            5.2 |
| Stack Exchange               |            29.8 |         9.2 |           50.5 |           22.6 |           12.4 |
| Github                       |            28.8 |        17.1 |           84.3 |           44.9 |           20.7 |
| PubMed                       |            58.6 |         3.6 |           60.4 |            8.9 |           14.9 |
| Academic Papers (Arxiv)      |             1.6 |        11.3 |           44.9 |           24.0 |           11. 0|
| **Full CompactDS**           |       **638.4** |   **291.2** |    **1,502.5** |       **687.1**|          102.1 |

## Performance
We compare the performance of this released version of CompactDS with the two indices reported in the papers: 

**Table: Comparison between the RAG performance of three versions of CompactDS at k=10. Best results are in bold.**
|                  | Index Size | MMLU | MMLU Pro | AGI Eval | MATH | GPQA | **AVG** |
|------------------|--------|------------|------|----------|----------|------|------|
| No Retrieval    |  - | 68.9 | 39.8     | 56.2     | 46.9 | 29.9 | 48.3    |
| *All 12 data sources with # Sub quantizer = 64*|            |      |          |          |      |         |
| ANN Only | 125GB      | 75.0 | 50.9     | 57.9     | 53.0 | 34.8 | 54.3    |
| ANN + Exact Search      | 125GB      | 74.4 | 51.7     | **59.2** | 54.6 | 30.8 | 54.1    |
| *All 12 data sources with # Sub quantizer = 256* |    |       |      |          |          |      |       |
| ANN Only   | 456GB      | **75.3** | 50.1 | 57.4     | 51.9 | **36.4** | 54.2 |
| ANN + Exact Search  | 456GB      | **75.3** | **53.1** | 58.9 | **55.9** | 32.4 | **55.1** |
| ***Excluding Educational Text/Books with # Sub quantizer = 64 (This datastore)*** |    |       |      |          |          |      |       |
| ANN Only   | 102GB      | 73.6 | 46.8 | 57.5     | 51.6 | 30.8 | 52.0 |
| ANN + Exact Search  | 102GB      | 74.0 | 48.6 | 57.4 | 54.0 | 35.7 | 53.9 |


## Reproducibility
As we adopt Inverted File Product Quantization (IVFPQ) index that approximate the nearest neighbor search, potential discrepancy could exist among resulting indices from different training runs. To provide reference for reproducibility, we build Flat indices for Math and PeS2o that performs exact nearest neighbor search. We measure Recall@10 on GPQA for the IVFPQ indices using the retrieval results from Flat as the ground truth, and obtained 0.82 and 0.80. We also find that generation with vllm brings more discrepancy in results across machines, so we recommand using Hunggingface library for generation.

|                   | STEM | Human. | Social | Others | MMLU Pro | AGI Eval | MATH | Phys | Bio  | Chem | AVG   |
|-------------------|------|--------|--------|--------|----------|----------|------|------|------|------|-------|
| **No Retrieval**  | 60.2 | 72.0   | 78.7   | 68.9   | 39.8     | 56.2     | 46.9 | 26.7 | 47.4 | 25.7 | 48.3  |
| **_Math_**        ||||||||||||
| IVFPQ            | 64.2 | 73.5   | 80.3   | 70.1   | 43.4     | 57.4     | 50.6 | 32.1 | 50.0 | 22.4 | 50.7  |
| Flat             | 63.5 | 73.1   | 80.4   | 70.6   | 44.1     | 58.0     | 52.7 | 31.6 | 47.4 | 26.8 | 51.6  |
| **_PeS2o_**       ||||||||||||
| IVFPQ            | 58.8 | 73.6   | 79.8   | 70.0   | 42.1     | 55.9     | 45.7 | 30.5 | 53.8 | 29.0 | 49.4  |
| Flat             | 59.4 | 73.5   | 80.2   | 69.8   | 42.3     | 55.5     | 45.1 | 32.6 | 52.6 | 28.4 | 49.4  |


## Citation
```
@article{lyu2025compactds,
  title={Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks},
  author={Xinxi Lyu and Michael Duan and Rulin Shao and Pang Wei Koh and Sewon Min}
  journal={arXiv preprint arXiv:2507.01297},
  year={2025}
}
```