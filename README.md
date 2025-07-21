## Introduction

CompactDS is a diverse, high-quality, web-scale datastore that achieves high retrieval accuracy and subsecond latency on a single-node deployment, making it suitable for academic use. Its core design combines a compact set of high-quality, diverse data sources with in-memory approximate nearest neighbor (ANN) retrieval and on-disk exact search. We release CompactDS and our retrieval pipeline as a fully reproducible alternative to commercial search, supporting future research exploring retrieval-based AI systems. Check out our paper, [Frustratingly Simple Retrieval Improves
Challenging, Reasoning-Intensive Benchmarks](http://arxiv.org/abs/2507.01297), for full details.  

Due to data sensitivity issues, we release a version of CompactDS **excluding Educational Text and Books**. We also use an \# subquantizer of 64 (instead of 256 used in the paper) to build an 102GB ANN index. The full collection of the released assets is as following:
- [CompactDS-102GB ](https://huggingface.co/datasets/alrope/CompactDS-102GB) (this dataset): the Faiss IVFPQ index and chuncked passages.
- [CompactDS-102GB-raw-text](https://huggingface.co/datasets/alrope/CompactDS-102GB-raw-text): the raw text data from 10 data sources used to build compactds.
- [CompactDS-102GB-queries](https://huggingface.co/datasets/alrope/CompactDS-102GB-queries): the queries from the five datasets-MMLU, MMLU Pro, AGI Eval, GPQA, and MATH-that we report the RAG results with in the paper.
- [CompactDS-102GB-retrieval-results](https://huggingface.co/datasets/alrope/CompactDS-102GB-retrieval-results): the top-1000 retrieved passages from CompactDS using the queries from the five datasets-MMLU, MMLU Pro, AGI Eval, GPQA, and MATH-that we report the RAG results with in the paper.

For download / usage of these assets, refer to [compactds-retrieval](https://github.com/Alrope123/compactds-retrieval).

For RAG evaluation with the retrieved documents, refer to [compactds-eval](https://github.com/michaelduan8/compactds-eval).

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