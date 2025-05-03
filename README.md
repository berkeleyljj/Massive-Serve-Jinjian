# Massive Serve

A scalable search and retrieval system using FAISS indices.

## Environment Setup

### Using Conda (Recommended for GPU support)

1. Create a new conda environment:
```bash
conda env create -f conda-env.yml
conda activate massive-serve
```
To update the existing environment:
```bash
conda env update -n massive-serve -f conda-env.yml
```


Note: The pip installation will automatically choose the appropriate FAISS version (CPU or GPU) based on your system.

## Project Structure

- `src/indicies/`: Contains different index implementations
  - `ivf_flat.py`: IVF-Flat index implementation
  - `base.py`: Base indexer class
  - Other index implementations

## Usage

The system supports multiple types of indices:
- Flat index
- IVF-Flat index
- IVF-PQ index

Example usage:
```python
from src.indicies.base import Indexer

# Initialize the indexer with your configuration
indexer = Indexer(cfg)

# Search for similar passages
scores, passages, db_ids = indexer.search(query_embeddings, k=5)
```

## Requirements

- Python 3.8+
- CUDA support (optional, for GPU acceleration)
- See requirements.txt for full list of dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.