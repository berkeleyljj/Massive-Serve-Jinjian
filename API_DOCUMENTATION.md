# Compact-DS Search API Documentation

## 🚀 Overview

**Compact-DS** is a high-performance document search and retrieval system that combines approximate nearest neighbor (ANN) search with exact reranking capabilities. This demo showcases advanced search techniques including diverse result selection and configurable search parameters.

### Key Features
- **Fast ANN Search**: Efficient approximate search using IVF-PQ indexing
- **Exact Reranking**: Optional exact similarity computation for higher accuracy
- **Diverse Results**: Configurable diversity to avoid similar passages
- **Batched Queries**: Support for processing multiple queries simultaneously
- **Real-time Performance**: Optimized for low-latency search operations

---

## 📡 API Endpoint

```
POST http://192.222.59.156:30888/search
Content-Type: application/json
```

---

## 🔧 Parameters

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `query` | string | Your search query (for single queries) | `"machine learning algorithms"` |
| `queries` | array | Array of search queries (for batched queries) | `["query1", "query2", "query3"]` |

**Note**: Use either `query` (single) or `queries` (batched), not both.

### Optional Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_docs` | integer | 1 | 1-20 | Number of top passages to retrieve |
| `nprobe` | integer | 32 | 1, 8, 16, 32, 64, 128, 256 | Number of clusters to search (higher = more accurate but slower) |
| `exact_search` | boolean | false | true/false | Use exact search instead of ANN (more accurate but slower) |
| `diverse_search` | boolean | false | true/false | Use diverse search to avoid similar results |
| `lambda` | float | 0.5 | 0.0-1.0 | Diversity tradeoff parameter (higher = more diverse, lower = more relevant) |

### Special Parameters

| Parameter | Type | Description | Use Case |
|-----------|------|-------------|----------|
| `expand_index_id` | integer | ID of passage to expand | Get more context around a specific passage |
| `expand_offset` | integer | 1 | Expansion offset for passage expansion |

---

## 🎯 Parameter Details

### `nprobe` - Search Clusters
Controls the number of clusters searched during ANN retrieval:
- **Lower values (1-16)**: Faster but less accurate
- **Higher values (64-256)**: More accurate but slower
- **Recommended**: 32 for balanced performance, 64+ for high accuracy

### `exact_search` - Exact Reranking
When enabled, performs exact similarity computation instead of approximate search:
- **Benefits**: Higher accuracy, better relevance
- **Trade-offs**: Slower response time, higher computational cost
- **Use case**: When accuracy is critical over speed

### `diverse_search` - Diverse Results
Enables diversity-aware result selection to avoid similar passages:
- **Benefits**: More varied results, better coverage
- **Trade-offs**: May reduce relevance for some queries
- **Best with**: `lambda` parameter to control diversity vs relevance trade-off

### `lambda` - Diversity Trade-off
Controls the balance between relevance and diversity when `diverse_search` is enabled:
- **0.0**: Maximum relevance, no diversity
- **0.5**: Balanced approach (default)
- **1.0**: Maximum diversity, may reduce relevance

---

## 📝 Request Examples

### 1. Basic Single Query

```python
import requests
import json

url = "http://192.222.59.156:30888/search"
headers = {"Content-Type": "application/json"}

payload = {
    "query": "machine learning algorithms",
    "n_docs": 5,
    "nprobe": 32
}

response = requests.post(url, headers=headers, json=payload)
data = response.json()
print(f"Found {len(data['results']['passages'][0])} passages")
```

### 2. High-Accuracy Search

```python
payload = {
    "query": "neural network architecture",
    "n_docs": 3,
    "nprobe": 128,
    "exact_search": True
}
```

### 3. Diverse Search

```python
payload = {
    "query": "artificial intelligence applications",
    "n_docs": 5,
    "diverse_search": True,
    "lambda": 0.7
}
```

### 4. Batched Queries

```python
payload = {
    "queries": [
        "quantum computing",
        "blockchain technology",
        "computer vision"
    ],
    "n_docs": 3,
    "exact_search": True,
    "diverse_search": True,
    "lambda": 0.6
}
```

### 5. Passage Expansion

```python
# First, get a passage ID from a search
search_payload = {
    "query": "deep learning",
    "n_docs": 1
}
search_response = requests.post(url, headers=headers, json=search_payload)
search_data = search_response.json()
passage_id = search_data['results']['passages'][0][0]['index_id']

# Then expand that passage
expand_payload = {
    "query": "",
    "expand_index_id": passage_id,
    "expand_offset": 1
}
expand_response = requests.post(url, headers=headers, json=expand_payload)
```

---

## 📊 Response Format

### Single Query Response

```json
{
  "message": "Search completed for 'machine learning' from demo",
  "query": "machine learning",
  "n_docs": 5,
  "nprobe": 32,
  "results": {
    "scores": [[0.85, 0.82, 0.79, 0.76, 0.73]],
    "passages": [[
      {
        "text": "Machine learning is a subset of artificial intelligence...",
        "source": "c4_dclm_mixed",
        "index_id": 123456789,
        "passage_id": "passage_123"
      },
      // ... more passages
    ]]
  }
}
```

### Batched Query Response

```json
{
  "message": "Search completed for batched queries from demo",
  "query": ["query1", "query2", "query3"],
  "n_docs": 3,
  "nprobe": 32,
  "results": {
    "scores": [
      [0.85, 0.82, 0.79],  // Scores for query1
      [0.88, 0.85, 0.81],  // Scores for query2
      [0.83, 0.80, 0.77]   // Scores for query3
    ],
    "passages": [
      [/* passages for query1 */],
      [/* passages for query2 */],
      [/* passages for query3 */]
    ]
  }
}
```

---

## 🎛️ Parameter Optimization Guide

### For Speed
```python
payload = {
    "query": "your query",
    "nprobe": 16,           # Lower nprobe
    "exact_search": False,  # Disable exact search
    "diverse_search": False # Disable diverse search
}
```

### For Accuracy
```python
payload = {
    "query": "your query",
    "nprobe": 128,          # Higher nprobe
    "exact_search": True,   # Enable exact search
    "diverse_search": False # Keep diverse search off for pure accuracy
}
```

### For Diversity
```python
payload = {
    "query": "your query",
    "nprobe": 64,           # Moderate nprobe
    "exact_search": False,  # Optional: can be enabled
    "diverse_search": True, # Enable diverse search
    "lambda": 0.7          # Higher lambda for more diversity
}
```

### For Balanced Performance
```python
payload = {
    "query": "your query",
    "nprobe": 32,           # Default nprobe
    "exact_search": False,  # Disable for speed
    "diverse_search": True, # Enable for variety
    "lambda": 0.5          # Balanced lambda
}
```

---

## 🧪 Testing Script

Here's a complete testing script you can use:

```python
#!/usr/bin/env python3
import requests
import json
import time

def test_api():
    url = "http://192.222.59.156:30888/search"
    headers = {"Content-Type": "application/json"}
    
    # Test cases
    test_cases = [
        {
            "name": "Basic Search",
            "payload": {
                "query": "machine learning",
                "n_docs": 3
            }
        },
        {
            "name": "Exact Search",
            "payload": {
                "query": "neural networks",
                "n_docs": 3,
                "exact_search": True
            }
        },
        {
            "name": "Diverse Search",
            "payload": {
                "query": "artificial intelligence",
                "n_docs": 5,
                "diverse_search": True,
                "lambda": 0.7
            }
        },
        {
            "name": "Batched Queries",
            "payload": {
                "queries": ["quantum computing", "blockchain", "AI ethics"],
                "n_docs": 2
            }
        }
    ]
    
    for test in test_cases:
        print(f"\n🧪 Testing: {test['name']}")
        print(f"Payload: {json.dumps(test['payload'], indent=2)}")
        
        start = time.time()
        response = requests.post(url, headers=headers, json=test['payload'])
        latency = time.time() - start
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            passages = data.get('results', {}).get('passages', [])
            if isinstance(passages[0], list):  # Batched
                total_results = sum(len(p) for p in passages)
                print(f"✅ Success! {len(passages)} queries, {total_results} total results in {latency:.2f}s")
            else:  # Single
                print(f"✅ Success! {len(passages[0])} results in {latency:.2f}s")
        else:
            print(f"❌ Error: {response.text}")

if __name__ == "__main__":
    test_api()
```

---

## ⚠️ Important Notes

1. **Rate Limiting**: Be mindful of request frequency to avoid overwhelming the server
2. **Query Length**: Very long queries may impact performance
3. **Parameter Combinations**: Some parameter combinations may not be optimal (e.g., very high `nprobe` with `exact_search`)
4. **Response Time**: First requests may be slower due to model loading
5. **Error Handling**: Always check response status codes and handle errors gracefully

---

## 🐛 Troubleshooting

### Common Issues

**Status 408 (Timeout)**
- Reduce `nprobe` value
- Disable `exact_search`
- Reduce `n_docs`

**Status 500 (Server Error)**
- Check query format
- Ensure all parameters are valid
- Try simpler queries first

**Slow Response**
- Use lower `nprobe` values
- Disable `exact_search` for speed
- Consider batched queries for multiple searches

---

## 📞 Support

For issues, questions, or feedback:
- Check the troubleshooting section above
- Review parameter optimization guide
- Test with the provided testing script
- Contact the development team with specific error messages and request details

---

*This documentation covers the Compact-DS Search API v1.0. For updates and additional features, please refer to the latest version.* 