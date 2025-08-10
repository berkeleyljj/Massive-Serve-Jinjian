import requests
import json
import time

url = "http://192.222.59.156:30888/search"
headers = {"Content-Type": "application/json"}

print("🧪 Testing New API Parameters (exact_search & diverse_search)")
print("=" * 60)

# Test 1: Single Query
print("\n1️⃣ SINGLE QUERY Test")
payload = {
    "query": "artificial intelligence",  # Single string
    "nprobe": 32,
    "n_docs": 3,
    "exact_search": True,
    "diverse_search": True,
    "lambda": 0.6
}

print(f"Payload: {json.dumps(payload, indent=2)}")
start = time.time()
response = requests.post(url, headers=headers, json=payload)
latency = time.time() - start

print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    passages = data.get('results', {}).get('passages', [{}])[0]
    print(f"✅ Success! Found {len(passages)} passages in {latency:.2f}s")
    if passages:
        print(f"First result: {passages[0].get('text', '')[:80]}...")
else:
    print(f"❌ Error: {response.text}")

# Test 2: Batched Queries
print("\n\n2️⃣ BATCHED QUERIES Test")
payload = {
    "queries": [                          # Array of strings for batched
        "machine learning algorithms",
        "neural network architecture", 
        "deep learning optimization"
    ],
    "nprobe": 64,
    "n_docs": 2,
    "exact_search": True,
    "diverse_search": False,
    "lambda": 0.5
}

print(f"Payload: {json.dumps(payload, indent=2)}")
start = time.time()
response = requests.post(url, headers=headers, json=payload)
latency = time.time() - start

print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    passages = data.get('results', {}).get('passages', [])
    print(f"✅ Success! Processed {len(payload['queries'])} queries in {latency:.2f}s")
    
    # Show results for each query
    for i, query_passages in enumerate(passages):
        query_text = payload['queries'][i]
        print(f"  Query {i+1}: '{query_text}' → {len(query_passages)} results")
        if query_passages:
            print(f"    Sample: {query_passages[0].get('text', '')[:60]}...")
else:
    print(f"❌ Error: {response.text}")

# Test 3: Batched with Diverse Search
print("\n\n3️⃣ BATCHED + DIVERSE Search Test")
payload = {
    "queries": [
        "quantum computing applications",
        "blockchain technology",
        "computer vision algorithms"
    ],
    "nprobe": 32,
    "n_docs": 3,
    "exact_search": False,
    "diverse_search": True,    # Enable diversity for batched queries
    "lambda": 0.8             # High diversity
}

print(f"Payload: {json.dumps(payload, indent=2)}")
start = time.time()
response = requests.post(url, headers=headers, json=payload)
latency = time.time() - start

print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    passages = data.get('results', {}).get('passages', [])
    print(f"✅ Success! Diverse batched search completed in {latency:.2f}s")
    
    total_results = sum(len(query_passages) for query_passages in passages)
    print(f"Total results across all queries: {total_results}")
else:
    print(f"❌ Error: {response.text}")

# Test 4: Performance Comparison
print("\n\n4️⃣ PERFORMANCE COMPARISON")
queries = ["AI ethics", "robotics", "data science", "cybersecurity"]

# Individual requests
print("Individual requests:")
start = time.time()
individual_results = []
for query in queries:
    payload = {
        "query": query,
        "n_docs": 2,
        "exact_search": False,
        "diverse_search": False
    }
    resp = requests.post(url, headers=headers, json=payload)
    individual_results.append(resp.json())
individual_time = time.time() - start

# Batched request
print("Batched request:")
start = time.time()
payload = {
    "queries": queries,
    "n_docs": 2,
    "exact_search": False,
    "diverse_search": False
}
batch_resp = requests.post(url, headers=headers, json=payload)
batch_time = time.time() - start

print(f"Individual: {individual_time:.2f}s")
print(f"Batched:    {batch_time:.2f}s")
print(f"Speedup:    {individual_time/batch_time:.1f}x faster")

print("\n✅ All tests completed!")
print("\n📋 SUMMARY:")
print("• Single query: Use 'query' field with string")
print("• Batched queries: Use 'queries' field with array of strings")
print("• Both support exact_search and diverse_search parameters")
print("• Batched requests are typically faster than individual requests")