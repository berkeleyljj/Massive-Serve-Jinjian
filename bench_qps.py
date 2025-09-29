import argparse
import time
import requests
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def load_queries(query_file, num_queries=1000):
    # Load or generate dummy queries
    if query_file:
        queries = []
        with open(query_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_queries:
                    break
                line = line.strip()
                if not line:
                    continue
                # If jsonl, try to extract a 'query' field; otherwise use the raw line
                if query_file.endswith('.jsonl'):
                    try:
                        rec = json.loads(line)
                        q = rec.get('query') or rec.get('full_text') or rec.get('question')
                        queries.append(q if isinstance(q, str) else line)
                    except Exception:
                        queries.append(line)
                else:
                    queries.append(line)
    else:
        queries = [f"Test query {i}" for i in range(num_queries)]
    return queries

def sequential_search(endpoint, queries, n_docs=10, nprobe=32, timeout_s=10.0, progress_every=50, exact=False):
    times = []
    success = 0
    start_all = time.perf_counter()
    for i, query in enumerate(queries):
        t0 = time.perf_counter()
        payload = {"query": query, "n_docs": n_docs, "nprobe": nprobe, "exact_search": bool(exact)}
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout_s)
            if response.status_code != 200:
                print(f"  [{i+1}/{len(queries)}] HTTP {response.status_code}")
                continue
            dt = time.perf_counter() - t0
            times.append(dt)
            success += 1
            # lightweight progress
            if progress_every and (((i+1) % progress_every == 0) or (i+1 == len(queries))):
                print(f"  [{i+1}/{len(queries)}] {dt*1000:.1f} ms")
        except Exception as e:
            print(f"  [{i+1}/{len(queries)}] error: {e}")
            continue
    wall = time.perf_counter() - start_all
    qps = (success / wall) if wall > 0 else 0.0
    avg = np.mean(times) if times else 0.0
    std = np.std(times) if times else 0.0
    return qps, avg, std

def batched_search(endpoint, queries, batch_size, n_docs=10, nprobe=32, timeout_s=10.0):
    times = []
    sent = 0
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        t0 = time.perf_counter()
        payload = {"queries": batch, "n_docs": n_docs, "nprobe": nprobe}
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout_s)
            if response.status_code != 200:
                print(f"  [batch {sent+len(batch)}/{len(queries)}] HTTP {response.status_code}")
                continue
            dt = time.perf_counter() - t0
            times.append(dt)
            sent += len(batch)
            print(f"  [batch {sent}/{len(queries)}] {dt*1000:.1f} ms")
        except Exception as e:
            print(f"  [batch {sent+len(batch)}/{len(queries)}] error: {e}")
            continue
    total_time = sum(times)
    qps = len(queries) / total_time if total_time > 0 else 0.0
    return qps, (np.mean(times) if times else 0.0), (np.std(times) if times else 0.0)

def parallel_sequential_search(endpoint, queries, num_threads, n_docs=10, nprobe=32, timeout_s=10.0):
    def single_query(query):
        payload = {"query": query, "n_docs": n_docs, "nprobe": nprobe}
        response = requests.post(endpoint, json=payload, timeout=timeout_s)
        if response.status_code != 200:
            raise ValueError(f"Error: {response.text}")

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(single_query, queries)
    total_time = time.perf_counter() - start
    qps = len(queries) / total_time
    return qps

def main():
    parser = argparse.ArgumentParser(description="Benchmark QPS for /search endpoint")
    parser.add_argument("--endpoint", default="http://compactds.duckdns.org:30888/search", help="API endpoint")
    parser.add_argument("--query_file", default=None, help="File with queries (one per line)")
    parser.add_argument("--num_queries", type=int, default=1000, help="Number of queries")
    parser.add_argument("--n_docs", type=int, default=10, help="n_docs per query")
    parser.add_argument("--nprobe", type=int, default=256, help="nprobe value (ignored; forced to 256)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for batched mode")
    parser.add_argument("--num_threads", type=int, default=4, help="Threads for parallel sequential")
    parser.add_argument("--timeout", type=float, default=8.0, help="Per-request timeout in seconds")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup requests before timing")
    parser.add_argument("--progress_every", type=int, default=50, help="Print progress every N queries")
    parser.add_argument("--exact", action="store_true", help="Enable exact_search=true in requests")
    args = parser.parse_args()

    queries = load_queries(args.query_file, args.num_queries)

    print(f"Benchmarking with {len(queries)} queries against {args.endpoint}")
    print("Forcing nprobe=256 (UI parity)")

    # Warmup
    for i in range(max(0, int(args.warmup))):
        try:
            requests.post(args.endpoint, json={"query": f"warmup {i}", "n_docs": 1, "nprobe": 256, "exact_search": bool(args.exact)}, timeout=args.timeout)
        except Exception:
            pass

    # Sequential only
    print("\nSequential (single-threaded, ANN only):")
    seq_qps, avg_lat, std_lat = sequential_search(
        args.endpoint,
        queries,
        args.n_docs,
        256,  # forced nprobe
        timeout_s=args.timeout,
        progress_every=max(1, int(args.progress_every)),
        exact=bool(args.exact)
    )
    print(f"QPS: {seq_qps:.2f}, Avg latency: {avg_lat*1000:.2f}ms (±{std_lat*1000:.2f}ms)")

if __name__ == "__main__":
    main()
