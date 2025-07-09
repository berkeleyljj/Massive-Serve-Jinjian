import faiss
import os

index_path = "/home/ubuntu/Jinjian/massive-serve/FAISS_backup/index/index_full.faiss"
output_dir = "/home/ubuntu/Jinjian/massive-serve/FAISS_backup/shards"
os.makedirs(output_dir, exist_ok=True)

n_shards = 10  # ✅ adjust as needed

index = faiss.read_index(index_path)
assert isinstance(index, faiss.IndexIVF), "Only IndexIVF types can be sharded this way."

nlist = index.nlist
lists_per_shard = nlist // n_shards

for i in range(n_shards):
    start = i * lists_per_shard
    end = (i + 1) * lists_per_shard if i < n_shards - 1 else nlist
    print(f"Creating shard {i}: IVF lists {start} to {end}")

    # Clone and zero out all inverted lists
    shard_index = faiss.clone_index(index)
    for j in range(nlist):
        if j < start or j >= end:
            shard_index.invlists.replace_invlist(j, faiss.ArrayInvertedLists(0, 0))

    shard_path = os.path.join(output_dir, f"index_shard_{i:02d}.faiss")
    faiss.write_index(shard_index, shard_path)

print(f"✅ Sharded index written to: {output_dir}")
