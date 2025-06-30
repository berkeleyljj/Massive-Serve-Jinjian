import faiss

index_path = "/home/ubuntu/Jinjian/retrieval-scaling/built_index/index/index_IVFPQ.100000000.768.65536.64.faiss"
index = faiss.read_index(index_path)
print("Total vectors in index:", index.ntotal)
