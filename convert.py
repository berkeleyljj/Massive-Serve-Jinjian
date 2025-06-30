import os
import pickle
import numpy as np
import shutil
import gzip

# ====== Configure Paths ======
DOMAIN_DIR = "/home/ubuntu/Jinjian/retrieval-scaling/built_index"
META_FILE = os.path.join(DOMAIN_DIR, "index/index_IVFPQ.100000000.768.65536.64.faiss.meta")
META_NPY = os.path.join(DOMAIN_DIR, "index_id_to_db_id.npy")

NMAP_FILE = os.path.join(DOMAIN_DIR, "passage_pos_id_map.pkl")
NMAP_SAVE_FILE = os.path.join(DOMAIN_DIR, "passage_pos_id_map.pkl.gz")

BACKUP_DIR = os.path.expanduser("~/meta_backup")
os.makedirs(BACKUP_DIR, exist_ok=True)


def convert_meta_to_npy():
    print(f"Loading pickle: {META_FILE}")
    with open(META_FILE, "rb") as f:
        index_id_to_db_id = pickle.load(f)
    print(f"Type: {type(index_id_to_db_id)}, Length: {len(index_id_to_db_id)}")
    array = np.array(index_id_to_db_id, dtype=np.int64)
    print(f"Converted to numpy array of shape {array.shape} and dtype {array.dtype}")
    print(f"Saving .npy to {META_NPY}")
    np.save(META_NPY, array)
    backup_path = os.path.join(BACKUP_DIR, os.path.basename(META_FILE))
    print(f"Moving original pickle to backup: {backup_path}")
    shutil.move(META_FILE, backup_path)

def convert_passage_map():
    if not os.path.exists(NMAP_FILE):
        print(f"Not found: {NMAP_FILE}, skipping.")
        return

    print(f"Loading pickle: {NMAP_FILE}")
    with open(NMAP_FILE, "rb") as f:
        pos_map = pickle.load(f)

    print(f"Converting map of length {len(pos_map)}")
    compacted = {}

    for shard_id, pairs in pos_map.items():
        compacted_list = []
        for entry in pairs:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                f, pos = entry
                compacted_list.append((os.path.basename(f), int(pos)))
            elif isinstance(entry, int):
                compacted_list.append(int(entry))
            else:
                print(f"⚠️ Unexpected format in shard {shard_id}: {entry} (skipped)")
        compacted[shard_id] = compacted_list

    print(f"Saving gzip-compressed compacted map to: {NMAP_SAVE_FILE}")
    with gzip.open(NMAP_SAVE_FILE, "wb") as f:
        pickle.dump(compacted, f, protocol=pickle.HIGHEST_PROTOCOL)

    backup_path = os.path.join(BACKUP_DIR, os.path.basename(NMAP_FILE))
    print(f"Moving original pickle to backup: {backup_path}")
    shutil.move(NMAP_FILE, backup_path)

if __name__ == "__main__":
    # convert_meta_to_npy()  # Optional
    convert_passage_map()
    print("✅ Conversion complete.")
