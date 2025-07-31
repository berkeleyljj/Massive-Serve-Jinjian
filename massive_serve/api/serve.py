import os
import json
import traceback
import datetime
import socket
import getpass
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import queue
from flask import send_from_directory
import psutil
import time
from colorama import Fore, Style
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from gritlm import GritLM


startup_start = time.time()
query_tokenizer  = None
query_encoder = GritLM("GritLM/GritLM-7B", torch_dtype="auto", mode="embedding")
query_encoder.eval()
query_encoder = query_encoder.to(device)

# Function to measure memory usage
def print_mem_use(label=""):
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_mb = mem_bytes / 1024 / 1024
    print(f"[{label}] Memory usage: {mem_mb:.2f} MB")

from massive_serve.api.api_index import get_datastore

# ANSI color codes
CYAN = '\033[36m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
WHITE = '\033[37m'
RESET = '\033[0m'


class DatastoreConfig():
    def __init__(self,):
        super().__init__()
        self.data_root = os.environ.get('DATASTORE_PATH', os.path.expanduser('~'))
        self.domain_name = os.environ.get('MASSIVE_SERVE_DOMAIN_NAME', 'demo')
        self.config = json.load(open(os.path.join(self.data_root, self.domain_name, 'config.json')))
        assert self.config['domain_name'] == self.domain_name, f"Domain name in config.json ({self.config['domain_name']}) does not match the domain name in the environment variable ({self.domain_name})"
        self.query_encoder = self.config['query_encoder']
        self.query_tokenizer = self.config['query_tokenizer']
        self.index_type = self.config['index_type']
        self.per_gpu_batch_size = self.config['per_gpu_batch_size']
        self.question_maxlength = self.config['question_maxlength']
        self.nprobe = self.config['nprobe']

ds_cfg = DatastoreConfig()
        

app = Flask(__name__)
CORS(app)


class Item:
    def __init__(self, query=None, query_embed=None, domains=ds_cfg.domain_name, n_docs=1, nprobe=None, expand_index_id=None, expand_offset=None, exact_rerank=False, use_diverse=False, lambda_val=None) -> None:
        self.query = query
        self.query_embed = query_embed
        self.domains = domains
        self.n_docs = n_docs
        self.nprobe = nprobe
        self.searched_results = None
        self.expand_index_id = expand_index_id
        self.expand_offset = expand_offset
        self.exact_rerank = exact_rerank
        self.use_diverse = use_diverse
        self.lambda_val = lambda_val
    
    def get_dict(self,):
        dict_item = {
            'query': self.query,
            'query_embed': self.query_embed,
            'domains': self.domains,
            'n_docs': self.n_docs,
            'nprobe': self.nprobe,
            'searched_results': self.searched_results,
            'expand_index_id' : self.expand_index_id,
            'expand_offset' : self.expand_offset,
            'exact_rerank' : self.exact_rerank,
        }
        return dict_item


class SearchQueue:
    def __init__(self, log_queries=False):
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.current_search = None
        print_mem_use("Before get datastore")
        self.datastore = get_datastore(ds_cfg)
        print_mem_use("After get datastore")
        self.log_queries = log_queries
        self.query_log = 'cached_queries.jsonl'
    
    def search(self, item):
        with self.lock:
            if self.current_search is None:
                self.current_search = item
                if item.nprobe is not None:
                    self.datastore.index.nprobe = item.nprobe
                results = self.datastore.search(item.query, item.n_docs, item.nprobe, item.expand_index_id, item.expand_offset, item.exact_rerank, item.use_diverse, item.lambda_val)
                self.current_search = None
                return results
            else:
                future = threading.Event()
                self.queue.put((item, future))
                future.wait()
                return item.searched_results
    
    def process_queue(self):
        while True:
            item, future = self.queue.get()
            with self.lock:
                self.current_search = item
                item.searched_results = self.datastore.search(item)
                self.current_search = None
            future.set()
            self.queue.task_done()

search_queue = SearchQueue()
threading.Thread(target=search_queue.process_queue, daemon=True).start()

@app.route('/search', methods=['POST'])
def search():
    try:
        if "queries" in request.json:
            print(f"[Batched Search] Received {len(request.json['queries'])} queries.")
            query_input = request.json["queries"]
        else:
            print("Processing a single query at once. ")
            query_input=request.json['query']
        item = Item(
            query=query_input,
            n_docs=request.json.get('n_docs', 1),
            nprobe=request.json.get('nprobe', None),
            expand_index_id = request.json.get('expand_index_id'),
            expand_offset = request.json.get('expand_offset', 1),
            domains=ds_cfg.domain_name,
            exact_rerank = request.json.get('use_rerank', False),
            use_diverse=request.json.get('use_diverse', False),
            lambda_val=request.json.get('lambda', 0.5),
        )

        # Perform the search synchronously with 600s timeout
        timer = threading.Timer(600.0, lambda: (_ for _ in ()).throw(TimeoutError('Search timed out after 600 seconds')))
        timer.start()
        try:
            results = search_queue.search(item)
            timer.cancel()

            # print(" ========= DEBUGGGGGG ======= Structured results from search route:")
            # print(json.dumps(results, indent=2, ensure_ascii=False))

            def convert_ndarrays(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [convert_ndarrays(i) for i in obj]
                elif isinstance(obj, dict):
                    return {k: convert_ndarrays(v) for k, v in obj.items()}
                else:
                    return obj

            results = convert_ndarrays(results)

            return jsonify({
                "message": f"Search completed for '{item.query}' from {item.domains}",
                "query": query_input,
                "n_docs": item.n_docs,
                "nprobe": item.nprobe,
                "expand_index_id" : request.json.get('expand_index_id'),
                "expand_offset" : request.json.get('expand_offset', 1),
                "results": results,
            }), 200
        except TimeoutError as e:
            timer.cancel()
            return jsonify({
                "message": str(e),
                "query": item.query,
                "error": "timeout"
            }), 408
    except Exception as e:
        tb_lines = traceback.format_exception(e.__class__, e, e.__traceback__)
        error_message = f"An error occurred: {str(e)}\n{''.join(tb_lines)}"
        return jsonify({"message": error_message}), 500
    
@app.route('/current_search')
def current_search():
    with search_queue.lock:
        current = search_queue.current_search
        if current:
            return jsonify({
                "current_search": current.query,
                "domains": current.domains,
                "n_docs": current.n_docs,
            }), 200
        else:
            return jsonify({"message": "No search currently in progress"}), 200

@app.route('/queue_size')
def queue_size():
    size = search_queue.queue.qsize()
    return jsonify({"queue_size": size}), 200

@app.route("/")
def home():
    return jsonify("Hello! What you are looking for?")

@app.route('/ui')
def serve_ui():
    return send_from_directory('.', 'backup.html')

def find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.

@app.route('/<path:filename>')
def serve_static_file(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(script_dir, filename)

def main():
    #port = find_free_port()
    port = int(os.environ.get('MASSIVE_SERVE_PORT', 30888))
    server_id = socket.gethostname()
    domain_name = ds_cfg.domain_name
    serve_info = {'server_id': server_id, 'port': port}
    username = os.environ.get('USER') or os.environ.get('USERNAME') or getpass.getuser()
    endpoint = f'{username}@{server_id}:{port}/search'  # use actual username
    

    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    WHITE = Fore.WHITE
    RESET = Style.RESET_ALL
    box_width = 63  

    def pad(label, value):
        total_padding = box_width - len(label)
        return label + value.ljust(total_padding)

    banner = f"""
    {CYAN}╔{'═' * box_width}╗
    ║{GREEN}{'MASSIVE SERVE SERVER'.center(box_width)}{CYAN}║
    ╠{'═' * box_width}╣
    ║{YELLOW}{pad(" Domain:   ", WHITE + domain_name)}{CYAN}║
    ║{YELLOW}{pad(" Server:   ", WHITE + server_id)}{CYAN}║
    ║{YELLOW}{pad(" Port:     ", WHITE + str(port))}{CYAN}║
    ║{YELLOW}{pad(" Endpoint: ", WHITE + endpoint)}{CYAN}║
    ╚{'═' * box_width}╝{RESET}
    """
    print(banner)
    
    # Print test request with colors
    test_request = f"""
{GREEN}Test your server with this curl command:\n{RESET}
{CYAN}curl -X POST {endpoint} -H "Content-Type: application/json" -d '{YELLOW}{{"query": "Tell me more about the stories of Einstein.", "n_docs": 1, "domains": "{domain_name}"}}{CYAN}'{RESET}
"""
    print(test_request)
    print(f"Deployment completed in {time.time() - startup_start:.2f} seconds")
    app.run(host='0.0.0.0', port=port)
    # app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)



if __name__ == '__main__':
    main()