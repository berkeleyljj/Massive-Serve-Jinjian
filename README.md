# Massive Serve User Guide

One command to download and serve a datastore---that's it 😎.

## Installation
```bash
pip install massive-serve
```

## Usage

To serve a demo datastore (30MB):
```bash
massive-serve serve --domain_name demo
```

To serve a wikipedia datastore (60GB):
```bash
massive-serve serve --domain_name dpr_wiki_contriever
```

More domains and retriever combinations coming soon!

It will then download and serve the index and print the API and one example request in the terminal, e.g.,
```markdown
╔════════════════════════════════════════════════════════════╗
║                    MASSIVE SERVE SERVER                    ║
╠════════════════════════════════════════════════════════════╣
║ Domain: demo                                               ║
║ Server: XXX                                                ║
║ Port:   XXX                                                ║
║ Endpoint: XXX@XXX:XXX/search                               ║
╚════════════════════════════════════════════════════════════╝


Test your server with this curl command:

curl -X POST XXX@XXX:XXX/search -H "Content-Type: application/json" -d '{"query": "Tell me more about the stories of Einstein.", "n_docs": 1, "domains": "demo"}'
```

### Send Requests
If the API has been served, you can either send single or bulk query requests to it.

**Bash Examples.**

```bash
# single-query request
curl -X POST <user>@<address>:<port>/search -H "Content-Type: application/json" -d '{"query": "Where was Marie Curie born?", "n_docs": 1, "domains": "dpr_wiki_contriever"}'

# multi-query request
curl -X POST <user>@<address>:<port>/search -H "Content-Type: application/json" -d '{"query": ["Where was Marie Curie born?", "What is the capital of France?", "Who invented the telephone?"], "n_docs": 2, "dpr_wiki_contriever": "MassiveDS"}'
```

**Python Example.**
```python
import requests

json_data = {
    'query': 'Where was Marie Curie born?',
    "n_docs": 20,
    "domains": "dpr_wiki_contriever"
}
headers = {"Content-Type": "application/json"}

# Add 'http://' to the URL if it is not SSL/TLS secured, otherwise use 'https://'
response = requests.post('http://<user>@<address>:<port>/search', json=json_data, headers=headers)

print(response.status_code)
print(response.json())
```

Example output of a multi-query request:
```json
{
  "message": "Search completed for '['Where was Marie Curie born?', 'What is the capital of France?', 'Who invented the telephone?']' from MassiveDS",
  "n_docs": 2,
  "query": [
    "Where was Marie Curie born?",
    "What is the capital of France?",
    "Who invented the telephone?"
  ],
  "results": {
    "n_docs": 2,
    "query": [
      "Where was Marie Curie born?",
      "What is the capital of France?",
      "Who invented the telephone?"
    ],
    "results": {
      "IDs": [
        [
          [3, 3893807],
          [17, 11728753]
        ],
        [
          [14, 12939685],
          [22, 1070951]
        ],
        [
          [28, 18823956],
          [22, 10406782]
        ]
      ],
      "passages": [
        [
          "Marie Skłodowska Curie (November 7, 1867 – July 4, 1934) was a physicist and chemist of Polish upbringing and, subsequently, French citizenship. ...",
          "=> Maria Skłodowska, better known as Marie Curie, was born on 7 November in Warsaw, Poland. ..."
        ],
        [
          "Paris is the capital and most populous city in France, as well as the administrative capital of the region of Île-de-France. ...",
          "[paʁi] ( listen)) is the capital and largest city of France. ..."
        ],
        [
          "Antonio Meucci (Florence, April 13, 1808 – October 18, 1889) was an Italian inventor. ...",
          "The telephone or phone is a telecommunications device that transmits speech by means of electric signals. ..."
        ]
      ],
      "scores": [
        [
          1.8422218561172485,
          1.8394594192504883
        ],
        [
          1.5528039932250977,
          1.5502511262893677
        ],
        [
          1.714379906654358,
          1.706493854522705
        ]
      ]
    }
  }
}
```


# Massive Serve Developer Guide

## Environment Setup

### Using Conda (Recommended for GPU support)

1. Create a new conda environment:
```bash
git clone https://github.com/RulinShao/massive-serve.git
cd massive-serve
conda env create -f conda-env.yml
conda activate massive-serve
```
To update the existing environment:
```bash
conda env update -n massive-serve -f conda-env.yml
```

## Upload new index

```bash
python -m massive_serve.cli upload-data --domain_name demo
```

Test serving the index:
```bash
python -m massive_serve.cli serve --domain_name demo
```

## Update package
Make sure the version in the `setup.py` has been updated to a different version. Then run:
```bash
rm -rf dist/ build/ massive_serve.egg-info/
pip install build twine
python -m build
python -m twine upload dist/*
```
Users can refresh their installed repo via:
```bash
pip install --upgrade massive-serve
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.


# Citation
If you find our package helpful, please cite:
```
@article{shao2024scaling,
  title={Scaling retrieval-based language models with a trillion-token datastore},
  author={Shao, Rulin and He, Jacqueline and Asai, Akari and Shi, Weijia and Dettmers, Tim and Min, Sewon and Zettlemoyer, Luke and Koh, Pang Wei W},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={91260--91299},
  year={2024}
}
```
