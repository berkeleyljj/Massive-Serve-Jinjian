import os
import sys
import subprocess
import click

@click.group()
def cli():
    """Massive Serve CLI"""
    pass

@cli.command()
def dpr_wiki():
    """Run the DPR wiki worker node"""
    # Set PYTHONPATH to include the current directory
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    # Set datastore path to `~` if it is not already set
    if 'DATASTORE_PATH' not in env:
        env['DATASTORE_PATH'] = '~'
    
    # Download the wiki index dataset
    save_path = os.path.join(os.path.expanduser(env['DATASTORE_PATH']), 'dpr_wiki_contriever')
    subprocess.run(['huggingface-cli', 'download', 'rulins/massive_serve_dpr_wiki_contriever', '--repo-type', 'dataset', '--local-dir', save_path])
    
    # Run the worker node script
    subprocess.run(['python', 'api/serve_dpr_wiki.py'], env=env)

if __name__ == '__main__':
    cli() 