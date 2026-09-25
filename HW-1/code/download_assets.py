"""Download the specified GloVe 6B 100d and BERT assets, never replacement models."""
import argparse
import gzip
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
GLOVE_SHA256 = '95dde4dfd627ab26608d33e76d1195ec059734bd29089ea52cadb08d07c64544'
BERT_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'


def download(url, target):
    partial = target.with_suffix(target.suffix + '.part')
    print(f'Downloading {url}', flush=True)
    urllib.request.urlretrieve(url, partial)
    partial.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glove-source', choices=['mirror','stanford'], default='mirror')
    args = parser.parse_args()
    assets = ROOT/'assets'
    assets.mkdir(exist_ok=True)
    target = assets/'glove.6B.100d.txt'
    if not target.exists():
        if args.glove_source == 'stanford':
            archive = assets/'glove.6B.zip'
            download('https://nlp.stanford.edu/data/glove.6B.zip', archive)
            with zipfile.ZipFile(archive) as z, z.open(target.name) as src, target.open('wb') as dst:
                shutil.copyfileobj(src,dst)
        else:
            archive = assets/'glove.6B.100d.txt.gz'
            download('https://huggingface.co/datasets/SLU-CSCI4750/glove.6B.100d.txt/resolve/main/glove.6B.100d.txt.gz', archive)
            with gzip.open(archive,'rb') as src, target.open('wb') as dst:
                shutil.copyfileobj(src,dst)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != GLOVE_SHA256:
        raise ValueError(f'GloVe checksum mismatch: {digest}; remove {target} and retry')
    print('GloVe SHA256 verified:', digest)
    from huggingface_hub import snapshot_download
    path = snapshot_download('google-bert/bert-base-uncased', revision=BERT_REVISION,
        cache_dir=str(assets/'hf'), allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','vocab.txt','model.safetensors'])
    print('BERT snapshot:', path)


if __name__ == '__main__': main()
