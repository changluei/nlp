"""HW1 text classification: shared split, five LR baselines, full BERT fine-tuning."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BERT_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
LABELS = ['business', 'politics', 'sports']
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text):
    """Lowercase English word tokens; keep internal apostrophes, no stopword removal."""
    return TOKEN.findall(text.lower())


def stable_hash(word):
    return int.from_bytes(hashlib.md5(word.encode()).digest()[:8], 'little')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def prepare(root, output, seed):
    rows = read_csv(root / 'nyt.csv')
    assert all(r['text'].strip() and r['label'] in LABELS for r in rows)
    y = np.array([LABELS.index(r['label']) for r in rows])
    indices = np.random.RandomState(seed).permutation(len(rows))
    a, b = int(.8 * len(rows)), int(.9 * len(rows))
    split = {'train': indices[:a], 'validation': indices[a:b], 'test': indices[b:]}
    digest = hashlib.sha256((root / 'nyt.csv').read_bytes()).hexdigest()
    manifest = {'seed': seed, 'nyt_sha256': digest, 'method': 'RandomState.permutation; floor(0.8N), floor(0.9N)',
                'labels': LABELS, 'indices': {k: v.tolist() for k, v in split.items()}}
    path = output / 'split.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing split differs: use a new --output directory.')
    save_json(path, manifest)
    texts = [r['text'] for r in rows]
    text_sets = {k: set(texts[i] for i in ids) for k, ids in split.items()}
    stats = {'n': len(rows), 'label_counts': dict(zip(LABELS, np.bincount(y).tolist())),
             'split_counts': {k: dict(zip(LABELS, np.bincount(y[ids], minlength=3).tolist())) for k, ids in split.items()},
             'duplicate_text_rows': len(texts) - len(set(texts)),
             'overlap_unique_texts': {f'{a}_{b}': len(text_sets[a] & text_sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]},
             'word_length_percentiles': np.percentile([len(tokenize(t)) for t in texts], [0,25,50,75,90,100]).tolist()}
    save_json(output / 'data_stats.json', stats)
    return texts, y, split, digest


def metrics(y, pred):
    return {'accuracy': accuracy_score(y, pred), 'macro_f1': f1_score(y, pred, labels=range(3), average='macro', zero_division=0),
            'classification_report': classification_report(y, pred, labels=range(3), target_names=LABELS, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, pred, labels=range(3)).tolist()}


def record(output, name, result, y, pred, ids):
    save_json(output / f'{name}.json', result)
    directory = output / 'predictions'
    directory.mkdir(exist_ok=True)
    with (directory / f'{name}.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row_id', 'true_label', 'predicted_label'])
        writer.writerows((int(i), LABELS[int(a)], LABELS[int(b)]) for i,a,b in zip(ids,y,pred))
    print(name, json.dumps({k:result['test'][k] for k in ['accuracy','macro_f1']}), flush=True)


def train_lr(name, matrices, y, split, output, config, started):
    for matrix in matrices.values():
        values = matrix.data if hasattr(matrix, 'tocsr') else matrix
        if not np.isfinite(values).all():
            raise ValueError(f'{name}: non-finite input features')
    trials, best = [], None
    for c in [.01, .1, 1., 10.]:
        model = LogisticRegression(C=c, solver='lbfgs', max_iter=2000, tol=1e-4)
        model.fit(matrices['train'], y[split['train']])
        if not np.isfinite(model.coef_).all():
            raise ValueError(f'{name}: non-finite classifier weights')
        if model.n_iter_.max() >= model.max_iter:
            raise RuntimeError(f'{name}, C={c}: LR did not converge')
        val = metrics(y[split['validation']], model.predict(matrices['validation']))
        trials.append({'C': c, 'accuracy': val['accuracy'], 'macro_f1': val['macro_f1'], 'iterations': int(model.n_iter_.max())})
        if best is None or val['macro_f1'] > best[0]:
            best = (val['macro_f1'], model, val, c)
    _, model, val, c = best
    pred = model.predict(matrices['test'])
    result = {'method': name, 'config': {**config, 'C': c, 'solver': 'lbfgs', 'max_iter': 2000, 'class_weight': None},
              'validation_trials': trials, 'validation': val, 'test': metrics(y[split['test']], pred), 'elapsed_seconds': time.perf_counter()-started}
    record(output, name, result, y[split['test']], pred, split['test'])


def bow(texts, y, split, output):
    start = time.perf_counter()
    vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, min_df=2, dtype=np.float64)
    counts = {'train': vectorizer.fit_transform([texts[i] for i in split['train']])}
    for part in ['validation','test']:
        counts[part] = vectorizer.transform([texts[i] for i in split[part]])
    feature_seconds = time.perf_counter()-start
    for name in ['binary','frequency']:
        start = time.perf_counter()-feature_seconds
        if name == 'binary':
            matrices = {k:v.copy() for k,v in counts.items()}
            for v in matrices.values(): v.data[:] = 1
        else:
            matrices = counts
        train_lr(name, matrices, y, split, output, {'vocabulary_size': len(vectorizer.vocabulary_), 'min_df': 2,
                 'preprocessing': "[a-z]+(?:'[a-z]+)?; lowercase; no stopword removal", 'normalization': None}, start)


def mean_vectors(docs, vectors, dimension=100):
    matrix = np.zeros((len(docs), dimension), dtype=np.float32)
    hits = total = empty = 0
    for i, tokens in enumerate(docs):
        known = [vectors[t] for t in tokens if t in vectors]
        total += len(tokens)
        hits += len(known)
        if known: matrix[i] = np.mean(known, axis=0)
        else: empty += 1
    return matrix, {'token_coverage': hits / total if total else 0., 'matched_tokens': hits, 'total_tokens': total, 'empty_documents': empty}


def embedding(name, texts, y, split, root, output, seed, glove):
    start = time.perf_counter()
    docs = [tokenize(t) for t in texts]
    config = {'dimension': 100, 'pooling': 'mean over in-vocabulary token occurrences', 'scaler': 'StandardScaler fit on train only'}
    if name == 'glove':
        needed = {token for doc in docs for token in doc}
        vectors = {}
        with glove.open(encoding='utf-8') as stream:
            for line in stream:
                word, _, values = line.partition(' ')
                if word in needed:
                    vector = np.fromstring(values, sep=' ', dtype=np.float32)
                    if vector.size != 100: raise ValueError('Expected glove.6B.100d.txt')
                    vectors[word] = vector
        config['source'] = 'Stanford glove.6B.100d.txt'
        config['source_sha256'] = hashlib.sha256(glove.read_bytes()).hexdigest()
    else:
        from gensim.models import Word2Vec
        corpus = [tokenize(r['text']) for r in read_csv(root/'ag.csv')] if name=='w2v_ag' else [docs[i] for i in split['train']]
        settings = dict(vector_size=100, window=5, min_count=2, sg=1, negative=5, sample=1e-3, epochs=10, workers=1, seed=seed)
        model = Word2Vec(sentences=corpus, hashfxn=stable_hash, **settings)
        vectors = model.wv
        if not np.isfinite(vectors.vectors).all():
            raise ValueError(f'{name}: non-finite Word2Vec weights')
        config.update(settings)
        config['corpus_documents'] = len(corpus)
        config['corpus'] = 'AG News only' if name=='w2v_ag' else 'NYT training split only'
        config['vocabulary_size'] = len(vectors)
        import gensim
        config['blas_declaration_fix'] = (Path(gensim.__file__).parent/'models/word2vec_blas_fix.txt').exists()
    matrices, coverage = {}, {}
    for part, ids in split.items():
        matrices[part], coverage[part] = mean_vectors([docs[i] for i in ids], vectors)
    scaler = StandardScaler()
    matrices['train'] = scaler.fit_transform(matrices['train'])
    for part in ['validation','test']: matrices[part] = scaler.transform(matrices[part])
    config['coverage'] = coverage
    train_lr(name, matrices, y, split, output, config, start)


def bert(texts, y, split, root, output, seed, device_name, batch_size, model_path):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    start = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if device_name=='auto' else device_name)
    source = model_path or 'google-bert/bert-base-uncased'
    source_options = {} if model_path else {'revision': BERT_REVISION}
    tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=str(root/'assets/hf'), **source_options)
    encoded = tokenizer(texts, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
    raw_lengths = tokenizer(texts, truncation=False, padding=False, return_length=True, verbose=False)['length']
    class NewsDataset(Dataset):
        def __init__(self, ids): self.ids = ids
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            j = self.ids[i]
            return {**{k:v[j] for k,v in encoded.items()}, 'labels':torch.tensor(y[j], dtype=torch.long)}
    generator = torch.Generator().manual_seed(seed)
    loaders = {k:DataLoader(NewsDataset(ids), batch_size=batch_size if k=='train' else batch_size*2,
                          shuffle=k=='train', generator=generator if k=='train' else None, num_workers=0) for k,ids in split.items()}
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=3, id2label=dict(enumerate(LABELS)),
                label2id={v:k for k,v in enumerate(LABELS)}, cache_dir=str(root/'assets/hf'), **source_options).to(device)
    decay, no_decay = [], []
    for name,param in model.named_parameters():
        (no_decay if name.endswith('bias') or 'LayerNorm.weight' in name else decay).append(param)
    optimizer = torch.optim.AdamW([{'params':decay, 'weight_decay':.01}, {'params':no_decay, 'weight_decay':0.}], lr=2e-5)
    total_steps = len(loaders['train'])*3
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*.1), total_steps)
    def evaluate(part):
        model.eval()
        predictions=[]
        with torch.no_grad():
            for batch in loaders[part]:
                batch={k:v.to(device) for k,v in batch.items()}
                predictions.extend(model(**batch).logits.argmax(-1).cpu().tolist())
        return np.array(predictions)
    history, best_score, best_epoch = [], -1., None
    checkpoint = root/'checkpoints/bert-best.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1,4):
        model.train()
        loss_sum = 0.
        epoch_start = time.perf_counter()
        for step,batch in enumerate(loaders['train'],1):
            batch={k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss=model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()*len(batch['labels'])
            if step==1 or step%25==0:
                print(f'BERT epoch={epoch}/3 step={step}/{len(loaders["train"])} loss={loss.item():.4f} elapsed={time.perf_counter()-epoch_start:.1f}s device={device}',flush=True)
        val=metrics(y[split['validation']],evaluate('validation'))
        history.append({'epoch':epoch,'train_loss':loss_sum/len(split['train']), 'validation':val,'elapsed_seconds':time.perf_counter()-epoch_start})
        save_json(output/'bert_history.json',history)
        if val['macro_f1']>best_score:
            best_score,best_epoch=val['macro_f1'],epoch
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
        print(f'BERT epoch {epoch} validation {val["accuracy"]:.6f} {val["macro_f1"]:.6f}',flush=True)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    pred=evaluate('test')
    result={'method':'bert','config':{'model':'google-bert/bert-base-uncased','max_length':64,'epochs':3,'batch_size':batch_size,
            'learning_rate':2e-5,'weight_decay':.01,'warmup_ratio':.1,'gradient_clip':1.,'seed':seed,'device':str(device),
            'best_epoch':best_epoch,'selection':'highest validation macro-F1','precision':'float32',
            'truncation_fraction':{k:float(np.mean([raw_lengths[i]>64 for i in ids])) for k,ids in split.items()}},
            'history':history,'validation':history[best_epoch-1]['validation'],'test':metrics(y[split['test']],pred),'elapsed_seconds':time.perf_counter()-start}
    record(output,'bert',result,y[split['test']],pred,split['test'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['all','bow','glove','w2v_ag','w2v_nyt','bert','prepare'],default='all')
    parser.add_argument('--data-dir',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--glove',type=Path,default=ROOT/'assets/glove.6B.100d.txt')
    parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--model-path',default=None,help='Optional local snapshot of the required BERT model')
    args=parser.parse_args()
    texts,y,split,digest=prepare(args.data_dir,args.output,args.seed)
    save_json(args.output/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),'seed':args.seed,
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','gensim','torch','transformers']},'nyt_sha256':digest})
    majority=np.full(len(split['test']),np.bincount(y[split['train']]).argmax())
    save_json(args.output/'majority.json',{'method':'majority','test':metrics(y[split['test']],majority)})
    if args.task in ['all','bow']: bow(texts,y,split,args.output)
    for name in ['glove','w2v_ag','w2v_nyt']:
        if args.task in ['all',name]: embedding(name,texts,y,split,args.data_dir,args.output,args.seed,args.glove)
    if args.task in ['all','bert']: bert(texts,y,split,args.data_dir,args.output,args.seed,args.device,args.batch_size,args.model_path)


if __name__=='__main__': main()
