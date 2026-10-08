"""Recompute saved metrics and audit exact duplicate overlap without model selection."""
import csv
import json
from pathlib import Path
import numpy as np
from experiment import LABELS, metrics, read_csv, save_json

ROOT=Path(__file__).resolve().parents[1]
METHODS=['binary','frequency','glove','w2v_ag','w2v_nyt','bert']


def main():
    rows=read_csv(ROOT/'nyt.csv')
    split=json.loads((ROOT/'results/split.json').read_text())['indices']
    train_texts={rows[i]['text'] for i in split['train']}
    audit={}
    for method in METHODS:
        pred=read_csv(ROOT/'results/predictions'/f'{method}.csv')
        ids=[int(r['row_id']) for r in pred]
        assert ids==split['test'], f'{method}: test order mismatch'
        y=np.array([LABELS.index(r['true_label']) for r in pred])
        assert [LABELS[v] for v in y]==[rows[i]['label'] for i in ids]
        yh=np.array([LABELS.index(r['predicted_label']) for r in pred])
        result=json.loads((ROOT/'results'/f'{method}.json').read_text())
        check=metrics(y,yh)
        for key in ['accuracy','macro_f1']: assert abs(check[key]-result['test'][key])<1e-12
        assert check['confusion_matrix']==result['test']['confusion_matrix']
        keep=np.array([rows[i]['text'] not in train_texts for i in ids])
        errors=[{'row_id':i,'true_label':LABELS[int(a)],'predicted_label':LABELS[int(b)],'excerpt':rows[i]['text'][:500]} for i,a,b in zip(ids,y,yh) if a!=b]
        audit[method]={'original_test_n':len(ids),'no_train_overlap_n':int(keep.sum()),
                       'no_train_overlap':metrics(y[keep],yh[keep]),'error_count':len(errors),'example_errors':errors[:5]}
    save_json(ROOT/'results/robustness.json',audit)
    print('All six saved results match predictions; duplicate-excluded sensitivity computed.')


if __name__=='__main__':main()
