"""Plot metrics, confusion matrices and training history from measured JSON results."""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAMES={'binary':'Binary BoW','frequency':'Word frequency','tfidf':'TF-IDF','glove':'GloVe 6B','w2v_ag':'Word2Vec AG','w2v_nyt':'Word2Vec NYT','bert':'BERT'}


def main():
    results={k:json.loads((ROOT/'results'/f'{k}.json').read_text()) for k in NAMES}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                        'figure.dpi':140,'savefig.dpi':300,'savefig.bbox':'tight','legend.frameon':False})
    output=ROOT/'results/figures';output.mkdir(exist_ok=True)
    with (ROOT/'results/summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','test_accuracy','test_macro_f1','validation_macro_f1','seconds'])
        writer.writerows((k,v['test']['accuracy'],v['test']['macro_f1'],v['validation']['macro_f1'],v['elapsed_seconds']) for k,v in results.items())
    fig,ax=plt.subplots(figsize=(8,3.8))
    y=np.arange(len(NAMES))
    for metric,color,marker,label in [('accuracy','#0072B2','o','Accuracy'),('macro_f1','#D55E00','s','Macro-F1')]:
        ax.scatter([results[k]['test'][metric]*100 for k in NAMES],y,marker=marker,color=color,label=label,zorder=3)
    ax.set_yticks(y,list(NAMES.values()));ax.invert_yaxis();ax.set_xlabel('Test score (%)');ax.grid(axis='x',alpha=.25);ax.legend(loc='lower left')
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(output/f'comparison.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.3))
    for ax,name in zip(axes,['frequency','w2v_nyt','bert']):
        matrix=np.array(results[name]['test']['confusion_matrix']);norm=matrix/matrix.sum(1,keepdims=True)
        ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3): ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if norm[i,j]>.5 else 'black')
        ax.set_xticks(range(3),['Business','Politics','Sports'],rotation=25,ha='right');ax.set_yticks(range(3),['Business','Politics','Sports'])
        ax.set_title(NAMES[name]);ax.set_xlabel('Predicted label')
    axes[0].set_ylabel('True label');fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(output/f'confusion.{ext}')
    plt.close(fig)
    history=results['bert']['history'];fig,axes=plt.subplots(1,2,figsize=(8,3))
    epochs=[h['epoch'] for h in history]
    axes[0].plot(epochs,[h['train_loss'] for h in history],'o-',color='#0072B2');axes[0].set_ylabel('Training cross-entropy')
    axes[1].plot(epochs,[h['validation']['macro_f1']*100 for h in history],'s-',color='#D55E00');axes[1].set_ylabel('Validation Macro-F1 (%)')
    for ax in axes: ax.set_xlabel('Epoch');ax.set_xticks(epochs);ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(output/f'bert_history.{ext}')
    plt.close(fig)


if __name__=='__main__': main()
