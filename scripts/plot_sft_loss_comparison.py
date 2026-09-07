"""Compare fixed-set train/validation losses recorded by the same Trainer."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(source, output):
    history=json.loads(source.read_text(encoding='utf-8'))
    keys=['eval_train_fixed_loss','eval_validation_loss']
    series={key:{} for key in keys}
    for row in history:
        for key in keys:
            if key in row:
                step=int(row['step'])
                value=float(row[key])
                assert math.isfinite(value) and step not in series[key]
                series[key][step]=value
    assert series[keys[0]] and series[keys[0]].keys()==series[keys[1]].keys()
    steps=sorted(series[keys[0]])
    train=[series[keys[0]][s] for s in steps]
    val=[series[keys[1]][s] for s in steps]
    best=steps[val.index(min(val))]
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(max(10,len(steps)*1.25),6.2),layout='constrained')
    ax.plot(steps,val,'o-',color='#ed952d',label='Validation｜固定22条',linewidth=2.4,markersize=7)
    ax.plot(steps,train,'o-',color='#74a7d4',label='Train｜固定66条',linewidth=2.4,markersize=7)
    for s,v,t in zip(steps,val,train):
        ax.annotate(f'{v:.5f}',(s,v),xytext=(0,10),textcoords='offset points',ha='center',color='#995e14',fontsize=10)
        ax.annotate(f'{t:.5f}',(s,t),xytext=(0,-18),textcoords='offset points',ha='center',color='#386b9a',fontsize=10)
    ax.axvline(best,color='#84909c',linestyle='--',alpha=.65,label=f'验证loss最低：第{best}步')
    ax.set(title=f'{max(steps)}步 SFT：固定训练集与验证集 Loss\n实际评估点｜同为 QLoRA 模式｜未平滑、未补第0步',xlabel='优化器步数',ylabel='Teacher-forced loss',xticks=steps,ylim=(min(train)-.016,max(val)+.018))
    ax.spines[['right','top']].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,0.76),frameon=False)
    fig.savefig(output/'train_validation_loss.png',dpi=180)
    fig.savefig(output/'train_validation_loss.svg')
    plt.close(fig)
    with (output/'loss_data.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['step',*keys])
        writer.writerows(zip(steps,train,val))
    report={'source':str(source.resolve()),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'best_step':best,'steps':steps,'train':train,'validation':val,'validation_last_interval_steps':steps[-2:],'validation_last_interval_delta':val[-1]-val[-2] if len(val)>1 else None,'scope':'same Trainer fixed-set QLoRA evaluation; not final merged BF16 evaluation; not task success'}
    (output/'plot_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    render(args.source,args.output)
