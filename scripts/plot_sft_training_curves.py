"""Plot recorded SFT metrics; never fabricate steps or missing values."""
import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_rows(path):
    text = path.read_text(encoding='utf-8', errors='replace')
    if path.suffix == '.json':
        payload = json.loads(text)
        return payload if isinstance(payload, list) else payload['log_history']
    rows = []
    for match in re.finditer(r"\{[^\r\n]*'loss'[^\r\n]*\}", text):
        row = ast.literal_eval(match.group())
        steps = re.findall(r'(\d+)/20\s*\[', text[:match.start()])
        if not steps:
            raise ValueError('Console loss has no recorded progress step')
        row['step'] = int(steps[-1])
        rows.append(row)
    if not rows or any(a['step'] >= b['step'] for a,b in zip(rows, rows[1:])):
        raise ValueError('Missing, duplicate or out-of-order steps')
    return rows


def render(source, output, eval_before=None, eval_after=None):
    rows = [r for r in load_rows(source) if 'loss' in r and 'step' in r]
    output.mkdir(parents=True, exist_ok=True)
    metrics = [('loss','训练损失 / Train loss'),('eval_loss','验证损失 / Eval loss'),('learning_rate','学习率 / Learning rate'),('grad_norm','梯度范数 / Grad norm')]
    validation = None
    if eval_before is not None and eval_after is not None:
        before=json.loads(eval_before.read_text(encoding='utf-8'))
        after=json.loads(eval_after.read_text(encoding='utf-8'))
        assert before['rows']==after['rows']
        assert before['assistant_tokens']==after['assistant_tokens']
        assert [(r['candidate_id'],r['evaluated_tokens']) for r in before['per_row']]==[(r['candidate_id'],r['evaluated_tokens']) for r in after['per_row']]
        validation=[before['mean_assistant_loss'],after['mean_assistant_loss']]
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    for ax,(key,title) in zip(axes.flat,metrics):
        if key=='eval_loss':
            ax.set_title(title+'（仅前后两次评测）')
            if validation is not None:
                bars=ax.bar(['训练前','20步训练后'],validation,color=['#91a7b4','#197c86'],width=.48)
                ax.bar_label(bars,fmt='%.5f',padding=5)
                ax.set_ylim(0,max(validation)*1.22)
                ax.set_xlabel(f'同一22条验证集｜相对下降 {(1-validation[1]/validation[0])*100:.2f}%')
            else:
                ax.text(.5,.5,'未记录',ha='center',transform=ax.transAxes)
            ax.grid(axis='y',alpha=.18)
            ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            continue
        points=[(int(r['step']),float(r[key])) for r in rows if key in r]
        if points:
            x,y=zip(*points)
            ax.plot(x,y,'o-',color='#197c86',linewidth=1.7,markersize=4)
        else:
            ax.text(.5,.5,'未记录',ha='center',transform=ax.transAxes)
        ax.set(title=title,xlabel='优化器步数',xlim=(0.5,20.5))
        ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
        if key=='learning_rate':
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    last=max(int(r['step']) for r in rows)
    fig.suptitle(f'教师数据刷新 SFT｜已记录 {last}/20 步｜原始曲线，未平滑',fontsize=16)
    fig.savefig(output/'sft_training_curves.png',dpi=170)
    plt.close(fig)
    fields=['step']+[m[0] for m in metrics]+['entropy','num_tokens','epoch']
    with (output/'training_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    summary={'source':str(source.resolve()),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'recorded_steps':len(rows),'last_step':last,'first':rows[0],'last':rows[-1],'notes':['Console values may be rounded.','Token accuracy is teacher-forced, not task success.','No intermediate validation was logged.']}
    if validation is not None:
        summary['validation_loss_before_after']=validation
        summary['validation_sources']=[{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [eval_before,eval_after]]
    (output/'plot_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--eval-before',type=Path)
    parser.add_argument('--eval-after',type=Path)
    args=parser.parse_args()
    render(args.source,args.output,args.eval_before,args.eval_after)
