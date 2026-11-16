"""Equal-task quality/cost summaries; incomplete usage is never a zero cost."""
from __future__ import annotations

import csv
import itertools
from collections import defaultdict

import numpy as np

from core import catalogue, read_json, save


def aggregate(records, tasks, automatic=False):
    """Average repeats within each task, then weight all selected tasks equally."""
    groups = defaultdict(list)
    for record, scores in records:
        value = scores['automatic_grade'] if automatic else scores['final_grade']
        if value is None:
            raise ValueError('Review every run first, or explicitly use --automatic (/85)')
        groups[(record['library'],record['effort'])].append((record,value))
    rows = []
    for (library,effort), items in sorted(groups.items()):
        per_task = defaultdict(list)
        for record,score in items:
            per_task[record['task']].append((record,score))
        if set(per_task)!=set(tasks):
            raise ValueError('Frontier points must use exactly the same tasks')
        row = dict(library=library, effort=effort, tasks=len(tasks), attempts=len(items),
                   mean_grade=float(np.mean([np.mean([s for _,s in per_task[t]]) for t in tasks])),
                   grade_max=85 if automatic else 100)
        for key in ('total_tokens','estimated_cost_usd','elapsed_seconds'):
            row['mean_'+key] = None if any(r.get(key) is None for r,_ in items) else float(np.mean([np.mean([r[key] for r,_ in per_task[t]]) for t in tasks]))
        row['incomplete_usage_runs'] = sum(not r.get('usage_complete',False) for r,_ in items)
        # Dataset-cluster percentile intervals, descriptive for this selected suite.
        clusters = defaultdict(list)
        for task in tasks:
            clusters[catalogue()[task]['dataset']].append(float(np.mean([s for _,s in per_task[task]])))
        rng = np.random.default_rng(42)
        values = list(clusters.values())
        samples = [np.mean(list(itertools.chain.from_iterable(values[i] for i in rng.integers(len(values),size=len(values))))) for _ in range(1000)]
        row['grade_ci95'] = np.quantile(samples,[.025,.975]).tolist()
        rows.append(row)
    for key in ('mean_total_tokens','mean_estimated_cost_usd'):
        for row in rows:
            row['pareto_'+key] = None if row[key] is None else not any(
                other[key] is not None and other[key]<=row[key] and other['mean_grade']>=row['mean_grade']
                and (other[key]<row[key] or other['mean_grade']>row['mean_grade']) for other in rows)
    return rows


def summarize(campaign, automatic=False):
    from run import plan
    config = read_json(campaign/'manifest.json')['config']
    records = []
    for job in plan(config):
        folder = campaign/'runs'/job['id']
        if not (folder/'run.json').exists() or not (folder/'grade.json').exists():
            raise ValueError(f'Incomplete campaign: {job["id"]}; do not omit failed/unstarted attempts')
        record = read_json(folder/'run.json')
        if any(record[k]!=v for k,v in job.items()):
            raise ValueError('Run identity differs from frozen plan')
        records.append((record,read_json(folder/'grade.json')))
    rows = aggregate(records,config['tasks'],automatic)
    output = campaign/('summary-automatic' if automatic else 'summary-reviewed')
    output.mkdir(exist_ok=False)
    save(output/'summary.json',rows)
    with (output/'summary.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| Library | Effort | Mean grade | Tokens/task | USD/task | Missing usage |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    def show(value):
        return 'unavailable' if value is None else f'{value:.3f}'
    for row in rows:
        lines.append(f"| {row['library']} | {row['effort']} | {row['mean_grade']:.2f}/{row['grade_max']} | {show(row['mean_total_tokens'])} | {show(row['mean_estimated_cost_usd'])} | {row['incomplete_usage_runs']} |")
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,label in [('mean_total_tokens','Mean tokens per task'),('mean_estimated_cost_usd','Estimated USD per task')]:
        available = [r for r in rows if r[key] is not None]
        if not available:
            continue
        fig,ax = plt.subplots()
        for row in available:
            ax.scatter(row[key],row['mean_grade'])
            ax.annotate(f"{row['library']} / {row['effort']}",(row[key],row['mean_grade']),xytext=(4,4),textcoords='offset points')
            ax.vlines(row[key],*row['grade_ci95'],alpha=.4)
        ax.set(xlabel=label,ylabel=f"Mean grade / {85 if automatic else 100}",title='Selected StatLLM tasks: quality–cost trade-off')
        fig.tight_layout()
        fig.savefig(output/f'{key}.svg')
        plt.close(fig)
    print(output)
