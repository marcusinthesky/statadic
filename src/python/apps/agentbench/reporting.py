"""Equal-task quality/cost summaries; incomplete usage is never a zero cost."""
from __future__ import annotations

import csv
import itertools
import json
import re
from collections import defaultdict

import numpy as np

from core import catalogue, read_json, save


def failure_category(record, scores):
    """Descriptive precedence; additional failed stages remain in the run record."""
    errors = '\n'.join(record.get('errors', []))
    if 'UsageLimitExceeded' in errors:
        if 'request_limit' in errors:
            return 'request_limit'
        if 'tokens_limit' in errors or 'token budget' in errors.lower():
            return 'token_budget'
        return 'usage_limit'
    if 'TimeoutError' in errors:
        return 'timeout'
    if record.get('status') == 'failed':
        return 'tool_error' if 'exceeded max retries' in errors else 'agent_error'
    if not record.get('replay_ok'):
        return 'replay_contract'
    if not record.get('latex_ok'):
        return 'latex'
    if any(not passed for group in scores.get('checks', {}).values() for passed in group.values()):
        return 'numerical'
    return 'none'


def trace_facts(path):
    """Count recorded path mentions and returns, never infer package compliance."""
    keys = ('docs_path_calls', 'docs_path_returns', 'docs_path_denials',
            'outside_root_denials', 'introspection_calls', 'large_tool_returns')
    result = dict(trace_available=path.exists(), **dict.fromkeys(keys, None))
    if not result['trace_available']:
        return result
    result.update(dict.fromkeys(keys, 0))
    calls = {}
    for message in read_json(path):
        for part in message.get('parts', []):
            if part.get('part_kind') == 'tool-call':
                args = part.get('args', {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                tool = part.get('tool_name')
                command = args.get('command', '') if isinstance(args, dict) else ''
                target = args.get('path', '') if isinstance(args, dict) else ''
                command = command if isinstance(command, str) else ''
                target = target if isinstance(target, str) else ''
                docs = bool(re.search(r'/docs(?:/|\b)', command if tool == 'shell' else target))
                calls[part.get('tool_call_id')] = docs
                result['docs_path_calls'] += docs
                result['introspection_calls'] += bool(tool == 'shell' and re.search(
                    r'\b(?:dir|vars|getattr)\s*\(|inspect\.(?:signature|getsource|getdoc)|__doc__', command))
            elif part.get('part_kind') == 'tool-return':
                content = part.get('content', '')
                content = content if isinstance(content, str) else json.dumps(content)
                denied = 'outside the project root' in content
                docs = calls.get(part.get('tool_call_id'), False)
                result['docs_path_returns'] += docs
                result['docs_path_denials'] += docs and denied
                result['outside_root_denials'] += denied
                result['large_tool_returns'] += len(content) >= 10000
    return result


def _measurements(items):
    records = [record for record, _ in items]
    def mean(values):
        if any(value is None for value in values):
            return None
        per_task = defaultdict(list)
        for record, value in zip(records, values, strict=True):
            per_task[record['task']].append(value)
        return float(np.mean([np.mean(values) for values in per_task.values()]))
    categories = [failure_category(record, scores) for record, scores in items]
    row = dict(completion_rate=mean([r.get('status') == 'completed' for r in records]),
               replay_rate=mean([bool(r.get('replay_ok')) for r in records]),
               latex_rate=mean([bool(r.get('latex_ok')) for r in records]),
               failure_rate=mean([c != 'none' for c in categories]),
               budget_failure_rate=mean([c in ('token_budget', 'request_limit', 'usage_limit') for c in categories]),
               reviewed_runs=sum(s.get('review') is not None for _, s in items),
               failure_categories={c: categories.count(c) for c in sorted(set(categories))})
    numeric = [all(v for group in s.get('checks', {}).values() for v in group.values())
               if s.get('checks') else None for _, s in items]
    row['numerical_full_credit_rate'] = mean(numeric)
    row['mean_automatic_grade'] = mean([s.get('automatic_grade') for _, s in items])
    row['mean_final_grade'] = mean([s.get('final_grade') for _, s in items])
    for key in ('reproducibility', 'estimates', 'inference', 'latex'):
        row['mean_'+key+'_points'] = mean([s.get('components', {}).get(key) for _, s in items])
    for key in ('estimates', 'inference'):
        values = [all(s['checks'][key].values()) if s.get('checks', {}).get(key) else None
                  for _, s in items]
        row[key+'_full_credit_rate'] = mean(values)
    for name, key in [('input', 'input_tokens'), ('cached_input', 'cache_read_tokens'),
                      ('output', 'output_tokens')]:
        observed = [r.get('usage', {}).get(key) for r in records]
        row[f'observed_{name}_tokens_runs'] = sum(v is not None for v in observed)
        row[f'observed_{name}_tokens_subtotal'] = (sum(v for v in observed if v is not None)
                                                  if any(v is not None for v in observed) else None)
    row['trace_available_runs'] = sum(r.get('trace_facts', {}).get('trace_available', False)
                                      for r in records)
    for key in ('docs_path_calls', 'docs_path_returns', 'docs_path_denials',
                'outside_root_denials', 'introspection_calls', 'large_tool_returns'):
        values = [r.get('trace_facts', {}).get(key) for r in records]
        row[key] = None if any(v is None for v in values) else sum(values)
    return row


def aggregate(records, tasks, automatic=False):
    """Average repeats within each task, then weight all selected tasks equally."""
    records = list(records)
    conditioned = any('condition' in record for record, _ in records)
    if conditioned and any('condition' not in record for record, _ in records):
        raise ValueError('Do not mix conditioned and legacy runs')
    groups = defaultdict(list)
    for record, scores in records:
        value = scores['automatic_grade'] if automatic else scores['final_grade']
        if value is None:
            raise ValueError('Review every run first, or explicitly use --automatic (/85)')
        groups[(record.get('condition'),record['library'],record['effort'])].append((record,value))
    rows = []
    for (condition,library,effort), items in sorted(groups.items()):
        per_task = defaultdict(list)
        for record,score in items:
            per_task[record['task']].append((record,score))
        if set(per_task)!=set(tasks):
            raise ValueError('Frontier points must use exactly the same tasks')
        row = dict(library=library, effort=effort, tasks=len(tasks), attempts=len(items),
                   mean_grade=float(np.mean([np.mean([s for _,s in per_task[t]]) for t in tasks])),
                   grade_max=85 if automatic else 100)
        for key in ('total_tokens','estimated_cost_usd','elapsed_seconds'):
            missing = any(r.get(key) is None or (key != 'elapsed_seconds' and not r.get('usage_complete', False)) for r,_ in items)
            row['mean_'+key] = None if missing else float(np.mean([np.mean([r[key] for r,_ in per_task[t]]) for t in tasks]))
        row['incomplete_usage_runs'] = sum(not r.get('usage_complete',False) for r,_ in items)
        if conditioned:
            row = dict(condition=condition, **row)
            row.update(_measurements([(record, scores) for record, scores in records
                                      if (record.get('condition'), record['library'], record['effort'])
                                      == (condition, library, effort)]))
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


def condition_differences(rows):
    """Reference minus minimal; missing means remain missing, never available-case."""
    groups = defaultdict(dict)
    for row in rows:
        if 'condition' in row:
            groups[(row.get('task'), row['library'], row['effort'])][row['condition']] = row
    result = []
    for (task, library, effort), conditions in sorted(groups.items()):
        if set(conditions) != {'minimal', 'reference'}:
            raise ValueError('A/B differences require both minimal and reference conditions')
        a, b = conditions['minimal'], conditions['reference']
        if a['tasks'] != b['tasks'] or a['attempts'] != b['attempts']:
            raise ValueError('A/B differences require balanced tasks and attempts')
        row = dict(library=library, effort=effort, comparison='reference-minus-minimal',
                   descriptive=True, tasks=a['tasks'], attempts_per_condition=a['attempts'],
                   grade_max=a['grade_max'])
        if task is not None:
            row = dict(task=task, **row)
        for key in ('mean_grade', 'mean_automatic_grade', 'mean_final_grade', 'mean_total_tokens', 'mean_estimated_cost_usd',
                    'mean_elapsed_seconds', 'completion_rate', 'replay_rate', 'latex_rate',
                    'numerical_full_credit_rate', 'budget_failure_rate'):
            row['delta_'+key] = None if a.get(key) is None or b.get(key) is None else b[key]-a[key]
        for condition, source in [('minimal', a), ('reference', b)]:
            row[condition+'_incomplete_usage_runs'] = source['incomplete_usage_runs']
        result.append(row)
    return result


def _tables(output, name, rows, columns=None):
    save(output/f'{name}.json', rows)
    if not rows:
        return
    with (output/f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = columns or list(rows[0])
    def show(value):
        if value is None:
            return 'unavailable'
        return f'{value:.6g}' if isinstance(value, float) else str(value).replace('|', '\\|')
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join('---' for _ in columns)+' |']
    lines.extend('| '+' | '.join(show(row.get(key)) for key in columns)+' |' for row in rows)
    (output/f'{name}.md').write_text('\n'.join(lines)+'\n')


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
        records.append((dict(record, trace_facts=trace_facts(folder/'messages.json')),
                        read_json(folder/'grade.json')))
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
        library = f"{row['condition']} / {row['library']}" if 'condition' in row else row['library']
        lines.append(f"| {library} | {row['effort']} | {row['mean_grade']:.2f}/{row['grade_max']} | {show(row['mean_total_tokens'])} | {show(row['mean_estimated_cost_usd'])} | {row['incomplete_usage_runs']} |")
    if config.get('conditions'):
        lines.extend(['', 'Exploratory documentation A/B pilot. Differences and dataset-cluster intervals are descriptive; no superiority claim.',
                      'Observed token subtotals may be partial. Missing full usage/cost means remain unavailable.',
                      'Documentation counters describe tool path mentions and returns, not successful reading or package compliance.',
                      'Documentation corpus sizes are not matched between libraries; this compares the frozen bundles, not equal bytes or model knowledge.'])
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    attempts = []
    for record, scores in records:
        row = {key: record.get(key) for key in ('id', 'task', 'library', 'effort', 'repetition')}
        if 'condition' in record:
            row = dict(condition=record['condition'], **row)
        row.update(automatic_grade=scores['automatic_grade'], final_grade=scores['final_grade'],
                   reviewed=scores.get('review') is not None, status=record.get('status'),
                   replay_ok=record.get('replay_ok'), latex_ok=record.get('latex_ok'),
                   failure_category=failure_category(record, scores), errors=record.get('errors', []))
        for key in ('reproducibility', 'estimates', 'inference', 'latex'):
            row[key+'_points'] = scores.get('components', {}).get(key)
        for key in ('estimates', 'inference'):
            checks = scores.get('checks', {}).get(key)
            row[key+'_full_credit'] = all(checks.values()) if checks else None
        row['failed_quantities'] = [f'{group}{key}' for group, checks in scores.get('checks', {}).items()
                                    for key, passed in checks.items() if not passed]
        row.update({key: record.get(key) for key in ('usage_complete', 'total_tokens',
                    'estimated_cost_usd', 'elapsed_seconds', 'usage_incomplete_reasons',
                    'request_accounting')})
        for name, key in [('input', 'input_tokens'), ('cached_input', 'cache_read_tokens'), ('output', 'output_tokens')]:
            row[f'observed_{name}_tokens'] = record.get('usage', {}).get(key)
        row.update(record['trace_facts'])
        attempts.append(row)
    _tables(output, 'attempts', attempts, (['condition'] if config.get('conditions') else [])+
            ['id', 'task', 'library', 'automatic_grade', 'final_grade', 'status', 'replay_ok',
             'latex_ok', 'estimates_points', 'inference_points', 'failure_category',
             'usage_complete', 'total_tokens', 'elapsed_seconds'])
    per_task = [dict(task=task, **row) for task in config['tasks']
                for row in aggregate([(r, s) for r, s in records if r['task'] == task], [task], automatic)]
    _tables(output, 'per-task', per_task, (['condition'] if config.get('conditions') else [])+
            ['task', 'library', 'effort', 'attempts', 'mean_grade', 'mean_total_tokens',
             'mean_estimated_cost_usd', 'mean_elapsed_seconds', 'incomplete_usage_runs'])
    if config.get('conditions'):
        _tables(output, 'condition-differences', condition_differences(rows))
        _tables(output, 'task-condition-differences', condition_differences(per_task))
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
            name = f"{row['library']} / {row['effort']}"
            if 'condition' in row:
                name = f"{row['condition']} / {name}"
            ax.annotate(name,(row[key],row['mean_grade']),xytext=(4,4),textcoords='offset points')
            ax.vlines(row[key],*row['grade_ci95'],alpha=.4)
        ax.set(xlabel=label,ylabel=f"Mean grade / {85 if automatic else 100}",title='Selected StatLLM tasks: quality–cost trade-off')
        fig.tight_layout()
        fig.savefig(output/f'{key}.svg')
        plt.close(fig)
    print(output)
