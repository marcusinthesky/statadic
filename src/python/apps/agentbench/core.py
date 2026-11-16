"""Data contracts and grading; no model, sandbox or package under test is imported."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import stat
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats

HERE = Path(__file__).parent
REVISION = '3194677cc1c1590a95e2bf482af57dd7bb37f35a'
SOURCE = f'https://raw.githubusercontent.com/yili-hong/StatLLM/{REVISION}/Statistical_Analysis_Tasks'
IDENTITY = ('schema_version', 'task_id', 'row_ids', 'parameter_names', 'model', 'reference')


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_json(path):
    def reject(value):
        raise ValueError(f'Invalid JSON numeric literal: {value}')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    result = json.loads(read_regular(Path(path)).decode(), parse_constant=reject, object_pairs_hook=pairs)
    # Reject overflowing exponents too (json.loads('1e999') is otherwise infinity).
    json.dumps(result, allow_nan=False)
    return result


def read_regular(path, limit=20_000_000):
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f'Symlink is not an artifact: {path}')
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError(f'Expected bounded regular file: {path}')
    return path.read_bytes()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def catalogue():
    records = read_json(HERE / 'tasks.json')
    result = {t['id']: t for t in records}
    if len(result) != 20 or len(records) != 20:
        raise ValueError('Expected exactly 20 unique tasks')
    return result


def prepare_csv(task, raw, destination):
    required = list(dict.fromkeys([task['y'], *task['x'], *task['squares'], *task['factor_order']]))
    rows = []
    for index, source in enumerate(csv.DictReader(raw.splitlines()), 1):
        row = {k.strip().lower(): (v or '').strip() for k, v in source.items() if k and k.strip()}
        if any(row.get(k) != v for k, v in task.get('where', {}).items()):
            continue
        if any(k not in row for k in required):
            raise ValueError('Missing required source columns')
        if any(row[k].lower() in ('', '.', 'na', 'nan', 'null') for k in required):
            continue
        for key, levels in task['factors'].items():
            if row[key] not in levels:
                try:
                    row[key] = f'{float(row[key]):g}'
                except ValueError:
                    pass
            if row[key] not in levels:
                raise ValueError(f'Unexpected category {key}={row[key]}')
        rows.append({'row_id': index, **{k: row[k] for k in required}})
    if not rows:
        raise ValueError('No complete observations')
    with destination.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ['row_id', *required])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def design(task, rows):
    columns, names = [np.ones(len(rows))], ['intercept']
    for key in task['x']:
        columns.append(np.array([float(r[key]) for r in rows]))
        names.append(key)
    for key in task['squares']:
        columns.append(np.array([float(r[key]) ** 2 for r in rows]))
        names.append(key + '^2')
    for key in task['factor_order']:
        for level in task['factors'][key][1:]:
            columns.append(np.array([r[key] == level for r in rows], dtype=float))
            names.append(f'{key}[{level}]')
    if task['interaction']:
        a, b = task['factor_order']
        for left, right in itertools.product(task['factors'][a][1:], task['factors'][b][1:]):
            columns.append(np.array([r[a] == left and r[b] == right for r in rows], dtype=float))
            names.append(f'{a}[{left}]:{b}[{right}]')
    x = np.column_stack(columns)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite design')
    return x, names


def witness(task, rows):
    """Independent least-squares / likelihood witness, not a SAS-output emulator."""
    x, names = design(task, rows)
    y = np.array([r[task['y']] == task['event'] for r in rows], float) if 'event' in task else np.array([r[task['y']] for r in rows], float)
    if task.get('log_response'):
        if np.any(y <= 0):
            raise ValueError('Log response requires positive observations')
        y = np.log(y)
    n, p = x.shape
    if not np.isfinite(y).all() or np.linalg.matrix_rank(x) != p or n <= p:
        raise ValueError('Require finite response, full rank and positive residual degrees of freedom')
    if task['model'] == 'ols':
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        fitted = x @ beta
        residual = y - fitted
        variance = residual @ residual / (n - p)
        q, r = np.linalg.qr(x)
        inv_r = np.linalg.solve(r, np.eye(p))
        cov = variance * inv_r @ inv_r.T
        quantile = stats.t.ppf(.975, n - p)
    else:
        scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1.)
        z = x / scale
        logistic = task['model'] == 'logit'
        if logistic and (not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2):
            raise ValueError('Expected nonconstant binary response')
        if not logistic and (np.any(y < 0) or np.any(y != np.floor(y)) or y.sum() == 0):
            raise ValueError('Expected nondegenerate counts')
        def objective(theta):
            eta = z @ theta
            mu = special.expit(eta) if logistic else np.exp(eta)
            loss = np.logaddexp(0, eta) if logistic else mu
            return np.sum(loss - y*eta), z.T @ (mu-y)
        solved = optimize.minimize(objective, np.zeros(p), jac=True, method='BFGS', options={'gtol': 1e-9, 'maxiter': 1000})
        if np.max(np.abs(objective(solved.x)[1])) > 1e-5:
            raise ValueError('GLM witness failed convergence check')
        beta = solved.x / scale
        fitted = special.expit(x @ beta) if logistic else np.exp(x @ beta)
        weight = fitted*(1-fitted) if logistic else fitted
        information = z.T @ (weight[:, None] * z)
        if np.linalg.eigvalsh(information).min() < 1e-10:
            raise ValueError('Near-singular/separated GLM witness')
        cov = np.linalg.inv(information) / scale[:, None] / scale[None, :]
        residual = y-fitted
        quantile = stats.norm.ppf(.975)
    def intervals(matrix, critical=quantile):
        effect = matrix @ beta
        se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', matrix, cov, matrix), 0))
        return np.column_stack([effect-critical*se, effect+critical*se]).tolist()
    estimates = dict(coefficients=beta.tolist(), fitted=fitted.tolist(), residuals=residual.tolist())
    inference = dict(standard_errors=np.sqrt(np.diag(cov)).tolist(), ci95=intervals(np.eye(p)))
    if task['model'] == 'ols':
        estimates['r_squared'] = float(1 - residual @ residual / np.sum((y-y.mean())**2))
    if task.get('diagnostics'):
        h = (q*q).sum(axis=1)
        student = residual / np.sqrt(variance*(1-h))
        deleted = (residual @ residual - residual**2/(1-h))/(n-p-1)
        estimates['diagnostics'] = dict(leverage=h.tolist(), studentized_internal=student.tolist(), studentized_external=(residual/np.sqrt(deleted*(1-h))).tolist(), cooks_distance=(student**2*h/(p*(1-h))).tolist())
    if task.get('anova') or task.get('odds_ratios'):
        factors = task['factor_order']
        grid = [dict(zip(factors, levels, strict=True), **{k: float(np.mean([float(row[k]) for row in rows])) for k in task['x']}) for levels in itertools.product(*(task['factors'][k] for k in factors))]
        cells, _ = design(task, grid)
        marginal = {k: np.stack([cells[[r[k] == level for r in grid]].mean(axis=0) for level in task['factors'][k]]) for k in factors}
        if task.get('anova'):
            tests = {}
            restrictions = {k: m[1:]-m[0] for k, m in marginal.items()}
            if task['interaction']:
                restrictions['interaction'] = np.eye(p)[[':' in name for name in names]]
            for key, matrix in restrictions.items():
                effect = matrix @ beta
                rank = np.linalg.matrix_rank(matrix)
                f = float(effect @ np.linalg.solve(matrix @ cov @ matrix.T, effect)/rank)
                tests[key] = dict(statistic=f, df1=int(rank), df2=n-p, pvalue=float(stats.f.sf(f, rank, n-p)))
            inference['tests'] = tests
        if task.get('marginals'):
            estimates['marginal_means'], inference['marginal_ci95'], inference['pairwise'] = {}, {}, {}
            for key, matrix in {**marginal, 'cells': cells}.items():
                estimates['marginal_means'][key] = (matrix @ beta).tolist()
                inference['marginal_ci95'][key] = intervals(matrix)
                differences = np.stack([matrix[i]-matrix[j] for i, j in itertools.combinations(range(len(matrix)), 2)])
                inference['pairwise'][key] = np.column_stack([differences @ beta, intervals(differences)]).tolist()
        if task.get('odds_ratios'):
            matrix = marginal['treatment']
            differences = np.stack([matrix[i]-matrix[j] for i,j in itertools.combinations(range(3), 2)])
            inference['treatment_odds_ratios'] = np.exp(np.column_stack([differences @ beta, intervals(differences, stats.norm.ppf(1-.05/6))])).tolist()
    result = dict(schema_version=1, task_id=task['id'], row_ids=[int(r['row_id']) for r in rows], parameter_names=names, model=task['model'], reference='student_t' if task['model']=='ols' else 'normal', estimates=estimates, inference=inference)
    json.dumps(result, allow_nan=False)
    return result


def schema(value):
    if isinstance(value, dict):
        return dict(type='object', required=list(value), additionalProperties=False, properties={k:schema(v) for k,v in value.items()})
    if isinstance(value, list):
        return dict(type='array', minItems=len(value), maxItems=len(value), items=schema(value[0]) if value else {})
    return dict(type='string' if isinstance(value, str) else 'integer' if isinstance(value, int) else 'number')


def same(a, b, rtol=1e-4, atol=1e-6):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys()==b.keys() and all(same(v,b[k],rtol,atol) for k,v in a.items())
    if isinstance(a, list):
        return isinstance(b,list) and len(a)==len(b) and all(same(x,y,rtol,atol) for x,y in zip(a,b,strict=True))
    if isinstance(a,(int,float)):
        return type(b) in (int,float) and math.isfinite(b) and math.isclose(a,b,rel_tol=rtol,abs_tol=atol)
    return type(a) is type(b) and a==b


def leaves(value, prefix=''):
    result = {}
    for k,v in value.items():
        key = f'{prefix}.{k}'
        result.update(leaves(v,key) if isinstance(v,dict) else {key:v})
    return result


def grade(expected, actual, replay_ok, latex_ok):
    identity = isinstance(actual,dict) and expected.keys()==actual.keys() and all(same(expected[k],actual.get(k),0,0) for k in IDENTITY)
    scores = dict(reproducibility=10*bool(replay_ok and identity), latex=10*bool(replay_ok and latex_ok))
    checks = {}
    for group, points in [('estimates',40),('inference',25)]:
        wanted = leaves(expected[group])
        got = leaves(actual[group]) if isinstance(actual,dict) and isinstance(actual.get(group),dict) else {}
        checks[group] = {k:bool(replay_ok and identity and same(v,got.get(k))) for k,v in wanted.items()}
        scores[group] = points*sum(checks[group].values())/len(wanted)
    return dict(automatic_grade=sum(scores.values()), automatic_max=85, components=scores, checks=checks, final_grade=None, review=None)


def reviewed(grade_record, review, submission_sha):
    if not isinstance(submission_sha, str) or len(submission_sha) != 64 or review.get('submission_sha256') != submission_sha or not review.get('reviewer') or not review.get('notes'):
        raise ValueError('Review requires exact artifact identity, reviewer and rationale')
    if type(review.get('interpretation')) is not int or review['interpretation'] not in (0,5,10) or type(review.get('package_policy')) is not int or review['package_policy'] not in (0,5) or type(review.get('integrity_violation')) is not bool:
        raise ValueError('Invalid review anchors')
    return dict(grade_record, review=review, final_grade=0 if review['integrity_violation'] else grade_record['automatic_grade']+review['interpretation']+review['package_policy'])
