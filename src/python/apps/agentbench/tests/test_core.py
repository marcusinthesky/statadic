import copy
import importlib.util
import json

import numpy as np
import pytest
from scipy import stats

from core import catalogue, design, grade, prepare_csv, read_json, read_regular, reviewed, same, schema, witness
from reporting import aggregate, condition_differences, failure_category, summarize, trace_facts


def fixture(task_id, n=90):
    t = catalogue()[task_id]
    rng = np.random.default_rng(17)
    rows=[]
    import itertools
    combinations=list(itertools.product(*(t['factors'][k] for k in t['factor_order']))) or [()]
    for i in range(n):
        r = {'row_id':i+1,**{k:str(rng.normal()) for k in t['x']}}
        r.update(dict(zip(t['factor_order'],combinations[i%len(combinations)],strict=True)))
        rows.append(r)
    x,_=design(t,rows)
    eta=x @ np.linspace(.1,.4,x.shape[1])
    if t['model']=='logit':
        y=rng.binomial(1,1/(1+np.exp(-eta)))
    elif t['model']=='poisson':
        y=rng.poisson(np.exp(eta))
    else:
        y=eta+rng.normal(size=n)
        if t.get('log_response'):
            y=np.exp(y)
    for r,v in zip(rows,y,strict=True):
        r[t['y']]=('Yes' if v else 'No') if 'event' in t else str(v)
    return t,rows


@pytest.mark.parametrize('task_id',list(catalogue()))
def test_all_contracts_have_finite_synthetic_witnesses(task_id):
    t,rows=fixture(task_id)
    result=witness(t,rows)
    assert result['task_id']==task_id
    assert len(result['row_ids'])==len(rows)
    json.dumps(result,allow_nan=False)
    if importlib.util.find_spec('jsonschema'):
        import jsonschema
        jsonschema.validate(result,schema(result))


@pytest.mark.parametrize('task_id',['PD0007','PD0035','PD0068','PD0003','PD0096','PD0119','PD0128','PD0135','PD0142'])
def test_witness_against_statsmodels(task_id):
    sm=pytest.importorskip('statsmodels.api')
    t,rows=fixture(task_id)
    expected=witness(t,rows)
    x,_=design(t,rows)
    y=np.array([r[t['y']]=='Yes' for r in rows],float) if 'event' in t else np.array([r[t['y']] for r in rows],float)
    if t.get('log_response'):
        y=np.log(y)
    if t['model']=='ols':
        fit=sm.OLS(y,x).fit()
    else:
        fit=sm.GLM(y,x,family=sm.families.Binomial() if t['model']=='logit' else sm.families.Poisson()).fit(tol=1e-10)
    np.testing.assert_allclose(expected['estimates']['coefficients'],fit.params,rtol=1e-4,atol=1e-6)
    np.testing.assert_allclose(expected['inference']['standard_errors'],fit.bse,rtol=1e-4,atol=1e-6)
    np.testing.assert_allclose(expected['inference']['ci95'],fit.conf_int(),rtol=1e-4,atol=1e-6)
    if t.get('diagnostics'):
        influence=fit.get_influence()
        for key, reference in [('leverage',influence.hat_matrix_diag),('studentized_internal',influence.resid_studentized_internal),('studentized_external',influence.resid_studentized_external),('cooks_distance',influence.cooks_distance[0])]:
            np.testing.assert_allclose(expected['estimates']['diagnostics'][key],reference,rtol=1e-8,atol=1e-10)


def test_one_factor_f_matches_scipy():
    t,rows=fixture('PD0088')
    result=witness(t,rows)
    groups=[[float(r['whole_weight']) for r in rows if r['sex']==l] for l in t['factors']['sex']]
    oracle=stats.f_oneway(*groups)
    assert result['inference']['tests']['sex']['statistic']==pytest.approx(oracle.statistic)
    assert result['inference']['tests']['sex']['pvalue']==pytest.approx(oracle.pvalue)


def test_grading_is_partial_and_not_per_observation():
    t,rows=fixture('PD0007')
    expected=witness(t,rows)
    assert grade(expected,expected,True,True)['automatic_grade']==85
    actual=copy.deepcopy(expected)
    actual['estimates']['fitted'][0]+=100
    result=grade(expected,actual,True,True)
    assert result['automatic_grade']==75
    assert result['final_grade'] is None
    actual['row_ids'][0]=False
    assert grade(expected,actual,True,True)['automatic_grade']==10
    assert grade(expected,expected,False,True)['automatic_grade']==0


@pytest.mark.parametrize('text',['{"x": NaN}','{"x": Infinity}','{"x": 1e999}','{"x": 1,"x": 2}'])
def test_invalid_json_rejected(tmp_path,text):
    path=tmp_path/'data.json';path.write_text(text)
    with pytest.raises(ValueError):
        read_json(path)


def test_symlinks_not_read(tmp_path):
    target=tmp_path/'secret';target.write_text('secret')
    link=tmp_path/'output';link.symlink_to(target)
    with pytest.raises(ValueError):
        read_regular(link)


def test_complete_case_policy_and_categories(tmp_path):
    t=catalogue()['PD0035']
    rows=prepare_csv(t,'gender,height,weight,age,unused\nM,2,3,4,\nF,,2,3,x\n',tmp_path/'data.csv')
    assert len(rows)==1 and rows[0]['row_id']==1


def test_source_accounting_negatives_preserve_sign_and_scale(tmp_path):
    task=catalogue()['PD0129']
    rows=prepare_csv(task,'api00,acs_k3,meals,full\n713,(21.00),63,92\n487,18,,0.72\n',tmp_path/'data.csv')
    assert rows==[dict(row_id=1,api00='713',acs_k3='-21.0',meals='63',full='92')]
    # The source's fractional full values are data, not an implicit percent conversion.
    rows=prepare_csv(task,'api00,acs_k3,meals,full\n487,18,91,0.72\n',tmp_path/'fraction.csv')
    assert rows[0]['full']=='0.72'


def test_review_requires_artifact_binding():
    record=grade({'estimates':{'a':1},'inference':{'b':1}},None,False,False)
    review=dict(submission_sha256='a'*64,reviewer='r',notes='checked',interpretation=10,package_policy=5,integrity_violation=False)
    with pytest.raises(ValueError):
        reviewed(record,review,'b'*64)
    review['integrity_violation']=True
    assert reviewed(record,review,'a'*64)['final_grade']==0


def test_missing_usage_is_not_free_and_review_is_required():
    record=dict(task='PD0007',library='statadic',effort='medium',total_tokens=None,estimated_cost_usd=None,elapsed_seconds=5,usage_complete=False)
    scores=dict(automatic_grade=50,final_grade=None)
    with pytest.raises(ValueError):
        aggregate([(record,scores)],['PD0007'])
    row=aggregate([(record,scores)],['PD0007'],True)[0]
    assert row['mean_total_tokens'] is None
    assert row['pareto_mean_total_tokens'] is None
    assert row['grade_max']==85


def test_equal_task_weight_and_no_task_selection_bias():
    records=[]
    for t,score,count in [('PD0007',80,1),('PD0035',40,3)]:
        for _ in range(count):
            records.append((dict(task=t,library='statadic',effort='medium',total_tokens=10,estimated_cost_usd=.1,elapsed_seconds=1,usage_complete=True),dict(automatic_grade=score,final_grade=score)))
    assert aggregate(records,['PD0007','PD0035'])[0]['mean_grade']==60
    with pytest.raises(ValueError):
        aggregate(records,['PD0007'])


def test_boolean_does_not_equal_number_and_shapes_are_exact():
    assert not same(1,True)
    assert not same([1,2],[1])


def docs_records():
    tasks = ['PD0003', 'PD0007', 'PD0119', 'PD0036']
    result = []
    for condition, values in [('minimal', [0, 40, 80, 60]), ('reference', [30, 50, 85, 75])]:
        for library in ['statadic', 'statsmodels']:
            for task, value in zip(tasks, values, strict=True):
                for repetition in range(3):
                    failed = (condition, library, task, repetition) == ('minimal', 'statadic', 'PD0003', 0)
                    record = dict(condition=condition, library=library, task=task, effort='medium',
                                  repetition=repetition, status='failed' if failed else 'completed',
                                  errors=['UsageLimitExceeded: token budget'] if failed else [],
                                  replay_ok=not failed, latex_ok=not failed,
                                  usage_complete=not failed, total_tokens=None if failed else 100,
                                  estimated_cost_usd=None, elapsed_seconds=10+repetition,
                                  usage=dict(input_tokens=10 if failed else 80,
                                             cache_read_tokens=0 if failed else 20,
                                             output_tokens=2 if failed else 20))
                    scores = dict(automatic_grade=value, final_grade=None, review=None,
                                  checks=dict(estimates={'a': value > 0}, inference={'a': value == 85}),
                                  components=dict(reproducibility=0 if failed else 10,
                                                  latex=0 if failed else 10,
                                                  estimates=40 if value > 0 else 0,
                                                  inference=25 if value == 85 else 0))
                    result.append((record, scores))
    return tasks, result


def test_docs_groups_keep_failed_attempts_unknown_usage_and_paired_differences():
    tasks, records = docs_records()
    rows = aggregate(records, tasks, True)
    assert len(rows) == 4
    assert all(row['tasks'] == 4 and row['attempts'] == 12 for row in rows)
    minimal = next(row for row in rows if (row['condition'], row['library']) == ('minimal', 'statadic'))
    assert minimal['mean_grade'] == 45
    assert minimal['completion_rate'] == pytest.approx(11/12)
    assert minimal['failure_categories']['token_budget'] == 1
    assert minimal['mean_total_tokens'] is None
    assert minimal['mean_estimated_cost_usd'] is None
    assert minimal['observed_input_tokens_subtotal'] == 890
    assert minimal['observed_output_tokens_subtotal'] == 222
    assert minimal['incomplete_usage_runs'] == 1
    assert minimal['mean_final_grade'] is None
    differences = condition_differences(rows)
    assert len(differences) == 2
    assert all(row['delta_mean_grade'] == 15 and row['descriptive'] for row in differences)
    statadic = next(row for row in differences if row['library'] == 'statadic')
    assert statadic['delta_mean_total_tokens'] is None
    assert statadic['delta_completion_rate'] == pytest.approx(1/12)
    assert all(row['delta_mean_estimated_cost_usd'] is None for row in differences)
    assert all(row['delta_mean_final_grade'] is None for row in differences)
    with pytest.raises(ValueError, match='both minimal and reference'):
        condition_differences([row for row in rows if row['condition'] == 'minimal'])


def test_condition_metrics_weight_tasks_equally_and_legacy_shape_is_preserved():
    tasks, records = docs_records()
    chosen = [(r, s) for r, s in records if r['condition'] == 'minimal' and r['library'] == 'statadic'
              and r['task'] in tasks[:2] and (r['task'] != tasks[0] or r['repetition'] == 0)]
    row = aggregate(chosen, tasks[:2], True)[0]
    assert row['mean_grade'] == 20
    assert row['completion_rate'] == .5
    legacy = [(dict((k, v) for k, v in r.items() if k != 'condition'), s) for r, s in chosen]
    row = aggregate(legacy, tasks[:2], True)[0]
    assert set(row) == {'library', 'effort', 'tasks', 'attempts', 'mean_grade', 'grade_max',
                        'mean_total_tokens', 'mean_estimated_cost_usd', 'mean_elapsed_seconds',
                        'incomplete_usage_runs', 'grade_ci95', 'pareto_mean_total_tokens',
                        'pareto_mean_estimated_cost_usd'}
    with pytest.raises(ValueError, match='mix conditioned and legacy'):
        aggregate(chosen + legacy, tasks[:2], True)


def test_docs_trace_counters_are_observations_not_compliance_or_success(tmp_path):
    path = tmp_path/'messages.json'
    assert trace_facts(path)['docs_path_calls'] is None
    path.write_text(json.dumps([
        {'parts': [{'part_kind': 'tool-call', 'tool_name': 'read_file', 'tool_call_id': 'a',
                    'args': {'path': '/docs/INDEX.md'}}]},
        {'parts': [{'part_kind': 'tool-return', 'tool_call_id': 'a',
                    'content': 'outside the project root /work'}]},
        {'parts': [{'part_kind': 'tool-call', 'tool_name': 'shell', 'tool_call_id': 'b',
                    'args': json.dumps({'command': 'cat /docs/INDEX.md; print(dir(pkg))'})}]},
        {'parts': [{'part_kind': 'tool-return', 'tool_call_id': 'b', 'content': 'x'*10000}]},
        {'parts': [{'part_kind': 'tool-call', 'tool_name': 'write_file', 'tool_call_id': 'c',
                    'args': {'path': 'solution.py', 'content': 'print(dir(pkg)); # /docs'}}]},
        {'parts': [{'part_kind': 'tool-call', 'tool_name': 'shell', 'tool_call_id': 'd',
                    'args': {'command': None}}]},
        {'parts': [{'part_kind': 'tool-call', 'tool_name': 'read_file', 'tool_call_id': 'e',
                    'args': {'path': []}}]},
        {'parts': [{'part_kind': 'tool-call', 'tool_name': 'shell', 'tool_call_id': 'f',
                    'args': {'command': []}}]},
        {'parts': [{'part_kind': 'tool-call', 'tool_name': 'read_file', 'tool_call_id': 'g',
                    'args': {'path': None}}]},
        {'parts': [{'part_kind': 'tool-call', 'tool_name': 'shell', 'tool_call_id': 'h',
                    'args': 'null'}]},
        {'parts': [{'part_kind': 'tool-call', 'tool_name': 'read_file', 'tool_call_id': 'i',
                    'args': '[]'}]},
    ]))
    assert trace_facts(path) == dict(trace_available=True, docs_path_calls=2, docs_path_returns=2,
                                    docs_path_denials=1, outside_root_denials=1,
                                    introspection_calls=1, large_tool_returns=1)
    for message, category in [('total_tokens_limit', 'token_budget'), ('request_limit', 'request_limit'),
                              ('unknown limit', 'usage_limit')]:
        assert failure_category(dict(status='failed', errors=['UsageLimitExceeded: '+message]), {}) == category


def test_docs_summary_outputs_all_48_attempts_without_changing_grades(tmp_path):
    from run import plan
    tasks, records = docs_records()
    images = {'statadic': 'sha256:'+'a'*64, 'statsmodels': 'sha256:'+'b'*64}
    config = dict(tasks=tasks, conditions={'minimal': images, 'reference': images},
                  efforts=['medium'], repetitions=3, seed=42)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(config=config)))
    by_identity = {(r['condition'], r['library'], r['task'], r['repetition']): (r, s) for r, s in records}
    for job in plan(config):
        record, scores = by_identity[(job['condition'], job['library'], job['task'], job['repetition'])]
        folder = tmp_path/'runs'/job['id']
        folder.mkdir(parents=True)
        (folder/'run.json').write_text(json.dumps(dict(record, **{'id': job['id']})))
        (folder/'grade.json').write_text(json.dumps(scores))
    originals = {path: path.read_bytes() for path in (tmp_path/'runs').glob('*/grade.json')}
    summarize(tmp_path, True)
    output = tmp_path/'summary-automatic'
    for name, count in [('summary', 4), ('attempts', 48), ('per-task', 16),
                        ('condition-differences', 2), ('task-condition-differences', 8)]:
        rows = read_json(output/f'{name}.json')
        assert len(rows) == count
        assert (output/f'{name}.csv').is_file() and (output/f'{name}.md').is_file()
    assert (output/'mean_total_tokens.svg').is_file()
    assert not (output/'mean_estimated_cost_usd.svg').exists()
    assert all(path.read_bytes() == content for path, content in originals.items())
    attempts = read_json(output/'attempts.json')
    assert all(row['final_grade'] is None and not row['reviewed'] for row in attempts)
    assert all(row['estimates_points'] in (0, 40) and row['inference_points'] in (0, 25)
               for row in attempts)
    with pytest.raises(ValueError, match='Incomplete campaign'):
        (tmp_path/'runs'/plan(config)[-1]['id']/'grade.json').unlink()
        summarize(tmp_path, True)
