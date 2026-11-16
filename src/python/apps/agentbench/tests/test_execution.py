"""Native framework regressions without containers, credentials or provider requests."""
import asyncio
import json
from contextlib import asynccontextmanager

import pytest

import run
from core import read_json


@pytest.fixture
def config():
    return dict(
        model='openai-codex:gpt-6-luna',
        images={'statadic': 'sha256:' + 'a' * 64, 'statsmodels': 'sha256:' + 'b' * 64},
        tasks=['PD0007'], efforts=['medium'], repetitions=1, seed=42,
        timeout_seconds=30, request_limit=5, token_limit=1000, pricing=None,
    )


def test_plan_does_not_resolve_model_or_inspect_images(tmp_path, monkeypatch, capsys, config):
    import pydantic_ai.models

    def unexpected(*args, **kwargs):
        pytest.fail('A plan must not resolve credentials or contact a runtime')

    monkeypatch.setattr(run, 'checked', lambda campaign: {'config': config})
    monkeypatch.setattr(pydantic_ai.models, 'infer_model', unexpected)
    monkeypatch.setattr(run.subprocess, 'run', unexpected)

    asyncio.run(run.evaluate(tmp_path, execute=False))

    assert 'Plan only' in capsys.readouterr().out
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('conditions',[False,True])
def test_evaluation_shares_one_model_for_all_attempts(tmp_path, monkeypatch, config, conditions):
    import pydantic_ai.models
    from pydantic_ai.models.test import TestModel

    events = []
    resolutions = []
    seen = []
    inspected = []
    if conditions:
        minimal = config.pop('images')
        config['conditions'] = dict(minimal=minimal,
            reference={'statadic':'sha256:'+'c'*64,'statsmodels':'sha256:'+'d'*64})

    class SharedModel(TestModel):
        async def __aenter__(self):
            events.append('enter')
            return await super().__aenter__()

        async def __aexit__(self, *args):
            events.append('exit')
            return await super().__aexit__(*args)

    model = SharedModel(call_tools=[])

    def resolve(identifier):
        resolutions.append(identifier)
        return model

    async def fake_attempt(job, campaign, settings, shared_model):
        assert events == ['enter']
        seen.append((job, shared_model))
        folder = campaign / 'runs' / job['id']
        folder.mkdir(parents=True)
        return dict(job, actual=None, replay_ok=False, latex_ok=False,
                    elapsed_seconds=0, total_tokens=None, estimated_cost_usd=None)

    monkeypatch.setattr(run, 'checked', lambda campaign: {'config': config})
    monkeypatch.setattr(run, 'read_json', lambda path: {'estimates': {'a': 1}, 'inference': {'b': 1}})
    monkeypatch.setattr(run.subprocess, 'run', lambda args, **kwargs: inspected.append(args[-1]))
    monkeypatch.setattr(run, 'attempt', fake_attempt)
    monkeypatch.setattr(pydantic_ai.models, 'infer_model', resolve)

    asyncio.run(run.evaluate(tmp_path, execute=True))

    assert resolutions == [config['model']]
    assert events == ['enter', 'exit']
    assert {job['library'] for job, _ in seen} == {'statadic', 'statsmodels'}
    assert all(shared_model is model for _, shared_model in seen)
    assert len(list((tmp_path / 'runs').glob('*/grade.json'))) == (4 if conditions else 2)
    assert len(set(inspected)) == (4 if conditions else 2)
    assert {job['condition'] for job,_ in seen} == {'minimal','reference'} if conditions else all(
        'condition' not in job for job,_ in seen)


@pytest.fixture
def local_attempt(tmp_path, monkeypatch, config):
    """Use the production accounting hooks with native offline model/tool execution."""
    import pydantic_ai
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import Hooks
    from pydantic_ai.models.function import FunctionModel

    settings = {}

    async def local_probe(value: int) -> str:
        if settings.get('wait_tool'):
            await asyncio.Event().wait()
        return f'probe complete: {value}'

    def local_agent(shared_model, **kwargs):
        return Agent(shared_model, model_settings=kwargs['model_settings'], tools=[local_probe],
                     capabilities=[cap for cap in kwargs['capabilities'] if isinstance(cap,Hooks)])

    @asynccontextmanager
    async def local_sandbox(image, public, work, executable):
        work.mkdir(parents=True)
        if settings.get('fail_setup'):
            raise RuntimeError('sandbox setup failed before a model request')
        yield None, None

    monkeypatch.setattr(pydantic_ai, 'Agent', local_agent)
    monkeypatch.setattr(run, 'sandbox', local_sandbox)

    async def invoke(response, *, overrides=None, job=None, campaign=None, **behavior):
        settings.update(behavior)
        configured = dict(config, **(overrides or {}))
        selected = job or run.plan(configured)[0]
        destination = campaign or tmp_path
        result = await run.attempt(selected,destination,configured,FunctionModel(response))
        return result, destination/'runs'/selected['id']

    return invoke


def test_failed_attempt_retains_partial_native_messages(local_attempt):
    from pydantic_ai.messages import (
        ModelMessagesTypeAdapter, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart,
    )
    from pydantic_ai.usage import RequestUsage

    calls = 0

    async def response(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart('local_probe', {'value':1})],
                                 usage=RequestUsage(input_tokens=7, output_tokens=3))
        raise RuntimeError('provider interrupted')

    result,folder = asyncio.run(local_attempt(response))
    trace = ModelMessagesTypeAdapter.validate_json((folder / 'messages.json').read_bytes())
    parts = [part for message in trace for part in message.parts]
    assert calls == 2
    assert any(isinstance(part, UserPromptPart)
               and part.content == (folder / 'prompt.txt').read_text() for part in parts)
    assert any(isinstance(part, ToolCallPart) and part.tool_name == 'local_probe' for part in parts)
    assert any(isinstance(part, ToolReturnPart) and part.content == 'probe complete: 1' for part in parts)
    assert result['status'] == 'failed'
    assert any('provider interrupted' in error for error in result['errors'])
    assert result['usage']['input_tokens'] == 7
    assert result['usage']['output_tokens'] == 3
    assert result['usage_complete'] is False
    assert result['observed_total_tokens'] == 10
    assert result['request_accounting']['prepared_attempts'] == 2
    assert result['request_accounting']['returned_responses'] == 1
    assert result['request_accounting']['unresolved_attempts'] == 1
    assert result['usage_incomplete_reasons'] == ['unresolved_prepared_attempts']
    assert result['total_tokens'] is result['estimated_cost_usd'] is None
    assert read_json(folder / 'run.json') == result


@pytest.mark.parametrize('outcome',['success','tokens','requests','tool_arguments','request_timeout','tool_timeout'])
def test_native_usage_accounting_distinguishes_execution_from_metering(local_attempt, outcome):
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.usage import RequestUsage

    calls = 0

    async def response(messages, info):
        nonlocal calls
        calls += 1
        if outcome == 'request_timeout' and calls == 2:
            await asyncio.Event().wait()
        if outcome == 'tool_arguments':
            parts = [ToolCallPart('local_probe',{'value':'invalid integer'})]
        elif calls == 1 and outcome != 'tokens':
            parts = [ToolCallPart('local_probe',{'value':1})]
        else:
            parts = [TextPart('done')]
        return ModelResponse(parts,usage=RequestUsage(input_tokens=7,output_tokens=3))

    overrides = dict(token_limit=9) if outcome == 'tokens' else dict(request_limit=1) if outcome == 'requests' else {}
    if outcome.endswith('timeout'):
        overrides['timeout_seconds'] = 1
    result,folder = asyncio.run(local_attempt(response,overrides=overrides,
                                            wait_tool=outcome == 'tool_timeout'))
    assert result['status'] == ('completed' if outcome == 'success' else 'failed')
    expected_responses = 2 if outcome in ('success','tool_arguments') else 1
    assert result['usage']['requests'] == expected_responses
    assert result['usage']['input_tokens'] == 7*expected_responses
    assert result['observed_total_tokens'] == 10*expected_responses
    assert result['request_accounting']['returned_responses'] == expected_responses
    assert result['request_accounting']['unresolved_attempts'] == (outcome == 'request_timeout')
    assert result['usage_complete'] == (outcome != 'request_timeout')
    assert result['total_tokens'] == (None if outcome == 'request_timeout' else 10*expected_responses)
    assert result['estimated_cost_usd'] is None
    events = [json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    assert sum(event['event'] == 'returned' for event in events) == expected_responses
    assert sum(event['event'] == 'prepared' for event in events) == expected_responses+(outcome == 'request_timeout')
    assert all(set(event) <= {'event','step','attempt','elapsed_seconds','model','provider',
                             'provider_response_id','state','usage','error_type'} for event in events)
    if outcome == 'tokens':
        assert 'total_tokens_limit' in result['errors'][0]
    elif outcome == 'requests':
        assert 'request_limit' in result['errors'][0]
    elif outcome == 'tool_arguments':
        assert 'UnexpectedModelBehavior' in result['errors'][0]
    elif outcome.endswith('timeout'):
        assert result['errors'][0].startswith('TimeoutError:')


def test_returned_response_without_usage_remains_unknown(local_attempt):
    from pydantic_ai.exceptions import UsageLimitUnavailableWarning
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.usage import RequestUsage

    async def response(messages, info):
        return ModelResponse([TextPart('done')],usage=RequestUsage(unmeasured_requests=1))

    with pytest.warns(UsageLimitUnavailableWarning):
        result,_ = asyncio.run(local_attempt(response))
    assert result['status'] == 'completed'
    assert result['request_accounting']['unresolved_attempts'] == 0
    assert result['request_accounting']['unmeasured_responses'] == 1
    assert result['usage_incomplete_reasons'] == ['responses_missing_usage']
    assert result['usage_complete'] is False
    assert result['total_tokens'] is None


def test_setup_failure_records_known_zero_model_usage(local_attempt):
    async def unexpected(messages, info):
        pytest.fail('A failed sandbox must not call a model')

    result,_ = asyncio.run(local_attempt(unexpected,fail_setup=True))
    assert result['status'] == 'failed'
    assert result['request_accounting']['prepared_attempts'] == 0
    assert result['usage']['requests'] == result['total_tokens'] == 0
    assert result['usage_complete'] is True


def test_conditions_use_their_own_images_and_identical_discovery_prompt(local_attempt, config, tmp_path):
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.usage import RequestUsage

    minimal = config.pop('images')
    config['conditions'] = dict(minimal=minimal,
        reference={'statadic':'sha256:'+'c'*64,'statsmodels':'sha256:'+'d'*64})

    async def response(messages, info):
        return ModelResponse([TextPart('done')],usage=RequestUsage(input_tokens=1,output_tokens=1))

    prompts = []
    for condition in ('minimal','reference'):
        job = next(job for job in run.plan(config) if job['condition'] == condition and job['library'] == 'statadic')
        result,folder = asyncio.run(local_attempt(response,job=job,campaign=tmp_path/condition))
        assert result['condition'] == condition
        assert result['image'] == config['conditions'][condition]['statadic']
        prompts.append((folder/'prompt.txt').read_text())
    assert prompts[0] == prompts[1]
    assert ('Before implementing the analysis, read /docs/INDEX.md for the assigned statistical '
            'library and consult its relevant API examples. Perform fitting and inference with '
            'that library.') in prompts[0]


def test_invalid_condition_config_fails_before_model_resolution(tmp_path, monkeypatch, config):
    import pydantic_ai.models

    config['conditions'] = dict(minimal=config['images'],reference=config['images'])
    monkeypatch.setattr(run,'checked',lambda campaign:{'config':config})

    def unexpected(*args, **kwargs):
        pytest.fail('Invalid configuration must fail before credentials or the runtime are inspected')

    monkeypatch.setattr(pydantic_ai.models,'infer_model',unexpected)
    monkeypatch.setattr(run.subprocess,'run',unexpected)
    with pytest.raises(ValueError,match='either legacy images'):
        asyncio.run(run.evaluate(tmp_path,execute=True))
