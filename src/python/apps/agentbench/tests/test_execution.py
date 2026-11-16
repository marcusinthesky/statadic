"""Native framework regressions without containers, credentials or provider requests."""
import asyncio
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


def test_evaluation_shares_one_model_for_all_attempts(tmp_path, monkeypatch, config):
    import pydantic_ai.models
    from pydantic_ai.models.test import TestModel

    events = []
    resolutions = []
    seen = []

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
        seen.append((job['library'], shared_model))
        folder = campaign / 'runs' / job['id']
        folder.mkdir(parents=True)
        return dict(job, actual=None, replay_ok=False, latex_ok=False,
                    elapsed_seconds=0, total_tokens=None, estimated_cost_usd=None)

    monkeypatch.setattr(run, 'checked', lambda campaign: {'config': config})
    monkeypatch.setattr(run, 'read_json', lambda path: {'estimates': {'a': 1}, 'inference': {'b': 1}})
    monkeypatch.setattr(run.subprocess, 'run', lambda *args, **kwargs: None)
    monkeypatch.setattr(run, 'attempt', fake_attempt)
    monkeypatch.setattr(pydantic_ai.models, 'infer_model', resolve)

    asyncio.run(run.evaluate(tmp_path, execute=True))

    assert resolutions == [config['model']]
    assert events == ['enter', 'exit']
    assert {library for library, _ in seen} == {'statadic', 'statsmodels'}
    assert all(shared_model is model for _, shared_model in seen)
    assert len(list((tmp_path / 'runs').glob('*/grade.json'))) == 2


def test_failed_attempt_retains_partial_native_messages(tmp_path, monkeypatch, config):
    import pydantic_ai
    from pydantic_ai import Agent
    from pydantic_ai.messages import (
        ModelMessagesTypeAdapter, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart,
    )
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.usage import RequestUsage

    calls = 0

    async def response(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([ToolCallPart('local_probe', {})],
                                 usage=RequestUsage(input_tokens=7, output_tokens=3))
        raise RuntimeError('provider interrupted')

    model = FunctionModel(response)

    def local_agent(shared_model, **kwargs):
        assert shared_model is model
        agent = Agent(shared_model, model_settings=kwargs['model_settings'])

        @agent.tool_plain
        def local_probe() -> str:
            return 'probe complete'

        return agent

    @asynccontextmanager
    async def local_sandbox(image, public, work, executable):
        work.mkdir(parents=True)
        yield None, None

    monkeypatch.setattr(pydantic_ai, 'Agent', local_agent)
    monkeypatch.setattr(run, 'sandbox', local_sandbox)
    job = run.plan(config)[0]

    result = asyncio.run(run.attempt(job, tmp_path, config, model))

    folder = tmp_path / 'runs' / job['id']
    trace = ModelMessagesTypeAdapter.validate_json((folder / 'messages.json').read_bytes())
    parts = [part for message in trace for part in message.parts]
    assert calls == 2
    assert any(isinstance(part, UserPromptPart)
               and part.content == (folder / 'prompt.txt').read_text() for part in parts)
    assert any(isinstance(part, ToolCallPart) and part.tool_name == 'local_probe' for part in parts)
    assert any(isinstance(part, ToolReturnPart) and part.content == 'probe complete' for part in parts)
    assert result['status'] == 'failed'
    assert any('provider interrupted' in error for error in result['errors'])
    assert result['usage']['input_tokens'] == 7
    assert result['usage']['output_tokens'] == 3
    assert result['usage_complete'] is False
    assert result['total_tokens'] is result['estimated_cost_usd'] is None
    assert read_json(folder / 'run.json') == result
