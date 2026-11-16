"""Controller checks plus opt-in native-framework/OCI smoke; never call a model."""
import asyncio
import os

import pytest

from core import reviewed
from run import WRAPPER, plan, validate_config


def configuration():
    return dict(model='openai-responses:explicit-test-model',
                images={'statadic':'sha256:'+'a'*64, 'statsmodels':'sha256:'+'b'*64},
                tasks=['PD0007','PD0119'], efforts=['low','medium'], repetitions=2,
                timeout_seconds=60, request_limit=5, token_limit=1000, seed=41,
                pricing=None)


def test_plan_is_balanced_and_reproducible():
    config=validate_config(configuration())
    jobs=plan(config)
    assert len(jobs)==16 and jobs==plan(config)
    assert len({j['id'] for j in jobs})==16
    assert {j['task'] for j in jobs}==set(config['tasks'])
    assert all(sum(j['library']==library for j in jobs)==8 for library in config['images'])


@pytest.mark.parametrize('change',[
    {'model':''}, {'images':{'statadic':'latest','statsmodels':'latest'}},
    {'tasks':['PD0007','PD0007']}, {'repetitions':True}, {'token_limit':0},
    {'efforts':['medium','medium']}, {'pricing':{'as_of':'today','source':'test','input':-1,'cached_input':0,'output':0}},
])
def test_bad_config_fails_before_running(change):
    config=configuration();config.update(change)
    with pytest.raises(ValueError):
        validate_config(config)


def test_empty_review_identity_is_rejected():
    review=dict(submission_sha256=None, reviewer='r', notes='checked', interpretation=0,
                package_policy=0, integrity_violation=False)
    with pytest.raises(ValueError):
        reviewed(dict(automatic_grade=0),review,None)


@pytest.mark.integration
def test_native_pydantic_case_lifecycle_and_capability_api():
    pytest.importorskip('pydantic_evals')
    pytest.importorskip('pydantic_ai_harness')
    import sys
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator
    from pydantic_evals.lifecycle import CaseLifecycle
    from pydantic_ai_harness.coder import Coder
    from pydantic_ai_harness.docker_sandbox import DockerSandboxBackend
    from pydantic_ai.usage import RunUsage, UsageLimits

    hooks=[]
    class Lifecycle(CaseLifecycle):
        async def setup(self):
            hooks.append('setup')
        async def prepare_context(self,ctx):
            ctx.metrics['tokens']=7
            hooks.append('context')
            return ctx
        async def teardown(self,result):
            hooks.append('teardown')
    class Check(Evaluator):
        def evaluate(self,ctx):
            assert ctx.output==ctx.expected_output
            return {'grade':85.0}
    report=Dataset(cases=[Case(name='local',inputs=4,expected_output=8)],evaluators=[Check()]).evaluate_sync(
        lambda value:value*2,lifecycle=Lifecycle,max_concurrency=1)
    assert hooks==['setup','context','teardown']
    assert report.cases[0].metrics['tokens']==7
    assert Coder(repo_context=False,sub_agents=False) is not None
    assert DockerSandboxBackend(image='not-started',executable=sys.executable,allow_internet_access=False).ref is None
    usage=RunUsage()
    assert usage.input_tokens==usage.output_tokens==usage.cache_read_tokens==0
    assert UsageLimits(request_limit=2,total_tokens_limit=1000) is not None


@pytest.mark.integration
@pytest.mark.parametrize('library',['statadic','statsmodels'])
def test_native_image_smoke(tmp_path,library):
    image=os.environ.get(f'AGENTBENCH_{library.upper()}_IMAGE')
    if not image:
        pytest.skip('Set both immutable treatment image IDs to enable OCI smoke checks')
    pytest.importorskip('pydantic_ai_harness')
    from run import sandbox
    public=tmp_path/'public';public.mkdir()
    (public/'data.csv').write_text('x,y\n0,1\n1,2\n')
    other='statsmodels' if library=='statadic' else 'statadic'
    async def smoke():
        async with sandbox(image,public,tmp_path/'work',os.environ.get('AGENTBENCH_RUNTIME','docker')) as (workspace,_):
            result=await workspace.run(['python','-c',
                f'import {library}; import importlib.util,os,socket; '
                f'assert importlib.util.find_spec("{other}") is None; '
                'assert "OPENAI_API_KEY" not in os.environ; '
                'assert not os.path.exists("/input/expected.json"); '
                'assert socket.if_nameindex()==[(1,"lo")]'],timeout=60)
            assert result.exit_code==0, result.stderr
            await workspace.write_text('submission/report.tex','A reproducible results fragment.')
            await workspace.write_text('wrapper.tex',WRAPPER)
            result=await workspace.run(['tectonic','--untrusted','--only-cached','wrapper.tex'],timeout=60)
            assert result.exit_code==0,result.stderr
    asyncio.run(smoke())


def test_sandbox_cleanup_even_when_task_fails(tmp_path,monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from run import sandbox
    cleaned=[]
    captured={}
    class Backend:
        def __init__(self,**kwargs):
            captured.update(kwargs)
            self.ref='container-id'
    class Workspace:
        def __init__(self,backend):
            pass
        async def run(self,*args,**kwargs):
            return SimpleNamespace(exit_code=0)
    class Sandbox:
        def __init__(self,**kwargs):
            pass
        async def destroy(self,ref):
            cleaned.append(ref)
    workspaces=ModuleType('pydantic_ai.workspaces');workspaces.Workspace=Workspace
    docker=ModuleType('pydantic_ai_harness.docker_sandbox')
    docker.DockerSandbox=Sandbox;docker.DockerSandboxBackend=Backend
    monkeypatch.setitem(sys.modules,'pydantic_ai.workspaces',workspaces)
    monkeypatch.setitem(sys.modules,'pydantic_ai_harness.docker_sandbox',docker)
    async def exercise():
        async with sandbox('sha256:'+'a'*64,tmp_path,tmp_path/'work','docker'):
            raise RuntimeError('failure after setup')
    with pytest.raises(RuntimeError,match='failure after setup'):
        asyncio.run(exercise())
    assert cleaned==['container-id']
    assert captured['allow_internet_access'] is False
    assert 'OPENAI_API_KEY' not in captured['env']
    assert '--read-only' in captured['docker_args']
