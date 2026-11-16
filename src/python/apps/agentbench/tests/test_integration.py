"""Controller checks plus opt-in native-framework/OCI smoke; never call a model."""
import asyncio
import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from core import reviewed, save, schema
from run import plan, validate_config


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
    report=Dataset(name='local-framework-smoke',cases=[Case(name='local',inputs=4,expected_output=8)],evaluators=[Check()]).evaluate_sync(
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
def test_native_image_smoke(tmp_path,library,monkeypatch):
    image=os.environ.get(f'AGENTBENCH_{library.upper()}_IMAGE')
    if not image:
        pytest.skip('Set both immutable treatment image IDs to enable OCI smoke checks')
    from run import compile_report, replay, sandbox
    public=tmp_path/'public';public.mkdir()
    (public/'data.csv').write_text('x,y\n0,1\n1,2\n')
    expected={'sum_y':3.0}
    save(public/'schema.json',schema(expected))
    # Synthetic controller-only credentials and private grading material. Never
    # inspect or copy the developer's real subscription authentication files.
    private=tmp_path/'controller-only';private.mkdir()
    (private/'.codex').mkdir()
    (private/'.codex/auth.json').write_text('{"access_token":"synthetic-subscription-token"}')
    (private/'expected.json').write_text('{"private_oracle":true}')
    sentinels=('OPENAI_API_KEY','OPENAI_ACCESS_TOKEN','CODEX_API_KEY','AGENTBENCH_SECRET_SENTINEL')
    for name in sentinels:
        monkeypatch.setenv(name,'synthetic-controller-secret')
    hidden=[str(private),str(Path(__file__).resolve().parents[5]),
            '/input/expected.json','/env/AGENTS.md',
            '/env/src/python/apps/agentbench/core.py','/env/.codex/auth.json',
            '/tmp/.codex/auth.json','/root/.codex/auth.json',
            '/var/run/docker.sock','/run/docker.sock']
    other='statsmodels' if library=='statadic' else 'statadic'
    isolation=textwrap.dedent(f'''
        import {library}
        import importlib.util, os, socket
        from pathlib import Path
        assert importlib.util.find_spec({other!r}) is None
        assert all(name not in os.environ for name in {sentinels!r})
        assert 'CODEX_HOME' not in os.environ
        assert all(not os.path.exists(path) for path in {hidden!r})
        assert sorted(path.name for path in Path('/input').iterdir()) == ['data.csv','schema.json']
        assert os.geteuid() != 0
        status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
        assert int(status['CapEff'].strip(),16) == 0
        assert status['NoNewPrivs'].strip() == '1'
        mounts={{parts[4]:parts[5].split(',') for line in Path('/proc/self/mountinfo').read_text().splitlines() if (parts:=line.split())}}
        assert 'ro' in mounts['/'] and 'ro' in mounts['/input']
        for target in ('/agentbench-forbidden','/input/agentbench-forbidden'):
            try:
                Path(target).write_text('must fail')
            except OSError:
                pass
            else:
                raise AssertionError('Sandbox allowed a read-only write: '+target)
        assert socket.if_nameindex() == [(1,'lo')]
        try:
            connection=socket.create_connection(('192.0.2.1',443),timeout=0.5)
        except OSError:
            pass
        else:
            connection.close()
            raise AssertionError('Sandbox has outbound network access')
    ''')
    solution=textwrap.dedent(f'''
        import {library}
        import csv, json
        from pathlib import Path
        assert sorted(path.name for path in Path('submission').iterdir()) == ['solution.py']
        marker=Path('/tmp/agentbench-replay-marker')
        assert not marker.exists(), 'Replay reused a sandbox'
        marker.write_text('created in this sandbox only')
        with Path('/input/data.csv').open() as stream:
            total=sum(float(row['y']) for row in csv.DictReader(stream))
        Path('submission/results.json').write_text(json.dumps({{'sum_y':total}}))
        Path('submission/report.tex').write_text('A reproducible results fragment.')
    ''')
    executable=os.environ.get('AGENTBENCH_RUNTIME','docker')
    async def smoke():
        work=tmp_path/'work'
        async with sandbox(image,public,work,executable) as (workspace,backend):
            inspected=subprocess.run([executable,'inspect',backend.ref.id],check=True,
                                     capture_output=True,text=True)
            container=json.loads(inspected.stdout)[0]
            binds={(mount['Source'],mount['Destination'],mount['RW'])
                   for mount in container['Mounts'] if mount['Type']=='bind'}
            assert binds=={(str(public.resolve()),'/input',False),(str(work.resolve()),'/work',True)}
            assert container['HostConfig']['ReadonlyRootfs']
            assert container['HostConfig']['NetworkMode']=='none'
            assert container['HostConfig']['Memory']==4*1024**3
            assert container['HostConfig']['NanoCpus']==2_000_000_000
            assert container['HostConfig']['PidsLimit']==256
            result=await workspace.run(['python','-c',isolation],timeout=60)
            assert result.exit_code==0, result.stderr
        first=await replay(image,public,solution,tmp_path/'replay-1',executable)
        second=await replay(image,public,solution,tmp_path/'replay-2',executable)
        assert first==second==expected
        assert await compile_report(image,public,tmp_path/'replay-1',tmp_path/'tex',executable)
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
