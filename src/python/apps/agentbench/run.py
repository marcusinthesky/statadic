"""Pydantic Evals + Coder; isolation is delegated to DockerSandboxBackend."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import itertools
import json
import random
import re
import shutil
import subprocess
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from core import HERE, SOURCE, catalogue, grade, prepare_csv, read_json, read_regular, save, schema, sha, witness

WRAPPER = r'''\documentclass{article}
\usepackage{amsmath,booktabs,siunitx,graphicx,pgfplots,hyperref}
\pgfplotsset{compat=1.18}
\begin{document}
\input{submission/report.tex}
\end{document}
'''


def fingerprint():
    return {p.name: sha(p.read_bytes()) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.md','.toml')}


def validate_config(config):
    if not isinstance(config.get('model'), str) or not config['model'].strip():
        raise ValueError('Choose an explicit Pydantic model identifier; no paid default')
    if ('images' in config) == ('conditions' in config):
        raise ValueError('Choose either legacy images or documentation conditions, not both')
    if 'conditions' in config:
        if not isinstance(config['conditions'],dict) or set(config['conditions']) != {'minimal','reference'}:
            raise ValueError('Documentation conditions must be minimal and reference')
        image_sets = config['conditions'].values()
    else:
        image_sets = [config['images']]
    for images in image_sets:
        if not isinstance(images,dict) or set(images) != {'statadic','statsmodels'}:
            raise ValueError('Each treatment compares statadic and statsmodels on identical tasks')
        if any(not isinstance(v,str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',v) for v in images.values()):
            raise ValueError('Use local immutable image IDs from docker/podman image inspect, not tags')
    selected = list(catalogue()) if config['tasks']=='all' else config['tasks']
    if not selected or len(set(selected)) != len(selected) or set(selected)-catalogue().keys():
        raise ValueError('Choose distinct task IDs from the catalogue')
    if not config['efforts'] or len(set(config['efforts'])) != len(config['efforts']) or set(config['efforts'])-{'low','medium','high'}:
        raise ValueError('Choose distinct low/medium/high reasoning efforts')
    for key in ('repetitions','timeout_seconds','request_limit','token_limit'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['seed']) is not int or config.get('executable','docker') not in ('docker','podman'):
        raise ValueError('Invalid seed or container runtime')
    pricing = config.get('pricing')
    if pricing is not None:
        if not pricing.get('as_of') or not pricing.get('source'):
            raise ValueError('Pricing needs as_of and source')
        for k in ('input','cached_input','output'):
            v = pricing[k]
            if type(v) not in (int,float) or not 0 <= v < float('inf'):
                raise ValueError('Pricing must contain finite nonnegative USD/million rates')
    return dict(config, tasks=selected)


def prepare(config, destination):
    """Explicit network boundary; preserve source identity and adapted contracts."""
    config = validate_config(config)
    lock = HERE/'uv.lock'
    if not lock.exists():
        raise ValueError('Run explicit uv lock/setup first; a controller lock is required')
    versions = {p:importlib.metadata.version(p) for p in ('pydantic-ai-slim','pydantic-ai-harness','pydantic-evals','numpy','scipy')}
    repository_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    destination.mkdir(parents=True)  # Never overwrite a frozen campaign.
    for task_id in config['tasks']:
        task = catalogue()[task_id]
        folder = destination / 'tasks' / task_id
        public = folder / 'public'
        public.mkdir(parents=True)
        url = f"{SOURCE}/TaskDatasets/{task['dataset']}/{task['file']}"
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read(20_000_001)
        if len(raw)>20_000_000:
            raise ValueError('Oversized dataset')
        (folder/'source.csv').write_bytes(raw)
        rows = prepare_csv(task, raw.decode('utf-8-sig'), public/'data.csv')
        expected = witness(task, rows)
        save(folder/'expected.json', expected)
        save(public/'task.json', dict(task, parameter_names=expected['parameter_names'], row_ids=expected['row_ids']))
        save(public/'schema.json', schema(expected))
        (public/'contract.md').write_text((HERE/'prompt.md').read_text())
        save(folder/'source.json', dict(url=url, sha256=sha(raw)))
    files = {str(p.relative_to(destination)):sha(p.read_bytes()) for p in sorted((destination/'tasks').rglob('*')) if p.is_file()}
    shutil.copyfile(lock, destination/'controller.uv.lock')
    save(destination/'manifest.json', dict(config=config, harness=fingerprint(), files=files, versions=versions,
                                         controller_lock=sha(lock.read_bytes()),repository_commit=repository_commit))
    print(f'Prepared {len(config["tasks"])} tasks; no model calls made.')


def checked(campaign):
    manifest = read_json(campaign/'manifest.json')
    if fingerprint()!=manifest['harness'] or sha((HERE/'uv.lock').read_bytes())!=manifest['controller_lock']:
        raise ValueError('Harness/lock changed; create a new campaign')
    for path, expected in manifest['files'].items():
        if sha(read_regular(campaign/path))!=expected:
            raise ValueError(f'Frozen input changed: {path}')
    for package, version in manifest['versions'].items():
        if importlib.metadata.version(package)!=version:
            raise ValueError(f'Controller version changed: {package}')
    return manifest


@asynccontextmanager
async def sandbox(image, public, work, executable):
    from pydantic_ai.workspaces import Workspace
    from pydantic_ai_harness.docker_sandbox import DockerSandbox, DockerSandboxBackend
    work.mkdir(parents=True, exist_ok=True)
    work.chmod(0o777)
    backend = DockerSandboxBackend(image=image, executable=executable, working_dir='/work',
        allow_internet_access=False, env={'HOME':'/tmp','MPLBACKEND':'Agg','JAX_ENABLE_X64':'true','JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
        docker_args=['--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--user=65532:65532',
                     '--memory=4g','--cpus=2','--pids-limit=256','--tmpfs=/tmp:rw,size=512m,mode=1777',
                     '--mount',f'type=bind,src={public.resolve()},dst=/input,readonly',
                     '--mount',f'type=bind,src={work.resolve()},dst=/work'])
    try:
        workspace = Workspace(backend)
        await workspace.run(['mkdir','-p','submission'], timeout=15)
        yield workspace, backend
    finally:
        if backend.ref is not None:
            await asyncio.shield(DockerSandbox(executable=executable).destroy(backend.ref))


async def replay(image, public, code, folder, executable):
    async with sandbox(image,public,folder,executable) as (workspace,_):
        await workspace.write_text('submission/solution.py', code)
        result = await workspace.run(['python','submission/solution.py'], timeout=120)
    # The untrusted process is gone before the controller reads artifacts.
    (folder/'replay.log').write_text(result.stdout+'\n'+result.stderr)
    if result.exit_code != 0:
        raise ValueError('Clean replay failed; see replay.log')
    actual = read_json(folder/'submission/results.json')
    from jsonschema import validate
    validate(actual, read_json(public/'schema.json'))
    return actual


async def compile_report(image, public, replay_dir, destination, executable):
    # Compile in another fresh container. Never reuse a PDF or executable supplied
    # by solution.py. Copy only TeX and image artifacts, with bounded regular reads.
    destination.mkdir()
    submission = destination/'submission'
    submission.mkdir()
    paths = list((replay_dir/'submission').rglob('*'))
    if len(paths)>100:
        raise ValueError('Too many report artifacts')
    for source in paths:
        if source.suffix in ('.tex','.png','.pdf','.jpg','.bib'):
            data = read_regular(source, 10_000_000)
            target = submission/source.relative_to(replay_dir/'submission')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
    async with sandbox(image,public,destination,executable) as (workspace,_):
        await workspace.write_text('wrapper.tex', WRAPPER)
        await workspace.run(['mkdir','-p','build'],timeout=10)
        result = await workspace.run(['tectonic','--untrusted','--only-cached','--outdir','build','wrapper.tex'],timeout=60)
    (destination/'compile.log').write_text(result.stdout+'\n'+result.stderr)
    return result.exit_code==0 and len(read_regular(destination/'build/wrapper.pdf'))>0


async def attempt(job, campaign, config, model):
    from pydantic import TypeAdapter
    from pydantic_ai import Agent, capture_run_messages
    from pydantic_ai.capabilities import Hooks
    from pydantic_ai.messages import ModelMessagesTypeAdapter
    from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits
    from pydantic_ai_harness.coder import Coder
    folder = campaign/'runs'/job['id']
    folder.mkdir(parents=True)
    public = campaign/'tasks'/job['task']/'public'
    images = config['conditions'][job['condition']] if 'condition' in job else config['images']
    image = images[job['library']]
    executable = config.get('executable','docker')
    usage = RunUsage()
    output = dict(job, image=image, model=config['model'], status='started', actual=None,
                  replay_ok=False, latex_ok=False, errors=[], usage_complete=False,
                  usage_incomplete_reasons=['attempt_not_finished'],
                  submission_sha256=sha((job['id']+':no-submission').encode()))
    save(folder/'run.json',output)
    start = time.monotonic()
    requests = {}
    hooks = Hooks()
    ledger = folder/'requests.jsonl'
    ledger.touch()

    def record(event, ctx, request_context, **metadata):
        row = dict(event=event, step=ctx.run_step, attempt=request_context.attempt,
                   elapsed_seconds=time.monotonic()-start, **metadata)
        with ledger.open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')

    @hooks.on.prepare_model_request
    async def prepared(ctx, request_context):
        # Preparation precedes dispatch; an unreturned attempt has unknown usage.
        requests[ctx.run_step,request_context.attempt] = False
        record('prepared',ctx,request_context,model=request_context.model.model_name)
        return request_context

    @hooks.on.after_model_request
    async def returned(ctx, *, request_context, response):
        requests[ctx.run_step,request_context.attempt] = True
        record('returned',ctx,request_context,model=response.model_name,
               provider=response.provider_name,provider_response_id=response.provider_response_id,
               state=response.state,usage=TypeAdapter(RequestUsage).dump_python(response.usage,mode='json'))
        return response

    @hooks.on.model_request
    async def observe(ctx, *, request_context, handler):
        record('step_started',ctx,request_context)
        try:
            response = await handler(request_context)
        except BaseException as exc:
            record('step_error',ctx,request_context,error_type=type(exc).__name__)
            raise
        record('step_returned',ctx,request_context)
        return response

    with capture_run_messages() as messages:
        try:
            async with sandbox(image,public,folder/'work',executable) as (_,backend):
                agent = Agent(model, capabilities=[hooks,Coder(repo_context=False,sub_agents=False,
                                                               unrestricted_filesystem=True)],
                              model_settings={'openai_reasoning_effort':job['effort']})
                prompt = f"Use {job['library']} for {job['task']}. Read /input/task.json, /input/contract.md and /input/schema.json. Write the deliverables under /work/submission."
                if 'condition' in job:
                    prompt += (' Before implementing the analysis, read /docs/INDEX.md for the assigned '
                               'statistical library and consult its relevant API examples. Perform fitting '
                               'and inference with that library.')
                (folder/'prompt.txt').write_text(prompt)
                async with asyncio.timeout(config['timeout_seconds']):
                    await agent.run(prompt, workspace=backend, usage=usage,
                        usage_limits=UsageLimits(request_limit=config['request_limit'],total_tokens_limit=config['token_limit']))
            output['status'] = 'completed'
        except Exception as exc:  # Record failed attempts, never silently drop them.
            output.update(status='failed',errors=[f'{type(exc).__name__}: {exc}'])
        finally:
            (folder/'messages.json').write_bytes(ModelMessagesTypeAdapter.dump_json(messages))
    output['elapsed_seconds'] = time.monotonic()-start
    output['request_accounting'] = dict(scope='pydantic_model_request_attempt',
        prepared_attempts=len(requests),returned_responses=sum(requests.values()),
        unresolved_attempts=sum(not returned for returned in requests.values()),
        unmeasured_responses=usage.unmeasured_requests)
    output['usage_incomplete_reasons'] = []
    if not all(requests.values()):
        output['usage_incomplete_reasons'].append('unresolved_prepared_attempts')
    if usage.unmeasured_requests:
        output['usage_incomplete_reasons'].append('responses_missing_usage')
    if usage.requests != sum(requests.values()):
        output['usage_incomplete_reasons'].append('native_response_count_mismatch')
    output['usage_complete'] = not output['usage_incomplete_reasons']
    output['usage'] = TypeAdapter(RunUsage).dump_python(usage, mode='json')
    output['observed_total_tokens'] = usage.input_tokens+usage.output_tokens
    output['total_tokens'] = usage.input_tokens+usage.output_tokens if output['usage_complete'] else None
    output['estimated_cost_usd'] = None
    rates = config.get('pricing')
    if output['usage_complete'] and rates:
        cached = usage.cache_read_tokens
        output['estimated_cost_usd'] = ((usage.input_tokens-cached)*rates['input']+cached*rates['cached_input']+usage.output_tokens*rates['output'])/1e6
    try:
        code = read_regular(folder/'work/submission/solution.py', 1_000_000)
        output['submission_sha256'] = sha(code)
        first = await replay(image,public,code.decode(),folder/'replay-1',executable)
        second = await replay(image,public,code.decode(),folder/'replay-2',executable)
        from core import same
        output['actual'] = first
        output['replay_ok'] = same(first,second,rtol=1e-10,atol=1e-12)
        output['latex_ok'] = await compile_report(image,public,folder/'replay-1',folder/'tex',executable)
    except Exception as exc:
        output['errors'].append(f'{type(exc).__name__}: {exc}')
    save(folder/'run.json', output)
    return output


def plan(config):
    if 'conditions' in config:
        jobs = [dict(task=t,library=l,effort=e,repetition=r,condition=c)
            for t,c,l,e,r in itertools.product(config['tasks'],sorted(config['conditions']),
                ('statadic','statsmodels'),config['efforts'],range(config['repetitions']))]
    else:
        jobs = [dict(task=t,library=l,effort=e,repetition=r) for t,l,e,r in itertools.product(config['tasks'],sorted(config['images']),config['efforts'],range(config['repetitions']))]
    random.Random(config['seed']).shuffle(jobs)
    return [dict(job,id=f'run-{i:05d}') for i,job in enumerate(jobs)]


async def evaluate(campaign, execute):
    from pydantic_ai.models import infer_model
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator
    from pydantic_evals.lifecycle import CaseLifecycle
    manifest = checked(campaign)
    config = validate_config(manifest['config'])
    jobs = plan(config)
    if not execute:
        print(json.dumps(jobs,indent=2))
        print('Plan only; --execute permits model charges.')
        return
    if (campaign/'runs').exists():
        raise ValueError('Runs already exist; retain this campaign and prepare a new one')
    # Check every image is already local; sandbox creation must not pull a tag.
    image_sets = config['conditions'].values() if 'conditions' in config else [config['images']]
    for image in sorted({image for images in image_sets for image in images.values()}):
        subprocess.run([config.get('executable','docker'),'image','inspect',image],check=True,stdout=subprocess.DEVNULL)
    class Rubric(Evaluator):
        def evaluate(self, ctx):
            out = ctx.output
            scored = grade(ctx.expected_output,out['actual'],out['replay_ok'],out['latex_ok'])
            save(campaign/'runs'/out['id']/'grade.json',scored)
            return {'automatic_grade':scored['automatic_grade']}
    class Metrics(CaseLifecycle):
        async def prepare_context(self, ctx):
            for key in ('elapsed_seconds','total_tokens','estimated_cost_usd'):
                if ctx.output[key] is not None:
                    ctx.metrics[key] = ctx.output[key]
            return ctx
    dataset = Dataset(name='statllm-research-workflow',cases=[Case(name=j['id'],inputs=j,
        expected_output=read_json(campaign/'tasks'/j['task']/'expected.json'),
        metadata={key:j[key] for key in ('task','library','effort','condition') if key in j}) for j in jobs],evaluators=[Rubric()])
    # Keep subscription token refreshes in one provider without writing CLI credentials.
    async with infer_model(config['model']) as model:
        async def task(job):
            return await attempt(job,campaign,config,model)
        report = await dataset.evaluate(task,max_concurrency=1,lifecycle=Metrics)
    report.print(include_input=False,include_output=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    p = sub.add_parser('prepare')
    p.add_argument('config',type=Path)
    p.add_argument('campaign',type=Path)
    p.add_argument('--acknowledge-source-terms',action='store_true')
    p = sub.add_parser('run')
    p.add_argument('campaign',type=Path)
    p.add_argument('--execute',action='store_true')
    p = sub.add_parser('review')
    p.add_argument('run',type=Path)
    p.add_argument('review',type=Path)
    p = sub.add_parser('summarize')
    p.add_argument('campaign',type=Path)
    p.add_argument('--automatic',action='store_true')
    args = parser.parse_args()
    if args.command=='list':
        for t in catalogue().values():
            print(t['id'],t['title'])
    elif args.command=='prepare':
        if not args.acknowledge_source_terms:
            parser.error('Review the upstream dataset terms before --acknowledge-source-terms')
        prepare(read_json(args.config),args.campaign.resolve())
    elif args.command=='run':
        asyncio.run(evaluate(args.campaign.resolve(),args.execute))
    elif args.command=='review':
        from core import reviewed
        current = read_json(args.run/'grade.json')
        if current['review'] is not None:
            raise ValueError('Existing review: preserve it rather than overwrite')
        result = reviewed(current,read_json(args.review),read_json(args.run/'run.json').get('submission_sha256'))
        save(args.run/'grade.json',result)
    else:
        from reporting import summarize
        summarize(args.campaign,args.automatic)


if __name__=='__main__':
    main()
