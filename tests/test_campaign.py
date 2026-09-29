"""The unattended cluster campaign (scripts/cluster_campaign.sh + campaign_status.py + build_cluster_job.py)
and run_pipeline.sh's RUN_ROOT guard. The campaign runs against a stub run_pipeline.sh that records every call
and fakes the outputs the campaign reads, so phase order, probes, resume and catch-up are checked without a
GPU. Needs a POSIX bash (Git Bash on Windows)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASH = shutil.which('bash')
pytestmark = pytest.mark.skipif(not BASH or 'system32' in BASH.lower(), reason="needs a POSIX bash")

STUB = r'''#!/usr/bin/env bash
# test stub of run_pipeline.sh: one line per call, plus the files the campaign reads
root=$RUN_ROOT
[[ "${SMOKE:-0}" == 1 ]] && root="$(dirname "$RUN_ROOT")/smoke"
root="$root${DISTRACTORS:+_distractors-$DISTRACTORS}${MULTIHOP_PAD_CHARS:+_pad-$MULTIHOP_PAD_CHARS}"
echo "root=$(basename "$root") stages=[$STAGES] arms=[$EXTRA_ARMS] n=${N_TRAIN:-}/${N_DEV:-}/${N_TEST:-} seeds=${EXTRA_SEEDS:-}" >> "$STUB_CALLS"
ev="$root/results/eval_test"; mkdir -p "$ev"
if [[ " $STAGES " == *" select "* ]]; then
  for arm in ${EXTRA_ARMS//,/ }; do
    label=${arm%%=*}; label=${label%%:*}
    [[ ",${STUB_BROKEN:-}," == *",$label,"* ]] && { echo "!! $label: model does not load"; exit 1; }
    echo "{\"arm\": \"$label\", \"doc_id\": \"d1\"}" >> "$ev/selections_shard0.jsonl"
  done
fi
if [[ -n "${STUB_FAIL_STAGE:-}" && " $STAGES " == *" $STUB_FAIL_STAGE "* ]]; then echo "!! stub failure"; exit 1; fi
if [[ " $STAGES " == *" report "* ]]; then
  echo '{"cells": [], "hypotheses": []}' > "$ev/report.json"; echo "# report" > "$ev/report.md"
fi
exit 0
'''


def posix(path) -> str:
    return str(path).replace('\\', '/')


# what a surrounding run exports (these tests also run in run_pipeline.sh's preflight, inside the campaign,
# e.g. with DISTRACTORS=hard or baked MAIN_SIZES): every test sets its own
INHERITED = {'BASE', 'CAMPAIGN', 'PHASES', 'PILOT_SIZES', 'MAIN_SIZES', 'CANDIDATE_ARMS', 'ABL_PAD_CHARS',
             'ABL_EXTRA_SEEDS', 'PHASE_ATTEMPTS', 'HEARTBEAT_MIN', 'STATUS_REPO', 'UPLOAD_LABELS', 'PYLIBS',
             'BASELINE_SITE', 'LLMLINGUA_SITE', 'NUM_GPUS', 'RUN_ROOT', 'LABELS', 'FIT', 'MODELS', 'EVAL_DIR', 'LOGS',
             'SMOKE', 'SMOKE_KEEP', 'DISTRACTORS', 'MULTIHOP_PAD_CHARS', 'STAGES', 'EXTRA_ARMS', 'N_TRAIN', 'N_DEV',
             'N_TEST', 'ORACLE_N', 'ENV_FILE', 'TTC_UNPACK_ONLY', 'CUDA_VISIBLE_DEVICES'}


def bash_env(**extra) -> dict:
    """The environment minus a surrounding run's settings (and any HF token), this interpreter first on PATH
    (the scripts call `python`)."""
    env = {k: v for k, v in os.environ.items() if k not in INHERITED and not k.startswith(('HF_', 'STUB_'))}
    env.update(extra)
    env['PATH'] = os.path.dirname(sys.executable) + os.pathsep + env.get('PATH', '')
    return env


@pytest.fixture
def campaign(tmp_path):
    code = tmp_path / 'code'
    (code / 'scripts').mkdir(parents=True)
    for rel in ('scripts/cluster_campaign.sh', 'scripts/campaign_status.py', 'scripts/compare_runs.py', '.env.example'):
        shutil.copy(os.path.join(ROOT, rel), code / rel)
    shutil.copytree(os.path.join(ROOT, 'ttcompress'), code / 'ttcompress',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (code / 'run_pipeline.sh').write_bytes(STUB.encode())
    calls = tmp_path / 'calls.txt'

    def run(**extra):
        env = bash_env()
        env.update(BASE=posix(tmp_path / 'base'), CAMPAIGN='t', NUM_GPUS='1', HEARTBEAT_MIN='0', PHASE_ATTEMPTS='1',
                   HF_HOME=posix(tmp_path / 'hf'), STUB_CALLS=posix(calls),
                   CANDIDATE_ARMS='recomp,exit,xprovence=provence:naver/x',
                   PHASES='smoke probe pilot main abl_hard abl_pad compare')
        env.update(extra)
        res = subprocess.run([BASH, posix(code / 'scripts' / 'cluster_campaign.sh')], env=env, capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=600)
        lines = calls.read_text().splitlines() if calls.exists() else []
        calls.unlink(missing_ok=True)
        with open(tmp_path / 'base' / 'campaigns' / 't' / 'state.json', encoding='utf-8') as f:
            return res, lines, json.load(f)
    return run, tmp_path


def test_campaign_runs_every_phase_and_drops_a_broken_compressor(campaign):
    run, tmp_path = campaign
    res, calls, state = run(STUB_BROKEN='exit')
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert calls[0].startswith('root=smoke stages=[preflight prefetch labels fit ensemble train select answer report] arms=[]')
    probes = [c for c in calls if 'stages=[prefetch select]' in c]
    assert [c.split('arms=[')[1].split(']')[0] for c in probes] == ['recomp', 'exit', 'xprovence=provence:naver/x']
    assert {k: a['state'] for k, a in state['arms'].items()} == {'recomp': 'ok', 'exit': 'failed', 'xprovence': 'ok'}
    working = 'arms=[recomp,xprovence=provence:naver/x]'
    assert any('stages=[answer report]' in c and working in c for c in calls)            # smoke_arms
    pilot = [c for c in calls if c.startswith('root=pilot_300-30-50')]
    assert len(pilot) == 3 and all(working in c and 'n=300/30/50' in c for c in pilot)  # three timed stage groups
    main = [c for c in calls if c.startswith('root=main_3000-300-500 ')]
    assert len(main) == 3 and all('n=3000/300/500' in c for c in main)
    hard = [c for c in calls if c.startswith('root=main_3000-300-500_distractors-hard')]
    assert len(hard) == 3 and all('seeds=none' in c for c in hard)
    assert len([c for c in calls if c.startswith('root=main_3000-300-500_pad-20000')]) == 3
    assert set(state['runs']) == {'pilot', 'main', 'abl_hard', 'abl_pad', 'compare'}
    assert {p: v['state'] for p, v in state['phases'].items()} == {
        'smoke': 'done', 'smoke_arms': 'done', 'pilot': 'done', 'main': 'done', 'abl_hard': 'done', 'abl_pad': 'done',
        'compare': 'done'}
    assert 'main_hours' in state['projection']
    status = tmp_path / 'base' / 'campaigns' / 't' / 'status'
    assert (status / 'runs' / 'compare' / 'compare.md').exists() and (status / 'runs' / 'main' / 'report.md').exists()
    assert 'exit | failed' in (status / 'STATUS.md').read_text(encoding='utf-8')

    # re-submission once the broken compressor works: nothing reruns except its probe, the smoke answer/report
    # with the new set, the catch-up of every finished run and the comparison
    res, calls, state = run()
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert not any('preflight' in c for c in calls)
    assert sum('stages=[prefetch select]' in c for c in calls) == 1 and state['arms']['exit']['state'] == 'ok'
    catch_up = [c for c in calls if 'stages=[select answer report]' in c]
    assert len(catch_up) == 4 and all('exit' in c for c in catch_up)
    assert state['runs']['main']['arms'] == 'recomp,exit,xprovence=provence:naver/x'
    assert state['phases']['compare']['state'] == 'done'


def test_campaign_stops_when_the_smoke_test_fails(campaign):
    run, _ = campaign
    res, calls, state = run(STUB_FAIL_STAGE='labels')
    assert res.returncode == 1
    assert len(calls) == 1 and state['phases']['smoke']['state'] == 'failed'
    assert 'smoke test failed' in state['info']['result']


def test_projection_from_the_pilot(tmp_path):
    raw = tmp_path / 'labels' / 'raw' / 'Org--r1'
    for split, n in (('train', 3), ('dev', 1), ('test', 1)):
        d = raw / f'uit_viquad_{split}'
        d.mkdir(parents=True)
        (d / 'measure_config.json').write_text('{}')
        for i in range(n):
            (d / f'doc{i}.json').write_text(json.dumps({'seconds': 1.5, 'seconds_logprob': 0.5}))
    tool = [sys.executable, os.path.join(ROOT, 'scripts', 'campaign_status.py'), '--dir', str(tmp_path / 'c')]
    for group, sec in (('fit', '100'), ('select', '50')):
        subprocess.run(tool + ['timing', 'pilot', group, sec], check=True)
    env = dict(os.environ, LABEL_READERS='Org/r1', TRAIN_SOURCES='uit_viquad', EVAL_SOURCES='uit_viquad', ORACLE_N='5')
    subprocess.run(tool + ['projection', '--code-dir', ROOT, '--labels', str(tmp_path / 'labels'), '--gpus', '1',
                           '--pilot-sizes', '3 1 1', '--main-sizes', '10 2 5'], check=True, env=env, capture_output=True)
    with open(tmp_path / 'c' / 'state.json', encoding='utf-8') as f:
        proj = json.load(f)['projection']
    # labels (7 + 1 + 4 documents x 2 s + 3 model start-ups x 120 s) + train 100 x 10/3 + eval 50 x 5/1 + 900 s
    assert proj['main_hours'] == pytest.approx((24 + 360 + 1000 / 3 + 250 + 900) / 3600, abs=1e-4)
    # hard distractors: new documents, every one measured again: (10 + 2 + 5) x 2 s
    assert proj['abl_hard_hours'] == pytest.approx((34 + 360 + 1000 / 3 * 5 / 9 + 250 + 900) / 3600, abs=1e-4)


def test_probe_result_needs_a_selection(tmp_path):
    (tmp_path / 'selections_shard0.jsonl').write_text('{"arm": "exit", "doc_id": "a"}\n')
    (tmp_path / 'select_failures_shard0.jsonl').write_text('{"arm": "exit", "doc_id": "b", "error": "x"}\n')
    tool = [sys.executable, os.path.join(ROOT, 'scripts', 'campaign_status.py'), '--dir', str(tmp_path / 'c'),
            'probe-result', '--eval-dir', str(tmp_path)]
    ok = subprocess.run(tool + ['--arm', 'exit'], capture_output=True, text=True)
    assert ok.returncode == 0 and '1 selections, 1 documents failed' in ok.stdout
    assert subprocess.run(tool + ['--arm', 'recomp'], capture_output=True).returncode == 1


def test_built_job_unpacks_the_repository_without_secrets(tmp_path):
    job = tmp_path / 'job.sh'
    subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'build_cluster_job.py'), '--out', str(job),
                    '--set', 'CAMPAIGN=x y'], check=True, capture_output=True)
    text = job.read_text(encoding='utf-8')
    assert '\r' not in text and "CAMPAIGN='x y'" in text
    env = bash_env(BASE=posix(tmp_path / 'base'), TTC_UNPACK_ONLY='1')
    subprocess.run([BASH, posix(job)], env=env, check=True, capture_output=True, timeout=120)
    (code,) = (tmp_path / 'base' / 'code').iterdir()
    for rel in ('run_pipeline.sh', 'scripts/cluster_campaign.sh', 'ttcompress/pruner.py', '.env.example', 'BUILD_ID'):
        assert (code / rel).is_file(), rel
    assert not (code / '.env').exists() and not (code / 'paper').exists()


# ---------------------------------------------------------------------------- run_pipeline.sh RUN_ROOT guard

def _pipeline(run_root, n_train, labels):
    env = bash_env(RUN_ROOT=posix(run_root), LABELS=posix(labels), N_TRAIN=str(n_train), N_DEV='20',
               N_TEST='30', ORACLE_N='0', NUM_GPUS='1', STAGES='select', EXTRA_ARMS='')
    return subprocess.run([BASH, posix(os.path.join(ROOT, 'run_pipeline.sh'))], env=env, capture_output=True,
                          text=True, encoding='utf-8', errors='replace', timeout=300)


def test_run_root_refuses_another_configuration(tmp_path):
    first = _pipeline(tmp_path / 'run', 3000, tmp_path / 'labels')
    assert 'belongs to another configuration' not in first.stdout
    assert 'N_TRAIN=3000' in (tmp_path / 'run' / 'run_config.txt').read_text()
    second = _pipeline(tmp_path / 'run', 50, tmp_path / 'labels')
    assert second.returncode != 0 and 'belongs to another configuration' in second.stdout


def test_run_root_refuses_outputs_from_before_the_guard(tmp_path):
    (tmp_path / 'old' / 'models' / 'pruner_beta_primary').mkdir(parents=True)
    res = _pipeline(tmp_path / 'old', 3000, tmp_path / 'labels')
    assert res.returncode != 0 and 'predates run_config.txt' in res.stdout
