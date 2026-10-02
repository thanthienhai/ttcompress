"""run_pipeline.sh LOO_ARMS=1 (the leave-one-out labeling baseline) as a dry run: a stub `python` first on PATH
records every generate_labels.py / train_pruner.py / evaluate.py call (and fakes the pruner checkpoints) and hands
everything else to this interpreter, so the commands each stage issues are checked without a GPU. Plus
docs/prereg_loo.json through evaluate.py's report. Needs a POSIX bash (Git Bash on Windows)."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

import pytest

from evaluate import parse_arms
from tests.test_campaign import BASH, bash_env, posix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIMARY = 'Org--a'
LOO_ARM_NAMES = {'ours_loo', 'loo_sent', 'ours_k11', 'k11_sent', 'loocomp_bin', 'loocomp_bin_sent'}

STUB_PYTHON = r'''#!/usr/bin/env bash
# dry-run stand-in for `python`: records the pipeline's script calls, runs everything else for real
case "${1:-}" in
  generate_labels.py|train_pruner.py|evaluate.py)
    printf '%s\n' "$*" >> "$DRY_CALLS"
    if [[ "$1" == train_pruner.py ]]; then
      out=""; prev=""
      for a in "$@"; do [[ "$prev" == --out-dir ]] && out=$a; prev=$a; done
      mkdir -p "$out" && echo '{}' > "$out/pruner_config.json" && echo '{}' > "$out/train_log.json"
    fi
    exit 0 ;;
esac
exec "$REAL_PYTHON" "$@"
'''

needs_bash = pytest.mark.skipif(not BASH or 'system32' in BASH.lower(), reason="needs a POSIX bash")


@pytest.fixture
def dry_run(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'python').write_bytes(STUB_PYTHON.encode())
    os.chmod(bin_dir / 'python', 0o755)
    calls = tmp_path / 'calls.txt'

    def run(**extra):
        env = bash_env(RUN_ROOT=posix(tmp_path / 'run'), LABELS=posix(tmp_path / 'labels'), NUM_GPUS='2',
                       LABEL_READERS='Org/a Org/b', EVAL_READERS='Org/a', TRAIN_SOURCES='vimqa hotpotqa',
                       EVAL_SOURCES='vimqa,hotpotqa,2wiki', N_TRAIN='10', N_DEV='4', N_TEST='5', ORACLE_N='0',
                       EXTRA_SEEDS='1 2', EXTRA_ARMS='', FOLLOWUP_ARMS='0', ROUND2_ARMS='0', CPU_JOBS='2',
                       MEASURE_ARGS='', TRAIN_ARGS='', SELECT_ARGS='', RATIOS='4,8',
                       DRY_CALLS=posix(calls), REAL_PYTHON=posix(sys.executable))
        env['PATH'] = str(bin_dir) + os.pathsep + env['PATH']
        env.update(extra)   # set-but-empty keys keep the repo's .env values out (it only fills unset ones)
        res = subprocess.run([BASH, posix(os.path.join(ROOT, 'run_pipeline.sh'))], env=env, capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=600)
        lines = calls.read_text(encoding='utf-8').splitlines() if calls.exists() else []
        calls.unlink(missing_ok=True)
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
        return lines
    return run, tmp_path


def _flag(call: str, name: str):
    args = shlex.split(call)
    return args[args.index(name) + 1] if name in args else None


def _select_arms(calls) -> dict:
    (sel,) = [c for c in calls if c.startswith('evaluate.py select') and '--shard 0 ' in c + ' ']
    return dict(parse_arms(_flag(sel, '--arms')))


@needs_bash
def test_loo_arms_issue_the_loo_label_fit_train_and_select_commands(dry_run):
    run, tmp = dry_run
    calls = run(LOO_ARMS='1', LOO_BIN='1', LOO_BIN_THRESHOLD='0.0', STAGES='labels fit train select')
    labels, fit, models = (posix(tmp / 'labels'), posix(tmp / 'run' / 'labels_fit'), posix(tmp / 'run' / 'models'))

    measure = [c for c in calls if c.startswith('generate_labels.py measure')]
    loo = [c for c in measure if '--mask-scheme loo' in c]
    # primary reader only, train + dev of every training source, sharded over both GPUs, own out-root
    assert len(loo) == 2 * 2 * 2
    assert {_flag(c, '--reader-model') for c in loo} == {'Org/a'}
    assert {(_flag(c, '--source'), _flag(c, '--split'), _flag(c, '--n')) for c in loo} == {
        (s, sp, n) for s in ('vimqa', 'hotpotqa') for sp, n in (('train', '10'), ('dev', '4'))}
    assert {_flag(c, '--out-root') for c in loo} == {f'{labels}/raw_loo'}
    assert {_flag(c, '--shard') for c in loo} == {'0', '1'} and {_flag(c, '--outcomes') for c in loo} == {'f1'}
    # the main measure is unchanged: both readers, the method's masks, into raw
    main = [c for c in measure if '--mask-scheme' not in c]
    assert len(main) == 2 * 2 * 2 * 2 and {_flag(c, '--out-root') for c in main} == {f'{labels}/raw'}

    fits = [c for c in calls if c.startswith('generate_labels.py fit')]
    loo_fit = {_flag(c, '--out-dir'): c for c in fits if '--estimator loo' in c}
    assert set(loo_fit) == {f'{fit}/loo/{PRIMARY}/f1/{s}_{sp}' for s in ('vimqa', 'hotpotqa') for sp in ('train', 'dev')}
    c = loo_fit[f'{fit}/loo/{PRIMARY}/f1/vimqa_train']
    assert _flag(c, '--raw-dir') == f'{labels}/raw_loo/{PRIMARY}/vimqa_train' and _flag(c, '--n') == '10'
    assert '--alpha-from' not in c
    k11 = {_flag(c, '--out-dir'): c for c in fits if '--max-masks' in c}
    assert set(k11) == {f'{fit}/k11/{PRIMARY}/f1/{s}_{sp}' for s in ('vimqa', 'hotpotqa') for sp in ('train', 'dev')}
    c = k11[f'{fit}/k11/{PRIMARY}/f1/hotpotqa_dev']
    assert _flag(c, '--raw-dir') == f'{labels}/raw/{PRIMARY}/hotpotqa_dev'       # the existing random masks
    assert (_flag(c, '--max-masks'), _flag(c, '--mask-seed'), _flag(c, '--n'), _flag(c, '--alpha-n')) == ('11', '0', '4', '4')
    assert _flag(c, '--alpha-from') == f'{labels}/raw/{PRIMARY}/hotpotqa_dev'
    assert len(fits) - len(loo_fit) - len(k11) == 2 * 2 * 2 + 2 * 2               # main fits: f1 x 2 readers + logprob

    train = {_flag(c, '--out-dir').rsplit('/', 1)[1]: c for c in calls if c.startswith('train_pruner.py')}
    for name, source, fit_set in (('pruner_loo', 'beta', 'loo'), ('pruner_k11', 'beta', 'k11'),
                                  ('pruner_loo_bin', 'loo_bin', 'loo')):
        for suffix, seed in (('', None), ('_s1', '1'), ('_s2', '2')):
            c = train[name + suffix]
            assert _flag(c, '--label-source') == source and _flag(c, '--seed') == seed
            assert _flag(c, '--train-labels') == f'{fit}/{fit_set}/{PRIMARY}/f1/vimqa_train,{fit}/{fit_set}/{PRIMARY}/f1/hotpotqa_train'
            assert _flag(c, '--dev-labels') == f'{fit}/{fit_set}/{PRIMARY}/f1/vimqa_dev,{fit}/{fit_set}/{PRIMARY}/f1/hotpotqa_dev'
            assert ('--loo-threshold 0.0' in c) == (source == 'loo_bin')
    assert 'pruner_beta_primary' in train and 'pruner_beta_primary_s1' in train   # the main runs are still there

    arms = _select_arms(calls)
    for para, sent, pruner in (('ours_loo', 'loo_sent', 'pruner_loo'), ('ours_k11', 'k11_sent', 'pruner_k11'),
                               ('loocomp_bin', 'loocomp_bin_sent', 'pruner_loo_bin')):
        for suffix in ('', '_s1', '_s2'):
            assert arms[para + suffix] == f'pruner:{models}/{pruner}{suffix}'
            assert arms[sent + suffix] == f'sent+pruner:{models}/{pruner}{suffix}'

    # every arm the pre-registered LOO families read is declared (ours_sent + seeds come with FOLLOWUP_ARMS=1)
    with open(os.path.join(ROOT, 'docs', 'prereg_loo.json'), encoding='utf-8') as f:
        spec = json.load(f)
    needed = {a for fam in spec['families'] for a in [fam['ours'], *fam['vs']]}
    calls = run(LOO_ARMS='1', FOLLOWUP_ARMS='1', STAGES='select')
    declared = set(_select_arms(calls))
    assert needed <= declared, needed - declared
    assert not any(a.startswith('loocomp_bin') for a in declared)                 # LOO_BIN off


@needs_bash
def test_loo_only_skips_the_main_steps_and_only_arms_reads_the_replication_documents(dry_run):
    run, tmp = dry_run
    calls = run(LOO_ARMS='1', LOO_ONLY='1', EXTRA_SEEDS='none', STAGES='labels fit train')
    assert all('--mask-scheme loo' in c for c in calls if c.startswith('generate_labels.py measure'))
    assert all('--estimator loo' in c or '--max-masks' in c for c in calls if c.startswith('generate_labels.py fit'))
    trained = sorted(_flag(c, '--out-dir').rsplit('/', 1)[1] for c in calls if c.startswith('train_pruner.py'))
    assert trained == ['pruner_k11', 'pruner_loo']

    only = ('full,ours_loo=pruner:@MODELS@/pruner_loo,loo_sent=sent+pruner:@MODELS@/pruner_loo,'
            'ours_k11=pruner:@MODELS@/pruner_k11,k11_sent=sent+pruner:@MODELS@/pruner_k11')
    calls = run(LOO_ARMS='1', EXTRA_SEEDS='none', ONLY_ARMS=only, EVAL_N='2000', EVAL_OFFSET='500',
                EVAL_DIR=posix(tmp / 'run' / 'results' / 'eval_replication'), STAGES='select')
    sel = [c for c in calls if c.startswith('evaluate.py select')]
    assert len(sel) == 2 and all(_flag(c, '--n') == '2000' and _flag(c, '--offset') == '500' for c in sel)
    assert set(_select_arms(calls)) == {'full', 'ours_loo', 'loo_sent', 'ours_k11', 'k11_sent'}
    assert _select_arms(calls)['loo_sent'] == f"sent+pruner:{posix(tmp / 'run' / 'models')}/pruner_loo"


def test_loo_only_needs_loo_arms():
    if not BASH or 'system32' in BASH.lower():
        pytest.skip("needs a POSIX bash")
    res = subprocess.run([BASH, posix(os.path.join(ROOT, 'run_pipeline.sh'))], capture_output=True, text=True,
                         encoding='utf-8', errors='replace', timeout=120,
                         env=bash_env(LOO_ONLY='1', LOO_ARMS='0', STAGES='report', RUN_ROOT='unused'))
    assert res.returncode != 0 and 'LOO_ONLY=1 needs LOO_ARMS=1' in res.stdout


def _row(arm, ratio, doc, f1, source):
    return {'doc_id': f'{source}-{doc}', 'arm': arm, 'ratio': ratio, 'source': source, 'language': 'en',
            'hop': 'multi', 'cluster_id': f'{source}-c{doc // 2}', 'needle_relpos': None, 'num_chunks': 10,
            'full_tokens': 800, 'kept_tokens': 200, 'budget': 200, 'gold_recall': 1.0, 'seconds': 0.01,
            'reader': 'Qwen--Qwen3-8B', 'answer': '', 'em': 0.0, 'f1': f1, 'answer_recall': f1, 'n_gold': 2}


def test_prereg_loo_file_reads_with_seed_checks_and_a_two_sided_split(tmp_path):
    """docs/prereg_loo.json as committed, through evaluate.py report: L1 / L1p / L2a / L2b; the seed reruns of
    ours_k11 / ours_loo are recognized; L2's two directions cannot both hold and use alpha 0.025."""
    rows = []
    for source in ('hotpotqa', '2wiki'):
        for i in range(60):
            e = 0.01 * (i % 5)
            for ratio in (4.0, 8.0):
                for arm, f1 in (('ours_beta', 0.70), ('ours_beta_s1', 0.70), ('ours_loo', 0.60),
                                ('ours_loo_s1', 0.61), ('ours_k11', 0.68), ('ours_k11_s1', 0.67), ('ours_k11_s2', 0.69),
                                ('ours_sent', 0.75), ('ours_sent_s1', 0.74), ('loo_sent', 0.65)):
                    rows.append(_row(arm, ratio, i, f1 + e, source))
    with open(tmp_path / 'answers_Qwen--Qwen3-8B_shard0.jsonl', 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)
    res = subprocess.run([sys.executable, 'evaluate.py', 'report', '--out-dir', str(tmp_path), '--ours', 'ours_sent',
                          '--n-boot', '200', '--prereg', 'docs/prereg_loo.json', '--no-diagnostics'],
                         capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    report = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    fam = {f['name']: f for f in report['prereg']['families']}
    assert list(fam) == ['L1', 'L1p', 'L2a', 'L2b']
    assert fam['L1']['n_tests'] == 2 and fam['L1']['n_supported'] == 2               # 8x only; vimqa absent
    assert fam['L1']['n_seed_robust'] == 2                                            # ours_sent_s1 agrees
    assert fam['L1p']['n_tests'] == 4 and fam['L1p']['n_supported'] == 4
    assert fam['L2a']['n_supported'] == 4 and fam['L2b']['n_supported'] == 0
    assert all(set(t['seed_diffs']) == {'ours_k11_s1', 'ours_k11_s2'} for t in fam['L2a']['tests'])
    assert all(set(t['seed_diffs']) == {'ours_loo_s1'} for t in fam['L2b']['tests'])
    with open(os.path.join(ROOT, 'docs', 'prereg_loo.json'), encoding='utf-8') as f:
        spec = json.load(f)
    assert {f['name']: f.get('alpha') for f in spec['families']} == {'L1': None, 'L1p': None, 'L2a': 0.025, 'L2b': 0.025}
    assert all(f['reader'] == 'Qwen--Qwen3-8B' and f['sources'] == ['vimqa', 'hotpotqa', '2wiki']
               for f in spec['families'])
