#!/usr/bin/env python3
"""Bookkeeping for scripts/cluster_campaign.sh: the campaign's state (phases, probes of the published
compressors, stage timings, finished runs), STATUS.md, the projection of the full run from the pilot, and
the upload of all of it to a private HuggingFace dataset repo -- the only window on the campaign for a user
who cannot log into the cluster.

    campaign_status.py --dir <campaign dir> phase pilot running --note "attempt 1/2" --log <log>
    campaign_status.py --dir ... publish [--heartbeat]    # STATUS.md + reports -> <dir>/status -> HF
    campaign_status.py --dir ... probe-result --eval-dir <smoke eval dir> --arm exit

Standard library only, except huggingface_hub for the upload (imported lazily; a failed upload is a
warning, never an error: the campaign must not stop because the Hub is unreachable).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys

TAIL_LINES = 200
REPORT_FILES = ('report.md', 'report.json')
LARGE_READER_PATTERN = r'(3[0-9]|7[0-9])B'   # run_pipeline.sh's default
MODEL_STARTUP_S = 120                          # per label job: vLLM start-up, not in the per-document seconds
STAGE_OVERHEAD_S = 900                         # preflight + prefetch of one run
SINGLE_HOP = {'uit_viquad', 'xquad_vi'}        # fallback when ttcompress.sources cannot be imported


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_time(s):
    return dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc) if s else None


@contextlib.contextmanager
def locked(campaign_dir: str):
    """The main loop and the heartbeat both write: serialize them (no-op where fcntl is missing)."""
    os.makedirs(campaign_dir, exist_ok=True)
    with open(os.path.join(campaign_dir, '.status.lock'), 'w') as f:
        try:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
        except ImportError:
            pass
        yield


def load(campaign_dir: str) -> dict:
    path = os.path.join(campaign_dir, 'state.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {'campaign': os.path.basename(campaign_dir.rstrip('/\\')), 'info': {}, 'order': [], 'phases': {},
            'arms': {}, 'timings': {}, 'runs': {}, 'current': {}, 'projection': {}}


def save(campaign_dir: str, state: dict):
    path = os.path.join(campaign_dir, 'state.json')
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(path + '.tmp', path)


def tail(path, n: int) -> str:
    if not path or not os.path.isfile(path):
        return ''
    with open(path, encoding='utf-8', errors='replace') as f:
        return ''.join(f.readlines()[-n:])


def newest_file(root):
    files = [p for p in glob.glob(os.path.join(root, '**', '*'), recursive=True) if os.path.isfile(p)] if root else []
    return max(files, key=os.path.getmtime) if files else None


def cell(s) -> str:
    return str(s or '').replace('|', '\\|').replace('\n', ' ')[:300]


# ---------------------------------------------------------------------------- state updates

def cmd_phase(args, state):
    if args.name not in state['order']:
        state['order'].append(args.name)
    p = state['phases'].setdefault(args.name, {})
    if args.state == 'running' and p.get('state') != 'running':
        p['started'], p['ended'] = now(), None
    if args.state in ('done', 'failed', 'interrupted'):
        p['ended'] = now()
    p['state'] = args.state
    p['note'] = args.note
    if args.log:
        p['log'] = args.log


def cmd_arm(args, state):
    a = state['arms'].setdefault(args.label, {})
    a.update({'state': args.state, 'note': args.note, 'updated': now()})
    if args.spec:
        a['spec'] = args.spec


def cmd_timing(args, state):
    state['timings'].setdefault(args.run, {})[args.group] = args.seconds


def cmd_run(args, state):
    state['runs'][args.name] = {'path': args.path, 'arms': args.arms, 'updated': now()}


def cmd_current(args, state):
    state['current'] = {'phase': args.phase, 'log': args.log, 'run_root': args.run_root, 'since': now()}


def cmd_info(args, state):
    state['info'][args.key] = args.value


# ---------------------------------------------------------------------------- probe of a published compressor

def cmd_probe_result(args):
    """Did the arm produce selections on the smoke run? A tolerant arm can 'succeed' with every document
    failing (select_failures_shard*.jsonl), which is a broken arm, not a working one."""
    rows, failed = 0, set()
    for path in glob.glob(os.path.join(args.eval_dir, 'selections_shard*.jsonl')):
        with open(path, encoding='utf-8') as f:
            rows += sum(1 for line in f if line.strip() and json.loads(line).get('arm') == args.arm)
    for path in glob.glob(os.path.join(args.eval_dir, 'select_failures_shard*.jsonl')):
        with open(path, encoding='utf-8') as f:
            failed |= {r['doc_id'] for r in map(json.loads, filter(str.strip, f)) if r.get('arm') == args.arm}
    print(f"{rows} selections, {len(failed)} documents failed")
    return 0 if rows else 1


# ---------------------------------------------------------------------------- projection from the pilot

def dotenv(code_dir: str) -> dict:
    """The settings run_pipeline.sh will use: .env.example, overridden by the environment."""
    env = {}
    with open(os.path.join(code_dir, '.env.example'), encoding='utf-8') as f:
        for line in f:
            m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line.rstrip('\r\n'))
            if m:
                env[m.group(1)] = m.group(2).strip().strip('"\'')
    env.update({k: v for k, v in os.environ.items() if k in env and v})
    return env


def label_dir_stats(path: str):
    """(documents measured, mean seconds per document incl. the log-prob pass) of one Stage A label dir."""
    secs = []
    for p in glob.glob(os.path.join(path, '*.json')):
        if os.path.basename(p) == 'measure_config.json':
            continue
        with open(p, encoding='utf-8') as f:
            rec = json.load(f)
        secs.append(float(rec.get('seconds') or 0) + float(rec.get('seconds_logprob') or 0))
    return len(secs), (sum(secs) / len(secs) if secs else None)


def cmd_projection(args, state):
    env = dotenv(args.code_dir)
    readers = env.get('LABEL_READERS', '').split()
    train_sources = env.get('TRAIN_SOURCES', '').split()
    eval_sources = [s for s in env.get('EVAL_SOURCES', '').split(',') if s]
    tp_large, pattern = int(env.get('TP_LARGE') or 2), env.get('LARGE_READER_PATTERN') or LARGE_READER_PATTERN
    p_train, p_dev, p_test = map(int, args.pilot_sizes.split())
    n_train, n_dev, n_test = map(int, args.main_sizes.split())
    oracle_n = min(int(env.get('ORACLE_N') or 100), n_test)
    try:
        sys.path.insert(0, args.code_dir)
        from ttcompress.sources import HOP
        single = {s for s, h in HOP.items() if h == 'single'}
    except Exception:  # noqa: BLE001 -- a projection must not fail on an import
        single = SINGLE_HOP
    raw = os.path.join(args.labels, 'raw')
    notes = []

    def label_seconds(sources, oracle_sources, fresh: bool) -> float:
        """Reader time still to spend on these sources; fresh = the documents change (an ablation), so
        nothing measured counts, only the per-document times are borrowed from the main labels."""
        total, jobs = 0.0, 0
        for i, reader in enumerate(readers):
            tag = reader.strip('/').replace('/', '--')
            procs = max(1, args.gpus // (tp_large if re.search(pattern, reader) else 1))
            dirs = glob.glob(os.path.join(raw, tag, '*'))
            all_means = [m for m in (label_dir_stats(d)[1] for d in dirs) if m]
            fallback = sum(all_means) / len(all_means) if all_means else None
            plan = [(s, split, n) for s in sources for split, n in (('train', n_train), ('dev', n_dev))]
            if i == 0:   # oracle_beta test labels: primary reader only
                plan += [(s, 'test', oracle_n) for s in oracle_sources]
            for src, split, n in plan:
                have, mean = label_dir_stats(os.path.join(raw, tag, f'{src}_{split}'))
                mean = mean or fallback
                if mean is None:
                    notes.append(f'no timing for {reader} {src}/{split}')
                    continue
                total += max(0, n - (0 if fresh else have)) * mean / procs
                jobs += 1
        return total + jobs * MODEL_STARTUP_S

    t = state['timings'].get('pilot', {})
    if not t.get('fit') or not t.get('select'):
        notes.append('pilot stage times missing: train/eval not projected')
    train_s = t.get('fit', 0) * n_train / p_train
    eval_s = t.get('select', 0) * n_test / p_test
    main_s = label_seconds(train_sources, eval_sources, False) + train_s + eval_s + STAGE_OVERHEAD_S
    abl_train_s = train_s * 5 / 9   # ablations skip the 4 seed reruns (ABL_EXTRA_SEEDS=none)
    hard_s = label_seconds([s for s in train_sources if s in single], [s for s in eval_sources if s in single],
                           True) + abl_train_s + eval_s + STAGE_OVERHEAD_S
    pad_s = label_seconds([s for s in train_sources if s not in single],
                          [s for s in eval_sources if s not in single], True) + abl_train_s + eval_s + STAGE_OVERHEAD_S
    start = dt.datetime.now(dt.timezone.utc)
    eta = lambda s: (start + dt.timedelta(seconds=s)).strftime('%Y-%m-%d %H:%M UTC')  # noqa: E731
    state['projection'] = {
        'computed': now(), 'main_hours': round(main_s / 3600, 4), 'abl_hard_hours': round(hard_s / 3600, 4),
        'abl_pad_hours': round(pad_s / 3600, 4), 'main_eta': eta(main_s), 'all_eta': eta(main_s + hard_s + pad_s),
        'notes': notes + ['labels from the pilot\'s per-document reader times; train and eval scaled linearly from '
                          'the pilot\'s stage times (model start-ups included, so rather an upper bound); '
                          'abl_pad documents are longer than the main ones: its label time is a lower bound']}
    print(json.dumps(state['projection'], indent=2))


# ---------------------------------------------------------------------------- STATUS.md + upload

def gpu_snapshot() -> str:
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                               '--format=csv,noheader'], capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:  # noqa: BLE001
        return ''


def render(state: dict, heartbeat: bool) -> str:
    info, t_now = state['info'], dt.datetime.now(dt.timezone.utc)
    out = [f"# ttcompress campaign `{state['campaign']}`", '',
           f"Updated {now()} ({'heartbeat' if heartbeat else 'event'}) · build `{info.get('build', '?')}` · "
           f"{info.get('gpus', '?')} GPU · host `{socket.gethostname()}`", '']
    if info.get('result'):
        out += [f"**{info['result']}**", '']
    out += ['## Phases', '', '| phase | state | started | ended | hours | note |', '|---|---|---|---|---|---|']
    for name in state['order']:
        p = state['phases'][name]
        start, end = parse_time(p.get('started')), parse_time(p.get('ended'))
        hours = f"{((end or t_now) - start).total_seconds() / 3600:.2f}" if start else ''
        out.append(f"| {name} | {p.get('state')} | {p.get('started') or ''} | {p.get('ended') or ''} | {hours} "
                   f"| {cell(p.get('note'))} |")
    if state['arms']:
        out += ['', '## Published compressors (probed on the smoke run)', '', '| arm | state | note |', '|---|---|---|']
        out += [f"| {k} | {a.get('state')} | {cell(a.get('note'))} |" for k, a in state['arms'].items()]
    proj = state.get('projection')
    if proj:
        out += ['', '## Projection from the pilot', '',
                f"- full run: {proj['main_hours']:.1f} h (done ≈ {proj['main_eta']}, from {proj['computed']})",
                f"- ablations: hard {proj['abl_hard_hours']:.1f} h, pad {proj['abl_pad_hours']:.1f} h "
                f"(everything done ≈ {proj['all_eta']})"] + [f"- {n}" for n in proj.get('notes', [])]
    if state['runs']:
        out += ['', '## Reports', '']
        out += [f"- `{name}` ({r['updated']}): [runs/{name}/](runs/{name}/)"
                + (f" · published compressors: {r['arms']}" if r.get('arms') else '') for name, r in state['runs'].items()]
    shown = {k: v for k, v in info.items() if k not in ('build', 'gpus', 'result')}
    if shown:
        out += ['', '## Settings', ''] + [f"- {k}: `{v}`" for k, v in shown.items()]
    cur = state.get('current') or {}
    if cur and not info.get('result'):
        out += ['', f"## Now: `{cur.get('phase')}` (since {cur.get('since')})", '', '```', tail(cur.get('log'), 15), '```']
        newest = newest_file(os.path.join(cur['run_root'], 'logs')) if cur.get('run_root') else None
        if newest:
            out += ['', f"newest run log `{os.path.relpath(newest, cur['run_root'])}`:", '```', tail(newest, 10), '```']
        gpus = gpu_snapshot()
        if gpus:
            out += ['', 'GPU (index, util, mem used, mem total):', '```', gpus, '```']
    return '\n'.join(out) + '\n'


def upload(folder: str, state: dict, campaign_dir: str) -> str:
    token = os.environ.get('HF_TOKEN')
    if not token:
        return 'no HF token: not uploaded'
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        repo = os.environ.get('STATUS_REPO') or state['info'].get('status_repo')
        if not repo:
            ns = os.environ.get('HF_NAMESPACE') or api.whoami()['name']
            repo = f"{ns}/{os.environ.get('HF_REPO_PREFIX') or 'ttcompress'}-campaign-{state['campaign']}"
        if state['info'].get('status_repo') != repo:
            state['info']['status_repo'] = repo
            save(campaign_dir, state)
        api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
        api.upload_folder(folder_path=folder, repo_id=repo, repo_type='dataset', commit_message=f'status {now()}')
        return f'uploaded to https://huggingface.co/datasets/{repo}'
    except Exception as exc:  # noqa: BLE001 -- progress reporting must never stop the campaign
        return f'upload failed: {type(exc).__name__}: {exc}'


def cmd_publish(args, state):
    folder = os.path.join(args.dir, 'status')
    os.makedirs(os.path.join(folder, 'logs'), exist_ok=True)
    for name, run in state['runs'].items():
        dst = os.path.join(folder, 'runs', name)
        src = os.path.join(run['path'], 'results', 'eval_test')
        if not os.path.isdir(src):     # e.g. the comparison: its own dir holds compare.{md,tex,csv}
            src = run['path']
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(src, '*')):
            if os.path.isfile(f) and (os.path.basename(f) in REPORT_FILES or src == run['path']):
                shutil.copy2(f, dst)
        if os.path.isdir(os.path.join(src, 'paper')):
            shutil.copytree(os.path.join(src, 'paper'), os.path.join(dst, 'paper'), dirs_exist_ok=True)
    for name, p in state['phases'].items():
        if p.get('log'):
            with open(os.path.join(folder, 'logs', f'{name}.txt'), 'w', encoding='utf-8') as f:
                f.write(tail(p['log'], TAIL_LINES))
    campaign_log = os.path.join(args.dir, 'logs', 'campaign.log')
    with open(os.path.join(folder, 'logs', 'campaign.txt'), 'w', encoding='utf-8') as f:
        f.write(tail(campaign_log, TAIL_LINES))
    text = render(state, args.heartbeat)
    with open(os.path.join(folder, 'STATUS.md'), 'w', encoding='utf-8') as f:
        f.write(text)
    result = upload(folder, state, args.dir)
    if not args.heartbeat:
        print(text + f"\n(status: {result})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help="the campaign dir ($BASE/campaigns/<name>)")
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('phase')
    p.add_argument('name')
    p.add_argument('state', choices=['running', 'done', 'failed', 'interrupted', 'skipped'])
    p.add_argument('--note', default='')
    p.add_argument('--log', default=None)
    p = sub.add_parser('arm')
    p.add_argument('label')
    p.add_argument('state', choices=['probing', 'ok', 'failed'])
    p.add_argument('--spec', default=None)
    p.add_argument('--note', default='')
    p = sub.add_parser('timing')
    p.add_argument('run')
    p.add_argument('group', help="first stage of the timed group: preflight | fit | select")
    p.add_argument('seconds', type=float)
    p = sub.add_parser('run', help="register a finished run: its report is published")
    p.add_argument('name')
    p.add_argument('path')
    p.add_argument('--arms', default='')
    p = sub.add_parser('current', help="what is running now (shown by the heartbeat)")
    p.add_argument('phase')
    p.add_argument('--log', default=None)
    p.add_argument('--run-root', default=None)
    p = sub.add_parser('info')
    p.add_argument('key')
    p.add_argument('value')
    p = sub.add_parser('probe-result')
    p.add_argument('--eval-dir', required=True)
    p.add_argument('--arm', required=True)
    p = sub.add_parser('projection')
    p.add_argument('--code-dir', required=True)
    p.add_argument('--labels', required=True, help="LABELS (holds raw/<reader>/<source>_<split>/)")
    p.add_argument('--gpus', type=int, required=True)
    p.add_argument('--pilot-sizes', required=True, help="'N_TRAIN N_DEV N_TEST' of the pilot")
    p.add_argument('--main-sizes', required=True, help="'N_TRAIN N_DEV N_TEST' of the full run")
    p = sub.add_parser('publish')
    p.add_argument('--heartbeat', action='store_true', help="periodic: no stdout")
    args = ap.parse_args()

    if args.cmd == 'probe-result':
        sys.exit(cmd_probe_result(args))
    with locked(args.dir):
        state = load(args.dir)
        handler = {'phase': cmd_phase, 'arm': cmd_arm, 'timing': cmd_timing, 'run': cmd_run, 'current': cmd_current,
                   'info': cmd_info, 'projection': cmd_projection, 'publish': cmd_publish}[args.cmd]
        handler(args, state)
        save(args.dir, state)


if __name__ == '__main__':
    main()
