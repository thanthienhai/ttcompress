#!/usr/bin/env python3
"""Build the ONE file to submit on the cluster: dist/ttcompress_job.sh = scripts/cluster_job_template.sh with
this repository (working tree, uncommitted changes included) embedded as a base64 tar.gz. What the job does:
the template's header; its settings: the top of scripts/cluster_campaign.sh.

    python scripts/build_cluster_job.py                                   # -> dist/ttcompress_job.sh
    python scripts/build_cluster_job.py --set PHASES="main abl_hard" --set CAMPAIGN=coling2027
    python scripts/build_cluster_job.py --out /tmp/job.sh

No secret is embedded: .env is never packed; the job looks for HF_TOKEN on the cluster (HF_TOKEN_SOURCES in
scripts/cluster_campaign.sh) unless the job's environment sets it. Packed: the top-level files, ttcompress/,
scripts/ and tests/ (tracked or not ignored); paper/, reports/, docs/ and data/ stay home.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import hashlib
import io
import os
import re
import shlex
import subprocess
import tarfile
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, 'scripts', 'cluster_job_template.sh')
INCLUDED_DIRS = ('ttcompress/', 'scripts/', 'tests/')   # plus the files at the top level
EXCLUDED_NAMES = {'.env', 'TODO.md'}
# CRLF -> LF: a Windows checkout (core.autocrlf) must not ship '\r' to bash on Linux
TEXT_SUFFIXES = ('.sh', '.py', '.md', '.txt', '.json', '.example', '.cfg', '.toml', '.ini', '.yaml', '.yml')


def git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def repo_files():
    for rel in git('ls-files', '-co', '--exclude-standard', '-z').split('\0'):
        if (not rel or ('/' in rel and not rel.startswith(INCLUDED_DIRS)) or os.path.basename(rel) in EXCLUDED_NAMES
                or rel.endswith('.zip') or not os.path.isfile(os.path.join(ROOT, rel))):
            continue
        yield rel


def pack(files, mtime: int) -> bytes:
    """Deterministic tar.gz: same files -> same bytes -> same build id."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.PAX_FORMAT) as tar:
        for rel in sorted(files):
            with open(os.path.join(ROOT, rel), 'rb') as f:
                data = f.read()
            if rel.endswith(TEXT_SUFFIXES) or os.path.basename(rel).startswith('.'):
                data = data.replace(b'\r\n', b'\n')
            info = tarfile.TarInfo(rel)
            info.size, info.mtime = len(data), mtime
            info.mode = 0o755 if rel.endswith('.sh') else 0o644
            tar.addfile(info, io.BytesIO(data))
    return gzip.compress(buf.getvalue(), mtime=0)


def settings_block(pairs) -> str:
    """--set KEY=VALUE -> lines that keep a value the job's environment already has."""
    lines = []
    for pair in pairs:
        key, sep, value = pair.partition('=')
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        lines.append(f'[[ -n "${{{key}+x}}" ]] || {key}={shlex.quote(value)}; export {key}')
    return '\n'.join(lines) or '# (none: build_cluster_job.py --set KEY=VALUE adds them here)'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(ROOT, 'dist', 'ttcompress_job.sh'))
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help="bake a setting of scripts/cluster_campaign.sh into the job (repeat)")
    args = ap.parse_args()

    files = list(repo_files())
    head = git('rev-parse', '--short', 'HEAD').strip()
    changed = {line[3:].split(' -> ')[-1] for line in git('status', '--porcelain', '-uall').splitlines()}
    dirty = bool(changed & set(files))
    commit_time = int(git('log', '-1', '--format=%ct').strip())
    payload = pack(files, commit_time)
    build_id = f"{head}{'-dirty' if dirty else ''}-{hashlib.sha256(payload).hexdigest()[:10]}"
    info = (f"commit {head}{' + uncommitted changes' if dirty else ''}, {len(files)} files, "
            f"built {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC")

    with open(TEMPLATE, encoding='utf-8') as f:
        text = f.read().replace('\r\n', '\n')
    encoded = '\n'.join(textwrap.wrap(base64.b64encode(payload).decode('ascii'), 76))
    for key, value in (('@@SETTINGS@@', settings_block(args.set)), ('@@BUILD_ID@@', build_id),
                       ('@@BUILD_INFO@@', info), ('@@PAYLOAD@@', encoded)):
        text = text.replace(key, value)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    print(f"{args.out}: build {build_id} ({info}; {os.path.getsize(args.out) / 1024:.0f} KB)")
    if args.set:
        print("baked settings:\n  " + settings_block(args.set).replace('\n', '\n  '))


if __name__ == '__main__':
    main()
