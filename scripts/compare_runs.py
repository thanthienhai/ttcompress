#!/usr/bin/env python3
"""Side-by-side view of several finished runs: the distractor ablation (main vs DISTRACTORS=hard) and the
data-size curve (mid vs full). Numbers are copied from each run's report.json, never recomputed.

    python scripts/compare_runs.py --primary-reader Qwen--Qwen3-8B --out-dir results/compare \\
        --run random=/mnt/.../runs/main/results/eval_test/report.json \\
        --run hard=/mnt/.../runs/main_distractors-hard/results/eval_test/report.json

Writes compare.md, compare.tex, compare.csv (token F1 [95% CI] per source x ratio x arm, one column per
run) and the per-run hypothesis verdicts. Runs are compared descriptively: their test documents differ
(other distractors, or a different N_TEST -- nested, so a smaller run's test set is a subset), so the
paired tests of each report stay the inferential evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

DEFAULT_ARMS = 'lead,bm25,embed,reranker,xprovence,llmlingua2,span_sup,ours_beta,ours_ens,oracle_span,oracle_beta'


def load_runs(specs):
    runs = []
    for spec in specs:
        if '=' not in spec:
            raise SystemExit(f"--run expects name=path/to/report.json, got {spec!r}")
        name, path = spec.split('=', 1)
        with open(path, encoding='utf-8') as f:
            runs.append((name, json.load(f)))
    return runs


def f1_table(runs, reader, arms):
    """[(source, ratio, arm, {run: cell or None})], full context first per source."""
    index = {name: {(c['source'], c['ratio'], c['arm']): c for c in r['cells'] if c['reader'] == reader}
             for name, r in runs}
    keys = sorted({k[:2] for cells in index.values() for k in cells},
                  key=lambda k: (k[0], k[1] != 'full', float(k[1]) if k[1] != 'full' else 0.0))
    rows = []
    for source, ratio in keys:
        for arm in (['full'] if ratio == 'full' else arms):
            per_run = {name: index[name].get((source, ratio, arm)) for name, _ in runs}
            if any(per_run.values()):
                rows.append((source, ratio, arm, per_run))
    return rows


def fmt_cell(c, digits=3):
    if not c:
        return ''
    lo, hi = c['f1']['ci95']
    ci = f" [{lo:.{digits}f}, {hi:.{digits}f}]" if lo is not None else ''
    return f"{c['f1']['mean']:.{digits}f}{ci}"


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True, help="name=path/to/report.json (repeat, >= 2)")
    ap.add_argument('--primary-reader', required=True, help="reader tag, e.g. Qwen--Qwen3-8B")
    ap.add_argument('--arms', default=DEFAULT_ARMS, help="comma list, in display order (missing arms are skipped)")
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    runs = load_runs(args.run)
    names = [n for n, _ in runs]
    arms = [a for a in args.arms.split(',') if a]
    os.makedirs(args.out_dir, exist_ok=True)
    rows = f1_table(runs, args.primary_reader, arms)

    md = [f"# Run comparison — reader `{args.primary_reader}`", '',
          'Token F1 [95% cluster-bootstrap CI], copied from each report. Descriptive: the runs\' test documents '
          'differ; each report\'s paired tests are the inferential evidence.', '',
          '| source | ratio | arm | ' + ' | '.join(f"{n} (n)" for n in names) + ' |',
          '|---|---|---|' + '---|' * len(names)]
    for source, ratio, arm, per_run in rows:
        md.append(f"| {source} | {ratio} | {arm} | " + ' | '.join(
            f"{fmt_cell(c)} ({c['n']})" if c else '' for c in per_run.values()) + ' |')
    md += ['', '## Hypothesis verdicts per run (supported / tests)', '',
           '| family | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    families = []
    for _, r in runs:
        families += [f['name'] for f in r.get('hypotheses', []) if f['name'] not in families]
    for fam in families:
        cells = []
        for _, r in runs:
            f = next((x for x in r.get('hypotheses', []) if x['name'] == fam), None)
            cells.append('' if not f or not f['n_tests'] else f"{f['n_supported']}/{f['n_tests']}")
        md.append(f"| {fam} | " + ' | '.join(cells) + ' |')
    with open(os.path.join(args.out_dir, 'compare.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')

    tex = [r'\begin{tabular}{lll' + 'r' * len(names) + '}', r'\toprule',
           ' & '.join(['source', 'ratio', 'arm'] + [n.replace('_', r'\_') for n in names]) + r' \\', r'\midrule']
    for source, ratio, arm, per_run in rows:
        ratio_tex = 'full' if ratio == 'full' else f"{float(ratio):g}$\\times$"
        tex.append(' & '.join([source.replace('_', r'\_'), ratio_tex, arm.replace('_', r'\_')] +
                              [f"{100 * c['f1']['mean']:.1f}" if c else '' for c in per_run.values()]) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(args.out_dir, 'compare.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')

    with open(os.path.join(args.out_dir, 'compare.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source', 'ratio', 'arm'] + [f"{n}_{k}" for n in names for k in ('f1', 'ci_lo', 'ci_hi', 'n')])
        for source, ratio, arm, per_run in rows:
            vals = []
            for c in per_run.values():
                vals += [c['f1']['mean'], *c['f1']['ci95'], c['n']] if c else ['', '', '', '']
            w.writerow([source, ratio, arm] + vals)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
