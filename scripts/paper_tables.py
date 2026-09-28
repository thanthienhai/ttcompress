#!/usr/bin/env python3
"""Paper-ready tables from a finished report (evaluate.py report -> report.json).

    python scripts/paper_tables.py --report results/eval_test/report.json --primary-reader Qwen--Qwen3-8B

Writes <report dir>/paper/:
  main_results.tex     token F1 per source x ratio x arm, primary reader (best compressor in bold,
                       oracles and full context set apart)
  hypotheses.tex       one row per confirmatory family: tests supported / run
  retention.tex        upgrade retention of every chunk arm, stable reader pairs only
  label_quality.tex    Stage A diagnostics (cv R², gold recall, position R², cross-reader rho)
  cost.tex             label cost per document vs selection ms per document
  seeds.tex            ours_beta / ours_ens F1 mean ± SD over training seeds (EXTRA_SEEDS)
  cells.csv, paired.csv, hypotheses.csv, retention.csv, by_depth.csv   everything, for plots / appendix
Numbers are copied, never recomputed: report.json is the single source of truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

# display order; arms absent from the report are skipped
ARM_ORDER = ['lead', 'random', 'bm25', 'embed', 'reranker', 'provence', 'xprovence', 'recomp', 'exit', 'llmlingua',
             'longllmlingua', 'llmlingua2', 'span_sup', 'abl_logprob', 'abl_posadj', 'ours_beta', 'ours_ens']
REFERENCE_ARMS = ['full', 'oracle_span', 'oracle_support', 'oracle_beta']   # upper bounds, never bolded
TEX_NAME = {'ours_beta': r'\textsc{Ours}-$\beta$', 'ours_ens': r'\textsc{Ours}-ens', 'span_sup': 'span-sup',
            'abl_logprob': r'abl.\ log-prob', 'abl_posadj': r'abl.\ pos-adj', 'llmlingua2': 'LLMLingua-2',
            'xprovence': 'XProvence', 'provence': 'Provence', 'recomp': 'RECOMP', 'exit': 'EXIT',
            'llmlingua': 'LLMLingua', 'longllmlingua': 'LongLLMLingua', 'bm25': 'BM25', 'embed': 'bge-m3',
            'reranker': 'bge-reranker',
            'oracle_span': r'oracle$_\text{span}$', 'oracle_support': r'oracle$_\text{support}$',
            'oracle_beta': r'oracle$_\beta$'}


def tex(s) -> str:
    return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def arm_tex(a) -> str:
    return TEX_NAME.get(a, tex(a))


def ratio_tex(r) -> str:
    return 'full' if r == 'full' else f"{float(r):g}$\\times$"


def write(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  {path}")


def write_csv(path, rows, fields):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"  {path}")


def main_results(report, reader) -> str:
    cells = [c for c in report['cells'] if c['reader'] == reader]
    arms = [a for a in ARM_ORDER if any(c['arm'] == a for c in cells)]
    refs = [a for a in REFERENCE_ARMS if a != 'full' and any(c['arm'] == a for c in cells)]
    by = {(c['source'], c['ratio'], c['arm']): c for c in cells}
    sources = sorted({c['source'] for c in cells})
    ratios = sorted({c['ratio'] for c in cells if c['ratio'] != 'full'}, key=float)
    cols = 'll' + 'r' * len(arms) + ('|' + 'r' * len(refs) if refs else '') + '|r'
    lines = [r'\begin{tabular}{' + cols + '}', r'\toprule',
             ' & '.join(['source', 'ratio'] + [arm_tex(a) for a in arms] + [arm_tex(a) for a in refs] + ['full'])
             + r' \\', r'\midrule']
    for s in sources:
        full = by.get((s, 'full', 'full'))
        for i, r in enumerate(ratios):
            vals = {a: by[(s, r, a)]['f1']['mean'] for a in arms + refs if (s, r, a) in by}
            best = max((vals[a] for a in arms if a in vals), default=None)
            row = [tex(s) if i == 0 else '', ratio_tex(r)]
            for a in arms + refs:
                v = vals.get(a)
                cell = '' if v is None else f"{100 * v:.1f}"
                row.append(r'\textbf{' + cell + '}' if v is not None and a in arms and v == best else cell)
            row.append(f"{100 * full['f1']['mean']:.1f}" if full and i == 0 else '')
            lines.append(' & '.join(row) + r' \\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    return '\n'.join(lines) + '\n'


def hypotheses(report) -> str:
    lines = [r'\begin{tabular}{llrrr}', r'\toprule',
             r'hypothesis & test & supported & \ldots{} and every seed agrees & budget-mismatched \\', r'\midrule']
    for f in report.get('hypotheses', []):
        sup = 'not run' if not f['n_tests'] else f"{f['n_supported']}/{f['n_tests']}"
        robust = '--' if f.get('n_seed_robust') is None else f"{f['n_seed_robust']}/{f['n_tests']}"
        margin = f" ($\\delta={f['margin']:g}$)" if f['margin'] else ''
        lines.append(f"{tex(f['name'])} & {f['kind']}{margin} & {sup} & {robust} & {f['n_budget_mismatch']} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def retention(report) -> str:
    rows = [u for u in report['upgrade_retention'] if u.get('stable', True) and u['ratio'] != 'full']
    arms = [a for a in ARM_ORDER + ['oracle_beta'] if any(u['arm'] == a for u in rows)]
    keys = sorted({(u['source'], u['weak'], u['strong'], u['ratio']) for u in rows})
    by = {(u['source'], u['weak'], u['strong'], u['ratio'], u['arm']): u for u in rows}
    lines = [r'\begin{tabular}{lll' + 'r' * len(arms) + '}', r'\toprule',
             ' & '.join(['source', r'weak $\to$ strong', 'ratio'] + [arm_tex(a) for a in arms]) + r' \\',
             r'\midrule']
    for k in keys:
        row = [tex(k[0]), f"{tex(k[1])} $\\to$ {tex(k[2])}", ratio_tex(k[3])]
        row += [f"{by[k + (a,)]['retention']:.2f}" if k + (a,) in by else '' for a in arms]
        lines.append(' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def label_quality(report) -> str:
    fmt = lambda v, f='.2f': '' if v is None or v != v else format(v, f)  # noqa: E731
    lines = [r'\begin{tabular}{lllrrrrr}', r'\toprule',
             r'reader & target & set & docs (inf.) & cv $R^2$ & gold rec.@g & pos.\ $R^2$ & $\rho_\text{readers}$ \\',
             r'\midrule']
    for r in report.get('label_quality', []):
        rho = r.get('cross_reader_spearman') or {}
        rho_mean = sum(rho.values()) / len(rho) if rho else None
        lines.append(f"{tex(r['reader'])} & {tex(r['target'])} & {tex(r['set'])} & {r['n_docs']} ({r['n_informative']}) & "
                     f"{fmt(r['mean_cv_r2'])} & {fmt(r['gold_recall_at_g'])} & {fmt(r['position_r2'])} & "
                     f"{fmt(rho_mean)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def seeds(report, reader) -> str:
    lines = [r'\begin{tabular}{lllr}', r'\toprule', r'arm & source & ratio & F1 (mean $\pm$ SD over seeds) \\',
             r'\midrule']
    for r in report.get('seeds', []):
        if r['reader'] == reader:
            lines.append(f"{arm_tex(r['arm'])} & {tex(r['source'])} & {ratio_tex(r['ratio'])} & "
                         f"{100 * r['mean']:.1f} $\\pm$ {100 * r['sd']:.1f} ({len(r['f1_by_seed'])} seeds) \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def cost(report) -> str:
    c = report.get('cost') or {}
    lines = [r'\begin{tabular}{llrrr}', r'\toprule', r'reader & label set & docs & reader calls / doc & s / doc \\',
             r'\midrule']
    for r in c.get('labels', []):
        lines.append(f"{tex(r['reader'])} & {tex(r['set'])} & {r['n_docs']} & {r['calls_per_doc']:.0f} & "
                     f"{r['seconds_per_doc']:.1f} \\\\")
    lines += [r'\midrule', r'source & arm & & & ms / doc \\', r'\midrule']
    for r in c.get('select_ms', []):
        lines.append(f"{tex(r['source'])} & {arm_tex(r['arm'])} & & & {r['select_ms']:.0f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', required=True, help="report.json written by evaluate.py report")
    ap.add_argument('--primary-reader', default=None, help="reader tag for the main table (default: first reader)")
    ap.add_argument('--out-dir', default=None, help="default: <report dir>/paper")
    args = ap.parse_args()
    with open(args.report, encoding='utf-8') as f:
        report = json.load(f)
    reader = args.primary_reader if args.primary_reader in report['readers'] else report['readers'][0]
    out = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.report)), 'paper')
    os.makedirs(out, exist_ok=True)
    print(f"paper tables (main reader {reader}) -> {out}")

    write(os.path.join(out, 'main_results.tex'), main_results(report, reader))
    write(os.path.join(out, 'hypotheses.tex'), hypotheses(report))
    write(os.path.join(out, 'retention.tex'), retention(report))
    if report.get('label_quality'):
        write(os.path.join(out, 'label_quality.tex'), label_quality(report))
    if report.get('cost'):
        write(os.path.join(out, 'cost.tex'), cost(report))
    if report.get('seeds'):
        write(os.path.join(out, 'seeds.tex'), seeds(report, reader))
        write_csv(os.path.join(out, 'seeds.csv'),
                  [{**{k: r[k] for k in ('arm', 'reader', 'source', 'ratio', 'mean', 'sd')},
                    'f1_by_seed': json.dumps(r['f1_by_seed'])} for r in report['seeds']],
                  ['arm', 'reader', 'source', 'ratio', 'mean', 'sd', 'f1_by_seed'])

    cells = [{**{k: c[k] for k in ('reader', 'source', 'ratio', 'arm', 'n', 'n_clusters', 'compression',
                                   'select_ms', 'truncated_rate')},
              'over_budget_rate': c.get('over_budget_rate'),
              **{f"{m}_{part}": (c[m]['mean'] if part == 'mean' else c[m]['ci95'][part == 'hi'])
                 for m in ('f1', 'em', 'answer_recall', 'gold_recall') for part in ('mean', 'lo', 'hi')}}
             for c in report['cells']]
    write_csv(os.path.join(out, 'cells.csv'), cells, list(cells[0]) if cells else [])
    paired = [{**p, 'ci_lo': p['ci95'][0], 'ci_hi': p['ci95'][1]} for p in report['paired']]
    write_csv(os.path.join(out, 'paired.csv'), paired,
              ['reader', 'source', 'ratio', 'ours', 'vs', 'diff', 'ci_lo', 'ci_hi', 'p', 'q_bh', 'p_holm', 'n'])
    hyp = [{'family': f['name'], **t, 'ci_lo': t['ci95'][0], 'ci_hi': t['ci95'][1]}
           for f in report.get('hypotheses', []) for t in f['tests']]
    write_csv(os.path.join(out, 'hypotheses.csv'), hyp,
              ['family', 'reader', 'weak', 'strong', 'source', 'ratio', 'ours', 'vs', 'diff', 'ci_lo', 'ci_hi', 'p',
               'p_holm', 'supported', 'seeds_agree', 'n', 'n_clusters', 'gap', 'tokens_ours_over_vs',
               'budget_mismatch'])
    ret = [{**u, 'ci_lo': u['ci95'][0], 'ci_hi': u['ci95'][1]} for u in report['upgrade_retention']]
    write_csv(os.path.join(out, 'retention.csv'), ret,
              ['source', 'weak', 'strong', 'ratio', 'arm', 'retention', 'ci_lo', 'ci_hi', 'gap', 'stable', 'n'])
    depth = [{'reader': d['reader'], 'source': d['source'], 'ratio': d['ratio'], 'arm': d['arm'], 'bin': b, **v}
             for d in report['by_depth'] for b, v in d['bins'].items()]
    write_csv(os.path.join(out, 'by_depth.csv'), depth,
              ['reader', 'source', 'ratio', 'arm', 'bin', 'n', 'f1', 'gold_recall'])


if __name__ == '__main__':
    main()
