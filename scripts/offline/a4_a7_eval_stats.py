"""A4-A7 on the stored test-set answers (primary reader Qwen3-8B); no reader calls.

A4  ours_beta - ours_logprob (arm label `abl_logprob`), F1 and EM, paired cluster bootstrap (two-sided,
    ttcompress.metrics.paired_bootstrap_diff, 5000 reps), Holm across the 10 source x ratio cells.
A5  small-cluster robustness of H2b (TOST +-0.02) and H1-oracle (non-inferiority, margin 0.05): cluster
    bootstrap (the report's procedure) vs wild cluster bootstrap CI (Rademacher on cluster residual sums),
    a cluster sign-flip test (= restricted wild bootstrap; exact enumeration when G <= 20) and a CR1 t CI.
A6  design MDE = (z_.95 + z_.80) * SE, SE = cluster-robust SE of the paired mean; for TOST the margin
    needed for 80% power at a true difference of 0 is (z_.95 + z_.90) * SE. H3 SE from the report's CIs.
A7  realized budget use / compression per arm and F1 interpolated (linear in log compression) at EXIT's
    realized compression, from followup/report.json (contains the fill / sent arms).

    python scripts/offline/a4_a7_eval_stats.py
"""
from __future__ import annotations

import glob
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import norm, t as tdist

from common import DATA, PRIMARY, SOURCES, save
from ttcompress.metrics import holm_adjust, paired_bootstrap_diff, paired_bootstrap_test

N_BOOT = 5000
RATIOS = ('4.0', '8.0')


def load_cell(reader=PRIMARY):
    cell = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(DATA, 'eval', f'answers_{reader}_shard*.jsonl'))):
        with open(p, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                cell[(r['source'], str(r['ratio']), r['arm'])][r['doc_id']] = r
    return cell


def paired(cell, source, ratio, a, b, metric='f1'):
    A, B = cell.get((source, ratio, a)), cell.get((source, ratio, b))
    if not (A and B):
        return None
    common = sorted(set(A) & set(B))
    return (np.array([A[d][metric] for d in common], float) - np.array([B[d][metric] for d in common], float),
            [A[d]['cluster_id'] for d in common])


def cluster_sums(d, cl):
    idx = defaultdict(list)
    for i, c in enumerate(cl):
        idx[c].append(i)
    g = list(idx.values())
    return np.array([d[i].sum() for i in g]), np.array([len(i) for i in g], float)


def cr_se(d, cl):
    S, n = cluster_sums(d, cl)
    G, N = len(S), n.sum()
    e = S - n * d.mean()
    return float(np.sqrt(G / (G - 1) * np.sum(e ** 2)) / N), G


def rademacher(G, n_draw=20000, seed=0):
    if G <= 20:
        return np.array(list(itertools.product((-1.0, 1.0), repeat=G)))
    return np.random.default_rng(seed).choice((-1.0, 1.0), size=(n_draw, G))


def wild_ci(d, cl):
    S, n = cluster_sums(d, cl)
    N, m = n.sum(), d.mean()
    W = rademacher(len(S))
    boot = m + W @ (S - n * m) / N
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def signflip_p(d, cl, delta0, direction):
    """One-sided p for H0: mean <= delta0 (direction 'greater') or >= delta0 ('less'); flips cluster sums of
    (d - delta0). Exact when G <= 20."""
    S, n = cluster_sums(d, cl)
    e = S - n * delta0
    W = rademacher(len(S))
    T = W @ e
    obs = e.sum()
    if direction == 'greater':
        return float(((T >= obs - 1e-12).sum()) / len(T))
    return float(((T <= obs + 1e-12).sum()) / len(T))


def a4(cell):
    out = {}
    for metric in ('f1', 'em'):
        rows = []
        for s, r in itertools.product(SOURCES, RATIOS):
            pr = paired(cell, s, r, 'ours_beta', 'abl_logprob', metric)
            if pr is None:
                continue
            res = paired_bootstrap_diff(pr[0], pr[0] * 0, pr[1], n_boot=N_BOOT)
            rows.append({'source': s, 'ratio': r, 'n_clusters': len(set(pr[1])), **res})
        for row, h in zip(rows, holm_adjust([x['p'] for x in rows])):
            row['p_holm'] = h
        out[metric] = rows
    return out


def a5_a6(cell):
    out = {'H2b': [], 'H1-oracle': []}
    z_ni, z_tost = norm.ppf(0.95) + norm.ppf(0.8), norm.ppf(0.95) + norm.ppf(0.9)
    z_holm10 = norm.ppf(1 - 0.05 / 10) + norm.ppf(0.8)
    specs = [('H2b', s, r, 'span_sup', 'equivalence', 0.02) for s in ('uit_viquad', 'xquad_vi') for r in RATIOS]
    specs += [('H1-oracle', s, r, 'oracle_beta', 'noninferiority', 0.05) for s in SOURCES for r in RATIOS]
    for fam, s, r, vs, kind, margin in specs:
        d, cl = paired(cell, s, r, 'ours_beta', vs)
        boot = paired_bootstrap_test(d, d * 0, cl, kind, margin, N_BOOT)
        se, G = cr_se(d, cl)
        tq = tdist.ppf(0.975, G - 1)
        row = {'source': s, 'ratio': r, 'n': len(d), 'G': G, 'diff': float(d.mean()),
               'boot_ci': boot['ci95'], 'boot_p': boot['p'], 'wild_ci': wild_ci(d, cl),
               'cr1_t_ci': [float(d.mean() - tq * se), float(d.mean() + tq * se)], 'se_cr': se}
        if kind == 'equivalence':
            row['signflip_p'] = max(signflip_p(d, cl, -margin, 'greater'), signflip_p(d, cl, margin, 'less'))
            row['tost_margin_needed_80pct'] = float(z_tost * se)
            # 90% CI (the TOST decision interval) from the CR1 t
            tq90 = tdist.ppf(0.95, G - 1)
            row['cr1_t_90ci'] = [float(d.mean() - tq90 * se), float(d.mean() + tq90 * se)]
        else:
            row['signflip_p'] = signflip_p(d, cl, -margin, 'greater')
            row['mde'] = float(z_ni * se)
            row['mde_holm10'] = float(z_holm10 * se)
        out[fam].append(row)
    for fam in out:
        for row, h1, h2 in zip(out[fam], holm_adjust([x['boot_p'] for x in out[fam]]),
                               holm_adjust([x['signflip_p'] for x in out[fam]])):
            row['boot_p_holm'], row['signflip_p_holm'] = h1, h2
    return out


def h3_mde():
    with open(os.path.join(DATA, 'eval', 'report.json'), encoding='utf-8') as f:
        rep = json.load(f)
    z = norm.ppf(0.95) + norm.ppf(0.8)
    rows = []
    for fam in rep['hypotheses']:
        if fam['name'] not in ('H3', 'H3-heldout'):
            continue
        for t in fam['tests']:
            lo, hi = t['ci95']
            if lo is None:
                continue
            se = (hi - lo) / (2 * 1.96)
            rows.append({'family': fam['name'], 'source': t['source'], 'ratio': t['ratio'], 'weak': t['weak'],
                         'strong': t['strong'], 'gap': t['gap'], 'diff': t['diff'], 'se': se, 'mde': z * se})
    return rows


def a7():
    with open(os.path.join(DATA, 'followup', 'report.json'), encoding='utf-8') as f:
        rep = json.load(f)
    arms = ('ours_beta', 'ours_fill', 'ours_sent', 'exit', 'reranker', 'span_sup')
    cells = {(c['source'], c['ratio'], c['arm']): c for c in rep['cells'] if c['reader'] == PRIMARY}
    use = {(c['source'], c['ratio'], c['arm']): c['budget_use'] for c in rep['diagnostics']['coverage']}
    out = {}
    for s in SOURCES:
        rows = {}
        for a in arms:
            pts = [(cells[(s, r, a)]['compression'], cells[(s, r, a)]['f1']['mean']) for r in RATIOS if (s, r, a) in cells]
            rows[a] = {'compression': [p[0] for p in pts], 'f1': [p[1] for p in pts],
                       'budget_use': [use.get((s, r, a)) for r in RATIOS]}
        ex = rows['exit']['compression']
        for a, row in rows.items():
            if len(row['compression']) != 2:
                continue
            x = np.log(row['compression'])
            interp, extrap = [], []
            for target in ex:
                w = (np.log(target) - x[0]) / (x[1] - x[0])
                interp.append(float(row['f1'][0] + w * (row['f1'][1] - row['f1'][0])))
                extrap.append(bool(w < 0 or w > 1))
            row['f1_at_exit_compression'], row['extrapolated'] = interp, extrap
        out[s] = rows
    return out


def main():
    cell = load_cell()
    res = {'a4': a4(cell), 'a5_a6': a5_a6(cell), 'h3_mde': h3_mde(), 'a7': a7()}
    print(save('a4_a7_eval_stats.json', res))


if __name__ == '__main__':
    main()
