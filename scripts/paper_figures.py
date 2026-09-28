#!/usr/bin/env python3
"""Paper figures from evaluate.py's report.json (the tables come from paper_tables.py).

    python scripts/paper_figures.py --report results/eval_test/report.json --primary-reader Qwen--Qwen3-8B

  pareto.pdf     RQ1: F1 vs compression time per compressor, one panel per source (primary reader)
  retention.pdf  RQ3: upgrade retention of ours_beta vs ours_ens per reader pair, one panel per source
  depth.pdf      single-hop: F1 by needle-depth quintile (appendix)

Written to <report dir>/paper/figures (copy them to paper/figures/, where main.tex looks for them).
Static print figures, one color per entity across every figure (ours_beta blue, ours_ens orange, the
rest in grays with a shape per published compressor), from the validated palette in paper/main.tex;
results are never pooled across sources. The numbers themselves are in the tables."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLUE, ORANGE = '#2a78d6', '#eb6834'                  # categorical slots 1-2
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'   # primary / secondary / muted ink
GRID, AXIS = '#e1e0d9', '#c3c2b7'

SOURCES = [('uit_viquad', 'UIT-ViQuAD'), ('xquad_vi', 'XQuAD-vi'), ('vimqa', 'VIMQA'), ('hotpotqa', 'HotpotQA'),
           ('2wiki', '2Wiki')]
SIMPLE = ('lead', 'random', 'bm25', 'embed', 'reranker')
PUBLISHED = [('provence', 'Provence', 's'), ('xprovence', 'XProvence', 'D'), ('recomp', 'RECOMP', '^'),
             ('exit', 'EXIT', 'v'), ('llmlingua', 'LLMLingua', 'P'), ('longllmlingua', 'LongLLMLingua', 'X'),
             ('llmlingua2', 'LLMLingua-2', 'h')]
OURS = [('ours_beta', 'Ours-β', BLUE), ('ours_ens', 'Ours-ens', ORANGE)]
DEPTH_ARMS = [('ours_beta', 'Ours-β', BLUE, '-'), ('ours_ens', 'Ours-ens', ORANGE, '-'),
              ('reranker', 'bge-reranker', INK2, '-'), ('lead', 'lead', MUTED, '-')]


def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        # the paper's serif; every fallback covers Vietnamese
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'TeX Gyre Termes', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'legend.fontsize': 6.5,
        'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2, 'ytick.major.size': 2,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'grid.linestyle': '-',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
        'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,   # embedded TrueType (ACL requires embedding)
        'savefig.dpi': 300,
    })
    return plt


READER_ALIASES = {'Llama-SEA-LION-v3-8B': 'SEA-LION-8B'}


def short_reader(tag: str) -> str:
    name = tag.split('--')[-1]
    return READER_ALIASES.get(name, name)


def _legend(fig, handles, ncol, **kw):
    """Legend above the panels, outside the plot area (constrained layout makes room for it)."""
    try:
        return fig.legend(handles=handles, loc='outside upper center', ncol=ncol, **kw)
    except ValueError:   # matplotlib < 3.7 has no 'outside' locations
        return fig.legend(handles=handles, loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, 1.0), **kw)


def _ratio_match(value, ratio: float) -> bool:
    return value != 'full' and abs(float(value) - ratio) < 1e-9


def _present_sources(report):
    have = {c['source'] for c in report['cells']}
    return [(s, name) for s, name in SOURCES if s in have] + \
           sorted((s, s) for s in have if s not in dict(SOURCES))


def _mark(ax, x, y, color, marker='o', size=5.5, hollow=False, zorder=3):
    ax.plot([x], [y], linestyle='none', marker=marker, markersize=size, zorder=zorder,
            markerfacecolor='white' if hollow else color, markeredgecolor=color if hollow else 'white',
            markeredgewidth=1.1 if hollow else 0.8)


def pareto(report, reader, ratio, plt):
    """F1 against compression time: the amortization claim (RQ1) is ours_beta near oracle_beta in F1 at a
    small fraction of its time. oracle_beta's time is its Stage A labeling cost (K+1 reader calls per test
    document), not the table lookup `select` records for it; without label costs in the report it is left out."""
    from matplotlib.lines import Line2D
    sources = _present_sources(report)
    cells = {(c['source'], c['arm']): c for c in report['cells']
             if c['reader'] == reader and _ratio_match(c['ratio'], ratio)}
    full = {c['source']: c['f1']['mean'] for c in report['cells'] if c['reader'] == reader and c['arm'] == 'full'}
    label_ms = {lab['set'].rsplit('_', 1)[0]: 1000 * lab['seconds_per_doc']
                for lab in (report.get('cost') or {}).get('labels', [])
                if lab['reader'] == reader and lab['set'].endswith('_test')}
    if not cells:
        return None
    fig, axes = plt.subplots(1, len(sources), figsize=(6.3, 2.0), sharex=True, squeeze=False, layout='constrained')
    for ax, (src, name) in zip(axes[0], sources):
        ax.set_title(name, color=INK)
        ax.set_xscale('log')
        ax.grid(True, which='major', axis='both')
        ax.grid(False, which='minor')
        if src in full:
            ax.axhline(full[src], color=AXIS, linewidth=0.8, zorder=1)
            ax.annotate('toàn ngữ cảnh', xy=(0.02, full[src]), xycoords=('axes fraction', 'data'),
                        xytext=(0, 1.5), textcoords='offset points', color=MUTED, fontsize=5.5, va='bottom')
        for arm in SIMPLE:
            c = cells.get((src, arm))
            if c:
                _mark(ax, max(c['select_ms'], 1e-2), c['f1']['mean'], MUTED, size=4.5)
        for arm, _, marker in PUBLISHED:
            c = cells.get((src, arm))
            if c:
                _mark(ax, max(c['select_ms'], 1e-2), c['f1']['mean'], INK2, marker=marker, size=5)
        ob = cells.get((src, 'oracle_beta'))
        if ob and src in label_ms:
            # its F1 is over the ORACLE_N labeled test documents only (the caption says so); no line to
            # ours_beta, whose F1 is over every test document -- the paired H1-oracle test is the comparison
            _mark(ax, label_ms[src], ob['f1']['mean'], BLUE, hollow=True)
        for arm, _, color in OURS:
            c = cells.get((src, arm))
            if c:
                _mark(ax, max(c['select_ms'], 1e-2), c['f1']['mean'], color, zorder=4)
        ax.margins(x=0.12, y=0.15)
    axes[0][0].set_ylabel('F1')
    fig.supxlabel('Thời gian nén (ms/tài liệu, thang log)', fontsize=7, color=INK2)
    handles = [Line2D([], [], linestyle='none', marker='o', markersize=5, markerfacecolor=color,
                      markeredgecolor='white', label=label) for _, label, color in OURS]
    handles.append(Line2D([], [], linestyle='none', marker='o', markersize=5, markerfacecolor='white',
                          markeredgecolor=BLUE, markeredgewidth=1.1, label='oracle-β (chưa chưng cất)'))
    handles.append(Line2D([], [], linestyle='none', marker='o', markersize=4.5, markerfacecolor=MUTED,
                          markeredgecolor='white', label='baseline đơn giản'))
    handles += [Line2D([], [], linestyle='none', marker=marker, markersize=5, markerfacecolor=INK2,
                       markeredgecolor='white', label=label) for _, label, marker in PUBLISHED]
    _legend(fig, handles, ncol=6, handletextpad=0.2, columnspacing=1.0)
    return fig


def retention(report, ratio, heldout, plt):
    """Upgrade retention per stable weak->strong reader pair (RQ3 / H3): ours_ens vs ours_beta with 95% CIs;
    pairs with a held-out reader are starred. 1 = the compressor keeps the whole full-context gap."""
    from matplotlib.lines import Line2D
    sources = _present_sources(report)
    rows = [u for u in report.get('upgrade_retention', []) if u.get('stable') and _ratio_match(u['ratio'], ratio)
            and u['arm'] in dict((a, 1) for a, _, _ in OURS)]
    if not rows:
        return None
    gaps = defaultdict(list)
    for u in rows:
        gaps[(u['weak'], u['strong'])].append(u['gap'])
    pairs = sorted(gaps, key=lambda p: -max(gaps[p]))   # largest full-context gap on top
    ypos = {p: i for i, p in enumerate(reversed(pairs))}
    fig, axes = plt.subplots(1, len(sources), figsize=(6.3, 1.0 + 0.22 * len(pairs)), sharey=True, squeeze=False,
                             layout='constrained')
    for ax, (src, name) in zip(axes[0], sources):
        ax.set_title(name, color=INK)
        ax.grid(True, axis='x')
        ax.grid(False, axis='y')
        for ref in (0.0, 1.0):
            ax.axvline(ref, color=AXIS, linewidth=0.8, zorder=1)
        for k, (arm, _, color) in enumerate(OURS):
            for u in rows:
                if u['source'] != src or u['arm'] != arm:
                    continue
                y = ypos[(u['weak'], u['strong'])] + (0.14 if k == 0 else -0.14)
                lo, hi = u['ci95']
                if lo is not None and hi is not None:
                    ax.plot([lo, hi], [y, y], color=color, linewidth=1.0, solid_capstyle='round', zorder=2)
                _mark(ax, u['retention'], y, color, size=5, zorder=3)
        ax.set_ylim(-0.6, len(pairs) - 0.4)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['0', '0.5', '1'])
    labels = [f"{short_reader(w)} → {short_reader(s)}" + (' *' if (w in heldout or s in heldout) else '')
              for (w, s) in reversed(pairs)]
    axes[0][0].set_yticks(range(len(pairs)))
    axes[0][0].set_yticklabels(labels, color=INK2)
    fig.supxlabel('Mức bảo toàn nâng cấp UR (1 = giữ trọn khoảng cách toàn ngữ cảnh)', fontsize=7, color=INK2)
    handles = [Line2D([], [], color=color, marker='o', markersize=5, markeredgecolor='white', linewidth=1.0,
                      label=label) for _, label, color in OURS]
    _legend(fig, handles, ncol=2)
    return fig


def depth(report, reader, ratio, plt):
    """Single-hop: F1 by needle-depth quintile (1 = start of the haystack); the position-bias check."""
    from matplotlib.lines import Line2D
    entries = [e for e in report.get('by_depth', []) if e['reader'] == reader and _ratio_match(e['ratio'], ratio)]
    sources = [(s, n) for s, n in _present_sources(report) if any(e['source'] == s for e in entries)]
    if not sources:
        return None
    fig, axes = plt.subplots(1, len(sources), figsize=(3.03, 1.85), sharey=True, squeeze=False, layout='constrained')
    for ax, (src, name) in zip(axes[0], sources):
        ax.set_title(name, color=INK)
        for arm, _, color, style in DEPTH_ARMS:
            e = next((e for e in entries if e['source'] == src and e['arm'] == arm), None)
            if not e:
                continue
            qs = sorted(e['bins'], key=lambda q: int(q[1:]))
            xs, ys = [int(q[1:]) for q in qs], [e['bins'][q]['f1'] for q in qs]
            ax.plot(xs, ys, color=color, linestyle=style, linewidth=1.2, solid_capstyle='round',
                    solid_joinstyle='round', zorder=3 if arm.startswith('ours') else 2)
            for x, y in zip(xs, ys):
                _mark(ax, x, y, color, size=4, zorder=4 if arm.startswith('ours') else 2)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['1\nđầu', '2', '3', '4', '5\ncuối'])
        ax.set_xlim(0.7, 5.3)
    axes[0][0].set_ylabel('F1')
    fig.supxlabel('Vị trí đoạn kim (ngũ phân vị)', fontsize=7, color=INK2)
    handles = [Line2D([], [], color=color, linestyle=style, linewidth=1.2, marker='o', markersize=4,
                      markeredgecolor='white', label=label) for _, label, color, style in DEPTH_ARMS]
    _legend(fig, handles, ncol=4, handletextpad=0.3, columnspacing=0.8)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', required=True)
    ap.add_argument('--primary-reader', default=None, help="reader tag of the figures' F1 (default: first reader)")
    ap.add_argument('--ratio', type=float, default=4.0)
    ap.add_argument('--heldout-readers', default=None,
                    help="comma list of held-out reader tags to star (default: the report's own)")
    ap.add_argument('--out-dir', default=None, help="default: <report dir>/paper/figures")
    ap.add_argument('--png', action='store_true', help="also write PNG previews")
    args = ap.parse_args()

    with open(args.report, encoding='utf-8') as f:
        report = json.load(f)
    reader = args.primary_reader or report['readers'][0]
    heldout = set((args.heldout_readers if args.heldout_readers is not None
                   else ','.join(report.get('heldout_readers', []))).split(',')) - {''}
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.report)), 'paper', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    plt = setup_matplotlib()
    for name, fig in (('pareto', pareto(report, reader, args.ratio, plt)),
                      ('retention', retention(report, args.ratio, heldout, plt)),
                      ('depth', depth(report, reader, args.ratio, plt))):
        if fig is None:
            print(f"{name}: no data in the report, skipped")
            continue
        fig.savefig(os.path.join(out_dir, f'{name}.pdf'), bbox_inches='tight', pad_inches=0.02)
        if args.png:
            fig.savefig(os.path.join(out_dir, f'{name}.png'), bbox_inches='tight', pad_inches=0.02, dpi=200)
        plt.close(fig)
        print(f"{name}: {os.path.join(out_dir, name + '.pdf')}")


if __name__ == '__main__':
    main()
