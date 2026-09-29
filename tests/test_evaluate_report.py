import json
import subprocess
import sys

import numpy as np

from evaluate import parse_arms


def test_parse_arms_labels_and_prefixes():
    assert parse_arms('lead, beta=pruner:C:/m/x ,pruner:models/y') == [
        ('lead', 'lead'), ('beta', 'pruner:C:/m/x'), ('pruner:models/y', 'pruner:models/y')]
    assert parse_arms('reranker=reranker:BAAI/x,reranker:a=b') == [
        ('reranker', 'reranker:BAAI/x'), ('reranker:a=b', 'reranker:a=b')]


def _row(reader, arm, ratio, doc, f1, cluster):
    return {'doc_id': doc, 'arm': arm, 'ratio': ratio, 'source': 'hotpotqa', 'language': 'en', 'hop': 'multi',
            'cluster_id': cluster, 'needle_relpos': None, 'num_chunks': 10, 'full_tokens': 800,
            'kept_tokens': 800 if ratio == 'full' else 200, 'budget': 200, 'gold_recall': 1.0, 'seconds': 0.01,
            'reader': reader, 'answer': '', 'em': float(f1 == 1.0), 'f1': f1, 'answer_recall': f1}


def test_report_paired_diffs_and_upgrade_retention(tmp_path):
    rng = np.random.default_rng(0)
    for reader, full_level in (('weak', 0.4), ('strong', 0.8)):
        rows = []
        for i in range(40):
            doc, cluster = f'd{i}', f'c{i // 2}'
            rows.append(_row(reader, 'full', 'full', doc, full_level, cluster))
            ours = full_level - 0.05 + 0.01 * rng.standard_normal()
            rows.append(_row(reader, 'ours', 4.0, doc, ours, cluster))
            rows.append(_row(reader, 'lead', 4.0, doc, ours - 0.1, cluster))
        with open(tmp_path / f'answers_{reader}_shard0.jsonl', 'w', encoding='utf-8') as f:
            f.writelines(json.dumps(r) + '\n' for r in rows)
    subprocess.run([sys.executable, 'evaluate.py', 'report', '--out-dir', str(tmp_path), '--ours', 'ours',
                    '--n-boot', '300'], check=True, capture_output=True)
    report = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    vs_lead = [p for p in report['paired'] if p['vs'] == 'lead' and p['reader'] == 'strong'][0]
    assert abs(vs_lead['diff'] - 0.1) < 1e-6 and vs_lead['ci95'][0] > 0
    assert all(p['p_holm'] >= p['q_bh'] >= p['p'] for p in report['paired'])
    ur = [u for u in report['upgrade_retention'] if u['arm'] == 'ours'][0]
    assert ur['weak'] == 'weak' and ur['strong'] == 'strong'
    assert 0.9 < ur['retention'] < 1.1
    assert (tmp_path / 'report.md').read_text(encoding='utf-8').startswith('# Evaluation report')


def _hrow(reader, arm, ratio, doc, f1, source, kept=200):
    lang, hop = {'uit_viquad': ('vi', 'single'), 'hotpotqa': ('en', 'multi')}[source]
    return {'doc_id': f'{source}-{doc}', 'arm': arm, 'ratio': ratio, 'source': source, 'language': lang, 'hop': hop,
            'cluster_id': f'{source}-c{doc // 2}', 'needle_relpos': 0.5 if hop == 'single' else None,
            'num_chunks': 10, 'full_tokens': 800, 'kept_tokens': 800 if ratio == 'full' else kept, 'budget': 200,
            'gold_recall': 1.0, 'seconds': 0.01, 'reader': reader, 'answer': '', 'em': 0.0, 'f1': f1,
            'answer_recall': f1, 'n_gold': (1 if doc % 3 == 0 else 2) if hop == 'multi' else 1}


def test_report_hypothesis_families(tmp_path):
    """Synthetic run where every hypothesis holds by construction; each family must say so."""
    rng = np.random.default_rng(1)
    full_level = {'weak': 0.3, 'strong': 0.7, 'heldout': 0.9}
    for reader, lvl in full_level.items():
        rows = []
        for source in ('uit_viquad', 'hotpotqa'):
            for i in range(60):
                e = 0.005 * rng.standard_normal()
                rows.append(_hrow(reader, 'full', 'full', i, lvl, source))
                ens = lvl - 0.1 + e                         # keeps the whole weak->strong gap
                beta = 0.5 * lvl + 0.1 + e                  # keeps half of it
                rows.append(_hrow(reader, 'ours_ens', 4.0, i, ens, source))
                rows.append(_hrow(reader, 'ours_beta', 4.0, i, beta, source))
                rows.append(_hrow(reader, 'bm25', 4.0, i, beta - 0.1, source))
                rows.append(_hrow(reader, 'span_sup', 4.0, i, beta + (0.0 if source == 'uit_viquad' else -0.1), source))
                rows.append(_hrow(reader, 'oracle_span', 4.0, i, beta - 0.1, source))
                if i < 20:                                  # oracle_beta: only the labeled subset
                    rows.append(_hrow(reader, 'oracle_beta', 4.0, i, beta + 0.01, source))
                rows.append(_hrow(reader, 'llmlingua2', 4.0, i, beta - 0.05, source, kept=400))
        with open(tmp_path / f'answers_{reader}_shard0.jsonl', 'w', encoding='utf-8') as f:
            f.writelines(json.dumps(r) + '\n' for r in rows)
    subprocess.run([sys.executable, 'evaluate.py', 'report', '--out-dir', str(tmp_path), '--ours', 'ours_beta,ours_ens',
                    '--primary-reader', 'strong', '--heldout-readers', 'heldout', '--n-boot', '400'],
                   check=True, capture_output=True)
    report = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    fam = {f['name']: f for f in report['hypotheses']}

    # one family over cheap baselines and published compressors, only arms that were run (bm25, llmlingua2)
    assert {t['vs'] for t in fam['H1']['tests']} == {'bm25', 'llmlingua2'}
    assert fam['H1']['n_tests'] == 4 and fam['H1']['n_supported'] == 4          # 2 arms x 2 sources
    assert fam['H1']['n_budget_mismatch'] == 2                                  # llmlingua2 overshoots its budget
    assert fam['H1-oracle']['n_supported'] == 2 and all(t['n'] == 20 for t in fam['H1-oracle']['tests'])
    assert [t['source'] for t in fam['H2a']['tests']] == ['hotpotqa'] * 2 and fam['H2a']['n_supported'] == 2
    assert [t['source'] for t in fam['H2b']['tests']] == ['uit_viquad'] and fam['H2b']['n_supported'] == 1
    # weak->strong and weak->heldout, strong->heldout: two sources each; the held-out pairs are their own family
    assert fam['H3']['n_tests'] == 2 and fam['H3']['n_supported'] == 2
    assert fam['H3']['tests'][0]['diff'] > 0.4
    assert fam['H3-heldout']['n_tests'] == 4
    assert set(fam) == {'H1', 'H1-oracle', 'H2a', 'H2b', 'H3', 'H3-heldout'}
    # exploratory slice: multi-hop documents with >= 2 supporting paragraphs (2 of every 3 here)
    ms = report['h2a_multi_support']
    assert {(x['source'], x['vs']) for x in ms} == {('hotpotqa', 'span_sup'), ('hotpotqa', 'oracle_span')}
    assert all(x['n'] == 40 and x['diff'] > 0 for x in ms)
    over = [c for c in report['cells'] if c['arm'] == 'llmlingua2']
    assert all(c['over_budget_rate'] == 1.0 for c in over)
    assert all('stable' in u and 'gap' in u for u in report['upgrade_retention'])
    md = (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert '## Hypotheses' in md and '### H2b' in md


def test_report_cost_table_from_raw_labels(tmp_path):
    raw = tmp_path / 'labels' / 'raw' / 'Qwen--Qwen3-8B' / 'hotpotqa_train'
    raw.mkdir(parents=True)
    for i in range(3):
        (raw / f'd{i}.json').write_text(json.dumps({'masks': [[True]] * 64, 'seconds': 2.0}), encoding='utf-8')
    (raw / 'measure_config.json').write_text('{}', encoding='utf-8')
    rows = [_hrow('r', arm, ratio, i, 0.5, 'hotpotqa') for i in range(4)
            for arm, ratio in (('full', 'full'), ('ours_beta', 4.0))]
    (tmp_path / 'answers_r_shard0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    subprocess.run([sys.executable, 'evaluate.py', 'report', '--out-dir', str(tmp_path), '--ours', 'ours_beta',
                    '--labels-dir', str(tmp_path / 'labels'), '--n-boot', '100'], check=True, capture_output=True)
    cost = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))['cost']
    assert cost['labels'] == [{'reader': 'Qwen--Qwen3-8B', 'set': 'hotpotqa_train', 'n_docs': 3, 'calls_per_doc': 65.0,
                               'seconds_per_doc': 2.0, 'total_hours': 6.0 / 3600}]
    assert cost['select_ms'][0]['arm'] == 'ours_beta'


def test_ablation_cost_table_uses_its_own_labels_and_unknown_arms_are_listed(tmp_path):
    """An ablation passes its labels and the main run's: a label set in both is the ablation's (one row);
    an arm outside every family (e.g. a published compressor under an unknown label) is named in report.md."""
    for root, secs in (('labels_hard', 5.0), ('labels', 2.0)):
        for split in ('uit_viquad_train', 'hotpotqa_train') if root == 'labels' else ('uit_viquad_train',):
            raw = tmp_path / root / 'raw' / 'Qwen--Qwen3-8B' / split
            raw.mkdir(parents=True)
            (raw / 'd0.json').write_text(json.dumps({'masks': [[True]] * 64, 'seconds': secs}), encoding='utf-8')
    rows = [_hrow('r', arm, ratio, i, 0.5, 'hotpotqa') for i in range(4)
            for arm, ratio in (('full', 'full'), ('ours_beta', 4.0), ('provence_big', 4.0))]
    (tmp_path / 'answers_r_shard0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    subprocess.run([sys.executable, 'evaluate.py', 'report', '--out-dir', str(tmp_path), '--ours', 'ours_beta',
                    '--labels-dir', f"{tmp_path / 'labels_hard'},{tmp_path / 'labels'}", '--n-boot', '100'],
                   check=True, capture_output=True)
    report = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    assert [(c['set'], c['seconds_per_doc']) for c in report['cost']['labels']] == [
        ('hotpotqa_train', 2.0), ('uit_viquad_train', 5.0)]
    assert report['outside_families'] == ['provence_big']
    md = (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert 'Arms in no confirmatory family: provence_big' in md and '| answer recall |' in md


def test_text_arm_is_forced_into_the_budget(fake_tokenizer):
    from evaluate import _text_within_budget
    from tests.conftest import make_doc

    class Overshooting:  # compresses to ~1.5x the asked size, like a rate counted in another tokenizer
        def __init__(self):
            self.asked = []

        def compress_text(self, doc, ratio):
            self.asked.append(ratio)
            return ' '.join(['w'] * int(1.5 * 400 / ratio))

    count = lambda t: len(fake_tokenizer.encode(t))  # noqa: E731
    arm = Overshooting()
    text, truncated = _text_within_budget(arm, make_doc(['x']), 4.0, 100, fake_tokenizer, count)
    assert count(text) <= 100 and not truncated and arm.asked[1] > 4.0
    arm.compress_text = lambda doc, ratio: ' '.join(['w'] * 500)   # ignores the rate entirely
    text, truncated = _text_within_budget(arm, make_doc(['x']), 4.0, 100, fake_tokenizer, count)
    assert count(text) == 100 and truncated


def test_paper_tables_from_report(tmp_path):
    test_report_hypothesis_families(tmp_path)
    subprocess.run([sys.executable, 'scripts/paper_tables.py', '--report', str(tmp_path / 'report.json'),
                    '--primary-reader', 'strong'], check=True, capture_output=True)
    paper = tmp_path / 'paper'
    main = (paper / 'main_results.tex').read_text(encoding='utf-8')
    import re
    bold = [float(v) for v in re.findall(r'\\textbf\{([0-9.]+)\}', main)]
    assert len(bold) == 2 and all(59.5 < v < 60.5 for v in bold)  # ours_ens (0.7 - 0.1) is the best compressor
    assert 'oracle' in main
    assert (paper / 'hypotheses.tex').read_text(encoding='utf-8').count('/') >= 6   # supported/n, one per family
    for name in ('cells.csv', 'paired.csv', 'hypotheses.csv', 'retention.csv', 'by_depth.csv', 'retention.tex'):
        assert (paper / name).stat().st_size > 0


def _seed_run(tmp_path, s2_shift):
    """ours_beta beats bm25 by 0.1 on both sources; seed 1 agrees, seed 2 is shifted by s2_shift on hotpotqa."""
    rows = []
    for source in ('uit_viquad', 'hotpotqa'):
        for i in range(40):
            rows.append(_hrow('r', 'full', 'full', i, 0.8, source))
            rows.append(_hrow('r', 'bm25', 4.0, i, 0.4, source))
            rows.append(_hrow('r', 'ours_beta', 4.0, i, 0.5 + 0.001 * (i % 3), source))
            rows.append(_hrow('r', 'ours_beta_s1', 4.0, i, 0.52, source))
            rows.append(_hrow('r', 'ours_beta_s2', 4.0, i, 0.48 + (s2_shift if source == 'hotpotqa' else 0.0), source))
    (tmp_path / 'answers_r_shard0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    subprocess.run([sys.executable, 'evaluate.py', 'report', '--out-dir', str(tmp_path), '--ours', 'ours_beta',
                    '--primary-reader', 'r', '--n-boot', '300'], check=True, capture_output=True)
    return json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))


def test_report_seed_robustness(tmp_path):
    report = _seed_run(tmp_path, s2_shift=-0.2)          # seed 2 falls below bm25 on hotpotqa
    h1 = next(f for f in report['hypotheses'] if f['name'] == 'H1')
    assert h1['n_supported'] == 2 and h1['n_seed_robust'] == 1
    bad = next(t for t in h1['tests'] if t['source'] == 'hotpotqa')
    assert not bad['seeds_agree'] and abs(bad['seed_diffs']['ours_beta_s2'] + 0.12) < 1e-9
    seeds = {(r['source'], r['arm']): r for r in report['seeds']}
    assert set(seeds[('uit_viquad', 'ours_beta')]['f1_by_seed']) == {'ours_beta', 'ours_beta_s1', 'ours_beta_s2'}
    assert seeds[('hotpotqa', 'ours_beta')]['sd'] > seeds[('uit_viquad', 'ours_beta')]['sd']
    assert 'Training-seed variation' in (tmp_path / 'report.md').read_text(encoding='utf-8')
    # seed arms stay out of the paper's main table; they get their own
    subprocess.run([sys.executable, 'scripts/paper_tables.py', '--report', str(tmp_path / 'report.json'),
                    '--primary-reader', 'r'], check=True, capture_output=True)
    assert r'ours\_beta\_s' not in (tmp_path / 'paper' / 'main_results.tex').read_text(encoding='utf-8')
    assert (tmp_path / 'paper' / 'seeds.tex').read_text(encoding='utf-8').count('(3 seeds)') == 2


def test_compare_runs(tmp_path):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    _seed_run(tmp_path / 'a', 0.0)
    _seed_run(tmp_path / 'b', -0.2)
    out = tmp_path / 'cmp'
    subprocess.run([sys.executable, 'scripts/compare_runs.py', '--primary-reader', 'r', '--out-dir', str(out),
                    '--run', f"random={tmp_path / 'a' / 'report.json'}", '--run', f"hard={tmp_path / 'b' / 'report.json'}"],
                   check=True, capture_output=True)
    md = (out / 'compare.md').read_text(encoding='utf-8')
    assert '| random (n) | hard (n) |' in md and '| H1 | 2/2 | 2/2 |' in md
    rows = (out / 'compare.csv').read_text(encoding='utf-8').splitlines()
    assert rows[0].startswith('source,ratio,arm,random_f1') and any(',bm25,' in r for r in rows)


def test_paper_figures_from_report(tmp_path):
    import pytest
    pytest.importorskip('matplotlib')
    test_report_hypothesis_families(tmp_path)
    out = tmp_path / 'figs'
    res = subprocess.run([sys.executable, 'scripts/paper_figures.py', '--report', str(tmp_path / 'report.json'),
                          '--primary-reader', 'strong', '--out-dir', str(out)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    for name in ('pareto', 'retention', 'depth'):          # the synthetic run has single-hop depth bins too
        assert (out / f'{name}.pdf').stat().st_size > 1000
    report = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    assert report['heldout_readers'] == ['heldout']     # figures star the held-out pairs from the report itself
