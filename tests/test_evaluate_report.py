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
            'answer_recall': f1}


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

    assert fam['H1']['n_tests'] == 2 and fam['H1']['n_supported'] == 2          # vs bm25, 2 sources
    assert fam['H1-oracle']['n_supported'] == 2 and all(t['n'] == 20 for t in fam['H1-oracle']['tests'])
    assert [t['source'] for t in fam['H2a']['tests']] == ['hotpotqa'] * 2 and fam['H2a']['n_supported'] == 2
    assert [t['source'] for t in fam['H2b']['tests']] == ['uit_viquad'] and fam['H2b']['n_supported'] == 1
    # weak->strong and weak->heldout, strong->heldout: two sources each; the held-out pairs are their own family
    assert fam['H3b']['n_tests'] == 2 and fam['H3b']['n_supported'] == 2
    assert fam['H3b']['tests'][0]['diff'] > 0.4
    assert fam['H3b-heldout']['n_tests'] == 4
    h4 = fam['H4']['tests']
    assert {t['source'] for t in h4} == {'uit_viquad'} and all(t['budget_mismatch'] for t in h4)
    assert fam['H4']['n_budget_mismatch'] == 2
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
