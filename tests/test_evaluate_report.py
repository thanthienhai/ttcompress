import json
import subprocess
import sys

import numpy as np

from evaluate import parse_arms


def test_parse_arms_labels_and_prefixes():
    assert parse_arms('lead, beta=pruner:C:/m/x ,pruner:models/y') == [
        ('lead', 'lead'), ('beta', 'pruner:C:/m/x'), ('pruner:models/y', 'pruner:models/y')]


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
    ur = [u for u in report['upgrade_retention'] if u['arm'] == 'ours'][0]
    assert ur['weak'] == 'weak' and ur['strong'] == 'strong'
    assert 0.9 < ur['retention'] < 1.1
    assert (tmp_path / 'report.md').read_text(encoding='utf-8').startswith('# Evaluation report')
