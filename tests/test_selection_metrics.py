import math

import numpy as np

from ttcompress.metrics import (
    answer_recall, bh_adjust, bootstrap_mean_ci, exact_match, gold_chunk_recall, holm_adjust, ndcg_at_k,
    paired_bootstrap_diff, token_f1, upgrade_retention,
)
from ttcompress.selection import Arm, bm25_scores, budget_for, make_arm, select_by_scores
from tests.conftest import TINY_ENCODER, make_doc


# --- metrics ---------------------------------------------------------------

def test_answer_metrics_take_max_over_aliases():
    assert token_f1('Hà Nội', ['Sài Gòn', 'Hà Nội']) == 1.0
    assert exact_match('The Eiffel Tower.', ['eiffel tower']) == 1.0
    assert token_f1('mèo', ['chó']) == 0.0
    assert answer_recall('Đáp án là Hà Nội nhé', ['Hà Nội']) == 1.0


def test_gold_chunk_recall():
    assert gold_chunk_recall([0, 2], [2, 5]) == 0.5
    assert math.isnan(gold_chunk_recall([0], []))


def test_ndcg():
    assert ndcg_at_k([3, 2, 1], [1.0, 0.5, 0.0], 2) == 1.0
    assert ndcg_at_k([1, 2, 3], [1.0, 0.0, 0.0], 1) == 0.0


def test_cluster_bootstrap_and_paired_diff():
    a = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    mean, lo, hi = bootstrap_mean_ci(a, clusters=list('aabbccdd'), n_boot=500)
    assert lo <= mean <= hi
    # b is a shifted copy: marginal CIs overlap, the paired difference is exactly +0.1
    b = [x - 0.1 for x in a]
    res = paired_bootstrap_diff(a, b, n_boot=500)
    assert abs(res['diff'] - 0.1) < 1e-9 and res['ci95'][0] > 0 and res['p'] == 2 / 501  # smoothed floor


def test_multiple_comparison_adjustments():
    raw = [0.01, 0.04, None, 0.03, 0.2]
    # Holm: sorted 0.01,0.03,0.04,0.2 -> x4,x3,x2,x1 with running max
    assert holm_adjust(raw) == [0.04, 0.09, None, 0.09, 0.2]
    # BH: 0.01*4/1, 0.03*4/2, 0.04*4/3, 0.2*4/4 with running min from the top
    q = bh_adjust(raw)
    assert q[2] is None and [round(v, 4) for v in (q[0], q[3], q[1], q[4])] == [0.04, 0.0533, 0.0533, 0.2]
    assert all(h >= b for h, b in zip(holm_adjust(raw), q) if h is not None)


def test_upgrade_retention():
    assert upgrade_retention(0.3, 0.5, 0.4, 0.8) == 0.5
    assert math.isnan(upgrade_retention(0.3, 0.5, 0.4, 0.4))


# --- selection --------------------------------------------------------------

def test_budget_and_greedy_selection_preserve_order(fake_tokenizer):
    doc = make_doc(['a b c', 'd e', 'f g h i', 'j'])
    lengths = [3, 2, 4, 1]
    assert budget_for(10, 4) == 3
    sel = select_by_scores(doc, [0.1, 0.9, 0.5, 0.8], lengths, budget=3, tokenizer=fake_tokenizer)
    assert sel.kept == [1, 3] and sel.kept_tokens == 3 and sel.text == 'd e\n\nj'


def test_selection_truncates_best_chunk_when_nothing_fits(fake_tokenizer):
    doc = make_doc(['a b c d e f'])
    sel = select_by_scores(doc, [1.0], [6], budget=2, tokenizer=fake_tokenizer)
    assert sel.kept == [0] and sel.kept_tokens == 2


def test_bm25_ranks_the_matching_chunk_first():
    scores = bm25_scores('thủ đô của Việt Nam', ['mèo ăn cá', 'Hà Nội là thủ đô của Việt Nam', 'trời mưa'])
    assert int(np.argmax(scores)) == 1


def test_oracle_arms_and_lead_random():
    doc = make_doc(['intro', 'bridge fact', 'answer is Paris', 'noise'], gold=(1, 2), answers=('Paris',), hop='multi')
    span = Arm('oracle_span', 'chunk').scores(doc)
    support = Arm('oracle_support', 'chunk').scores(doc)
    assert int(np.argmax(span)) == 2 and sorted(np.argsort(support)[-2:]) == [1, 2]
    # answer-span oracle ranks the bridge chunk below the answer chunk: the gap RQ2 measures
    assert span[2] > span[1]
    assert Arm('lead', 'chunk').scores(doc) == [0.0, -1.0, -2.0, -3.0]
    assert Arm('random', 'chunk').scores(doc) == Arm('random', 'chunk').scores(doc)
    assert Arm('oracle_beta', 'chunk', table={}).scores(doc) is None


def test_reranker_arm_scores_every_chunk():
    import os
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    arm = make_arm(f'reranker:{TINY_ENCODER}', device='cpu')
    doc = make_doc(['một hai ba', 'bốn năm', 'sáu bảy tám chín'] * 20)
    scores = arm.scores(doc)
    assert len(scores) == 60 and all(isinstance(s, float) for s in scores)
    assert scores[:3] == scores[3:6]  # chunks scored independently of their neighbours


def test_directional_tests():
    from ttcompress.metrics import cluster_bootstrap_column_means, directional_p, paired_bootstrap_test
    boot = np.array([0.03, 0.04, 0.05, 0.06])
    assert directional_p(boot, 'superiority') == 1 / 5
    assert directional_p(boot, 'noninferiority', 0.02) == 1 / 5
    assert directional_p(boot, 'equivalence', 0.05) == 3 / 5          # 0.05 and 0.06 reach +margin
    a = [0.5 + 0.01 * (i % 3) for i in range(40)]
    same = paired_bootstrap_test(a, a, [i // 2 for i in range(40)], 'equivalence', 0.02, n_boot=300)
    assert same['diff'] == 0.0 and same['p'] < 0.01
    worse = paired_bootstrap_test([x - 0.1 for x in a], a, None, 'noninferiority', 0.05, n_boot=300)
    assert worse['p'] > 0.9
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    means = cluster_bootstrap_column_means(arr, ['a', 'a', 'b', 'b'], n_boot=200)
    assert means.shape == (200, 2) and set(np.round(means[:, 0], 6)) <= {2.0, 4.0, 6.0}


def test_oracle_beta_reads_several_label_dirs(tmp_path):
    from ttcompress.attribution import ChunkLabels, save_record
    from ttcompress.selection import load_beta_table
    dirs = []
    for src, beta in (('uit_viquad', [0.1, 0.9]), ('hotpotqa', [0.7, 0.2])):
        d = tmp_path / src
        d.mkdir()
        doc = {'doc_id': f'{src}-1', 'source': src, 'question': 'q', 'chunks': ['a', 'b'], 'gold_chunks': [0]}
        save_record(ChunkLabels(doc=doc, reader='r', target='f1', beta=beta, intercept=0.0, z=beta, informative=True,
                                alpha=1.0, cv_r2=0.5, full_f1=1.0), str(d))
        dirs.append(str(d))
    table = load_beta_table(','.join(dirs))
    assert table == {'uit_viquad-1': [0.1, 0.9], 'hotpotqa-1': [0.7, 0.2]}
