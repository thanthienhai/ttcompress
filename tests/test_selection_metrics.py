import math

import numpy as np

from ttcompress.metrics import (
    answer_recall, bootstrap_mean_ci, exact_match, gold_chunk_recall, ndcg_at_k, paired_bootstrap_diff, token_f1,
    upgrade_retention,
)
from ttcompress.selection import Arm, bm25_scores, budget_for, select_by_scores
from tests.conftest import make_doc


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
    assert abs(res['diff'] - 0.1) < 1e-9 and res['ci95'][0] > 0 and res['p'] == 0.0


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
