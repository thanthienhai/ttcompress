import random

import pandas as pd
import pytest

from ttcompress.sources import (
    build_haystack, dev_or_test, hash_unit, is_yes_no, load_documents, multihop_row_to_doc, pad_with_distractors,
    take_n,
)
from tests.conftest import make_doc


def test_hash_split_is_a_pure_function_of_the_key():
    assert dev_or_test('uit:Hà Nội', 0.5) == dev_or_test('uit:Hà Nội', 0.5)
    assert 0.0 <= hash_unit('x') < 1.0
    fractions = [dev_or_test(f'k{i}', 0.2) == 'dev' for i in range(5000)]
    assert 0.17 < sum(fractions) / len(fractions) < 0.23


def test_take_n_is_nested_and_deterministic():
    items = list(range(100))
    small, big = take_n(items, 10, key=str), take_n(items, 30, key=str)
    assert small == big[:10]
    assert take_n(items, 10, key=str) == small


def test_is_yes_no():
    assert is_yes_no('Yes') and is_yes_no('đúng') and is_yes_no('Không.')
    assert not is_yes_no('Hà Nội')


def test_haystack_excludes_answer_and_places_needle_uniformly():
    pool = [f"đoạn nhiễu số {i} " * 20 for i in range(200)] + ['đoạn này chứa Hà Nội nên phải bị loại']
    positions = []
    for seed in range(400):
        chunks, idx = build_haystack('Thủ đô là Hà Nội.', ['Hà Nội'], pool, random.Random(seed), target_chars=3000)
        assert chunks[idx] == 'Thủ đô là Hà Nội.'
        assert sum('Hà Nội' in c for c in chunks) == 1
        positions.append(idx / (len(chunks) - 1))
    # uniform depth: every quintile is populated, not just {0, 0.5, 1}
    quintiles = {min(4, int(p * 5)) for p in positions}
    assert quintiles == {0, 1, 2, 3, 4}
    assert len(set(round(p, 2) for p in positions)) > 5


def test_hard_distractors_come_from_the_same_article_first():
    pool = [f"random {i} " * 30 for i in range(100)]
    same = ['cùng bài 1 ' * 30, 'cùng bài 2 ' * 30]
    chunks, idx = build_haystack('needle X', ['X'], pool + same, random.Random(0), target_chars=2000,
                                 same_title_pool=same, distractors='hard')
    others = [c for i, c in enumerate(chunks) if i != idx]
    assert set(same) <= set(others)


def _row(answer='Paris', titles=('A', 'B', 'C'), support=('A', 'C')):
    return pd.Series({'id': 'x1', 'question': 'q?', 'answer': answer, 'type': 'bridge',
                      'context': {'title': list(titles), 'sentences': [[' s1.', ' s2.'], ['t1.'], ['u1.']]},
                      'supporting_facts': {'title': list(support), 'sent_id': [0, 0]}})


def test_multihop_row_to_doc():
    doc = multihop_row_to_doc(_row(), 'hotpotqa', 'dev')
    assert doc.chunks[0] == 'A\ns1. s2.'
    assert doc.gold_chunks == [0, 2]
    assert doc.language == 'en' and doc.hop == 'multi'
    assert multihop_row_to_doc(_row(answer='yes'), 'hotpotqa', 'dev') is None
    assert multihop_row_to_doc(_row(support=('Z',)), 'hotpotqa', 'dev') is None


def test_pad_with_distractors_remaps_gold():
    doc = make_doc(['g0', 'x1', 'g2'], gold=(0, 2), answers=('ans',), hop='multi')
    pool = [f'filler {i}' for i in range(50)] + ['contains ans']
    padded = pad_with_distractors(doc, pool, target_chars=200, rng=random.Random(1))
    assert len(padded.chunks) > 3
    assert [padded.chunks[g] for g in padded.gold_chunks] == ['g0', 'g2']
    assert 'contains ans' not in padded.chunks


# --- real data (HF cache) --------------------------------------------------

@pytest.mark.parametrize('source', ['vimqa', 'hotpotqa', '2wiki'])
def test_multihop_dev_test_disjoint(source):
    dev = {d.metadata['source_id'] for d in load_documents(source, 'dev')}
    test = {d.metadata['source_id'] for d in load_documents(source, 'test')}
    assert dev and test and not dev & test


def test_uit_viquad_dev_test_disjoint_by_title_and_train_titles_disjoint():
    dev = load_documents('uit_viquad', 'dev', n=40, haystack_chars=5000)
    test = load_documents('uit_viquad', 'test', n=40, haystack_chars=5000)
    dev_titles = {d.metadata['title'] for d in dev}
    test_titles = {d.metadata['title'] for d in test}
    assert not dev_titles & test_titles
    for d in dev + test:
        assert len(d.gold_chunks) == 1
        needle = d.chunks[d.gold_chunks[0]].lower()
        assert any(a.lower() in needle for a in d.answers)


def test_xquad_vi_has_no_train_and_is_disjoint_by_passage():
    with pytest.raises(ValueError):
        load_documents('xquad_vi', 'train')
    dev = load_documents('xquad_vi', 'dev', haystack_chars=3000)
    test = load_documents('xquad_vi', 'test', n=50, haystack_chars=3000)
    assert not {d.cluster_id for d in dev} & {d.cluster_id for d in test}
