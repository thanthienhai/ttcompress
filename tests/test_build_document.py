import random

import pytest

from ttcompress.build_document import (
    NEEDLE_POSITIONS,
    TARGET_CONTEXT_LEN,
    assign_needle_positions,
    build_base_haystack_text,
    build_document,
    build_documents,
    insert_needle,
)
from ttcompress.uit_viquad import ViquadSample


def _fake_pool(n=100, para_len=400):
    return [f"paragraph {i} " + ("filler " * (para_len // 8)) for i in range(n)]


def test_build_base_haystack_excludes_answer_bearing_paragraphs():
    pool = _fake_pool(50)
    tainted = pool[:5]
    for p in tainted:
        pass
    answer = "UNIQUE_ANSWER_TOKEN"
    pool_with_answer = pool + [f"has the {answer} inside it"]
    rng = random.Random(0)
    parts = build_base_haystack_text(pool_with_answer, answer, rng)
    assert all(answer not in p for p in parts)


def test_build_base_haystack_fills_near_budget():
    pool = _fake_pool(200, para_len=500)
    rng = random.Random(1)
    parts = build_base_haystack_text(pool, "no-such-answer", rng)
    total = sum(len(p) for p in parts)
    assert total >= TARGET_CONTEXT_LEN * 0.9  # close to budget, given whole-paragraph increments


def test_build_base_haystack_stops_when_pool_too_small():
    pool = _fake_pool(3, para_len=200)
    rng = random.Random(2)
    parts = build_base_haystack_text(pool, "no-such-answer", rng)
    assert len(parts) == 3  # exhausts the pool, never crashes


@pytest.mark.parametrize('position,expected_idx_fn', [
    ('beginning', lambda n: 0),
    ('middle', lambda n: n // 2),
    ('end', lambda n: n),
])
def test_insert_needle_position(position, expected_idx_fn):
    parts = [f"p{i}" for i in range(10)]
    chunks, idx = insert_needle(parts, "NEEDLE", position)
    assert idx == expected_idx_fn(len(parts))
    assert chunks[idx] == "NEEDLE"
    assert len(chunks) == len(parts) + 1


def test_insert_needle_invalid_position_raises():
    with pytest.raises(ValueError):
        insert_needle(['a'], 'NEEDLE', 'nowhere')


def test_assign_needle_positions_even_split():
    positions = assign_needle_positions(30, seed=0)
    assert len(positions) == 30
    counts = {p: positions.count(p) for p in NEEDLE_POSITIONS}
    assert set(counts.values()) in ({10}, {10, 10})  # 30/3 exact -> all 10
    assert sum(counts.values()) == 30


def test_assign_needle_positions_remainder_distributed():
    positions = assign_needle_positions(10, seed=0)
    counts = sorted(positions.count(p) for p in NEEDLE_POSITIONS)
    assert counts == [3, 3, 4]


def test_assign_needle_positions_empty():
    assert assign_needle_positions(0, seed=0) == []


def _fake_sample(idx=0):
    return ViquadSample(
        id=f"s{idx}", title="T", context=f"NEEDLE CONTEXT {idx} contains ANSWER{idx}.",
        question=f"Q{idx}?", answer_text=f"ANSWER{idx}", answer_start=0,
    )


def test_build_document_needle_present_and_indexed():
    pool = _fake_pool(80)
    sample = _fake_sample(0)
    doc = build_document("doc_0", sample, pool, position='middle', seed=5)
    assert doc.chunks[doc.needle_index] == sample.context
    assert sample.answer_text in doc.chunks[doc.needle_index]
    assert doc.num_chunks == len(doc.chunks)
    assert '\n\n'.join(doc.chunks) == doc.text()


def test_build_documents_one_per_sample_unique_ids():
    pool = _fake_pool(80)
    samples = [_fake_sample(i) for i in range(6)]
    docs = build_documents(samples, pool, seed=1)
    assert len(docs) == 6
    assert len({d.doc_id for d in docs}) == 6
    for d, s in zip(docs, samples):
        assert d.answer_text == s.answer_text
        assert d.question == s.question
