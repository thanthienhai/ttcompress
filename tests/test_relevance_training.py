from ttcompress.relevance_training import (
    CHUNK_SEP,
    assign_token_labels,
    build_training_windows,
    char_spans_for_chunks,
    collate_windows,
    window_sequence,
)


def test_char_spans_for_chunks_basic():
    chunks = ['ab', 'cde', 'f']
    spans = char_spans_for_chunks(chunks)
    joined = CHUNK_SEP.join(chunks)
    for (s, e), chunk in zip(spans, chunks):
        assert joined[s:e] == chunk


def test_assign_token_labels_maps_to_correct_chunk():
    chunks = ['aaaa', 'bbbb', 'cccc']  # spans: (0,4), (6,10), (12,16) with '\n\n' seps
    spans = char_spans_for_chunks(chunks)
    labels = [10.0, 20.0, 30.0]
    # one token per char position 0..15, plus one deliberately in the gap (4,6)
    offsets = [(i, i + 1) for i in range(16)]
    ids = list(range(16))
    token_labels, valid = assign_token_labels(ids, offsets, spans, labels)
    assert token_labels[0] == 10.0 and valid[0]
    assert token_labels[3] == 10.0 and valid[3]
    assert token_labels[6] == 20.0 and valid[6]
    assert token_labels[15] == 30.0 and valid[15]
    # position 4 and 5 fall in the '\n\n' gap between chunk 0 and chunk 1
    assert valid[4] is False
    assert valid[5] is False


def test_assign_token_labels_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        assign_token_labels([1], [(0, 1)], [(0, 1), (2, 3)], [1.0])


def test_window_sequence_splits_into_fixed_size_chunks():
    ids = list(range(10))
    labels = [float(i) for i in range(10)]
    valid = [True] * 10
    windows = window_sequence(ids, labels, valid, max_len=4)
    assert [len(w.input_ids) for w in windows] == [4, 4, 2]
    assert windows[0].input_ids == [0, 1, 2, 3]
    assert windows[-1].input_ids == [8, 9]


def test_build_training_windows_end_to_end():
    chunks = ['aa', 'bb']
    labels = [1.0, 2.0]
    text = CHUNK_SEP.join(chunks)
    ids = list(range(len(text)))
    offsets = [(i, i + 1) for i in range(len(text))]
    windows = build_training_windows(ids, offsets, chunks, labels, max_len=100)
    assert len(windows) == 1
    w = windows[0]
    # 'aa' at 0-2, sep at 2-4, 'bb' at 4-6
    assert w.labels[0] == 1.0 and w.valid_mask[0]
    assert w.valid_mask[2] is False  # separator
    assert w.labels[4] == 2.0 and w.valid_mask[4]


def test_collate_windows_pads_to_batch_max():
    from ttcompress.relevance_training import TokenLabelSequence
    w1 = TokenLabelSequence(input_ids=[1, 2, 3], labels=[0.1, 0.2, 0.3], valid_mask=[True, True, True])
    w2 = TokenLabelSequence(input_ids=[4, 5], labels=[0.4, 0.5], valid_mask=[True, False])
    input_ids, attn, labels, valid = collate_windows([w1, w2], pad_id=0)
    assert input_ids == [[1, 2, 3], [4, 5, 0]]
    assert attn == [[1, 1, 1], [1, 1, 0]]
    assert labels == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]]
    assert valid == [[True, True, True], [True, False, False]]
