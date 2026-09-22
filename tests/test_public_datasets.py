"""Dev/test must never share a source document (PCS_METHOD_SPEC.md §7:
sweeping lambda against contaminated data was a named past failure mode)."""
from ttcompress.public_datasets import _longbench_chunks, build_dev_test_split, split_infinitebench_passkey


def test_dev_test_share_no_documents():
    dev, test = build_dev_test_split(num_dev_essays=5, samples_per_essay=1, longbench_num_dev=5, seed=1)
    dev_docs = {s.doc_id for s in dev}
    test_docs = {s.doc_id for s in test}
    assert dev_docs.isdisjoint(test_docs)


def test_all_four_sources_present_in_both_splits():
    dev, test = build_dev_test_split(
        num_dev_essays=5, samples_per_essay=2, longbench_num_dev=5, infinitebench_num_dev=5, seed=1)
    expected = {'longbench_passage_retrieval_en', 'ruler_niah_single_1', 'kamradt_niah', 'infinitebench_passkey'}
    assert {s.metadata['dataset_source'] for s in dev} == expected
    assert {s.metadata['dataset_source'] for s in test} == expected


def test_samples_have_nonempty_context_and_answer():
    dev, _test = build_dev_test_split(num_dev_essays=3, samples_per_essay=1, longbench_num_dev=3, seed=2)
    for s in dev:
        assert s.context.strip()
        assert s.query.strip()
        assert s.reference_answer.strip()
        assert s.doc_id


def test_every_sample_has_chunks_populated():
    # Merged multilingual training (OUTCOME_SUPERVISED_RELEVANCE_SPEC.md
    # upgrade) needs chunk structure from every source, not just UIT-ViQuAD.
    dev, test = build_dev_test_split(num_dev_essays=3, samples_per_essay=2, longbench_num_dev=5, seed=1)
    for s in dev + test:
        assert s.chunks, s.sample_id
        assert all(c.strip() for c in s.chunks)


def test_ruler_kamradt_chunks_join_back_to_context():
    dev, _test = build_dev_test_split(num_dev_essays=3, samples_per_essay=2, longbench_num_dev=3, seed=1)
    for s in dev:
        if s.metadata['dataset_source'] == 'longbench_passage_retrieval_en':
            continue
        assert ' '.join(s.chunks) == s.context


def test_longbench_chunks_splitter():
    ctx = "Paragraph 1: alpha content here. Paragraph 2: beta content here. Paragraph 3: gamma."
    chunks = _longbench_chunks(ctx)
    assert chunks == [
        "Paragraph 1: alpha content here.",
        "Paragraph 2: beta content here.",
        "Paragraph 3: gamma.",
    ]


def test_longbench_real_data_has_30_paragraphs():
    dev, test = build_dev_test_split(num_dev_essays=1, samples_per_essay=1, longbench_num_dev=3, seed=1)
    lb = [s for s in dev + test if s.metadata['dataset_source'] == 'longbench_passage_retrieval_en']
    assert lb
    for s in lb:
        assert len(s.chunks) == 30
        assert all(c.startswith(f"Paragraph {i + 1}:") for i, c in enumerate(s.chunks))


def test_infinitebench_passkey_dev_test_disjoint():
    dev, test = split_infinitebench_passkey(num_dev=10, seed=5)
    dev_ids = {s.sample_id for s in dev}
    test_ids = {s.sample_id for s in test}
    assert len(dev) == 10
    assert dev_ids.isdisjoint(test_ids)


def test_infinitebench_passkey_needle_index_matches_reference_answer():
    dev, _test = split_infinitebench_passkey(num_dev=5, seed=5)
    for s in dev:
        idx = s.metadata['needle_index']
        assert 0 <= idx < len(s.chunks)
        assert s.reference_answer in s.chunks[idx]
        assert s.chunks[idx].startswith("The pass key is")
