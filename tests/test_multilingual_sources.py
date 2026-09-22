import pytest

from ttcompress.multilingual_sources import ALL_SOURCES, select_documents


def test_select_documents_all_sources_even_split():
    docs = select_documents(ALL_SOURCES, uit_viquad_split='dev', n=21, seed=1)
    assert len(docs) == 21
    from collections import Counter
    counts = Counter(d.metadata.get('dataset_source') for d in docs)
    assert counts == {'uit_viquad': 3, 'xquad_vi': 3, 'vimqa': 3, 'longbench_passage_retrieval_en': 3,
                       'ruler_niah_single_1': 3, 'kamradt_niah': 3, 'infinitebench_passkey': 3}


def test_select_documents_subset_of_sources():
    docs = select_documents(['longbench', 'kamradt'], uit_viquad_split='dev', n=7, seed=1)
    sources = {d.metadata.get('dataset_source') for d in docs}
    assert sources == {'longbench_passage_retrieval_en', 'kamradt_niah'}
    assert len(docs) == 7


def test_select_documents_uit_viquad_only_backward_compatible():
    docs = select_documents(['uit_viquad'], uit_viquad_split='train', n=5, seed=1)
    assert len(docs) == 5
    assert all(d.metadata.get('dataset_source') == 'uit_viquad' for d in docs)


def test_select_documents_unknown_source_raises():
    with pytest.raises(ValueError):
        select_documents(['not_a_source'], uit_viquad_split='dev', n=3, seed=1)


def test_select_documents_every_document_has_valid_needle_index():
    docs = select_documents(ALL_SOURCES, uit_viquad_split='dev', n=8, seed=2)
    for d in docs:
        assert 0 <= d.needle_index < d.num_chunks


def test_select_documents_doc_id_is_a_cluster_label_not_a_unique_id():
    # RULER/Kamradt legitimately reuse one doc_id per essay across several
    # distinct documents (PCS's own bootstrap-CI cluster label) -- this is
    # intentional, not a bug. Code that needs a unique per-document id (e.g.
    # generate_labels.py, which uses doc_id as a filename) must NOT assume
    # ConstructedDocument.doc_id is unique -- see its own disambiguation via
    # the loop index, tested in test_generate_labels.py.
    docs = select_documents(ALL_SOURCES, uit_viquad_split='dev', n=15, seed=1)
    doc_ids = [d.doc_id for d in docs]
    assert len(doc_ids) >= len(set(doc_ids))  # collisions allowed, just documenting the contract


def test_xquad_vi_documents_tagged_correctly():
    docs = select_documents(['xquad_vi'], uit_viquad_split='dev', n=4, seed=1)
    assert len(docs) == 4
    for d in docs:
        assert d.metadata['dataset_source'] == 'xquad_vi'
        assert d.doc_id.startswith('xquadvi_')


def test_vimqa_documents_tagged_correctly():
    docs = select_documents(['vimqa'], uit_viquad_split='dev', n=4, seed=1)
    assert len(docs) == 4
    for d in docs:
        assert d.metadata['dataset_source'] == 'vimqa'
        assert 0 <= d.needle_index < d.num_chunks
        assert all(0 <= i < d.num_chunks for i in d.metadata['needle_indices'])


def test_infinitebench_documents_tagged_correctly():
    docs = select_documents(['infinitebench'], uit_viquad_split='dev', n=4, seed=1)
    assert len(docs) == 4
    for d in docs:
        assert d.metadata['dataset_source'] == 'infinitebench_passkey'
        assert 0 <= d.needle_index < d.num_chunks
