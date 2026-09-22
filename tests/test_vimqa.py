from ttcompress.vimqa import _YES_NO_ANSWERS, load_split


def test_load_split_drops_yes_no_answers():
    docs = load_split('validation')
    assert docs
    assert all(d.answer_text not in _YES_NO_ANSWERS for d in docs)


def test_load_split_real_data_shape():
    docs = load_split('validation')
    for d in docs[:20]:
        assert d.num_chunks == 10
        assert all(c.strip() for c in d.chunks)
        assert d.question.strip()
        assert d.answer_text.strip()
        assert 0 <= d.needle_index < d.num_chunks
        assert d.metadata['needle_indices']
        assert all(0 <= i < d.num_chunks for i in d.metadata['needle_indices'])
        assert d.needle_index == min(d.metadata['needle_indices'])
        assert d.metadata['dataset_source'] == 'vimqa'


def test_load_split_doc_ids_unique_and_split_tagged():
    docs = load_split('validation')
    doc_ids = [d.doc_id for d in docs]
    assert len(doc_ids) == len(set(doc_ids))
    assert all(d.doc_id.startswith('vimqa_validation_') for d in docs)


def test_train_validation_test_disjoint_source_ids():
    train = load_split('train')
    val = load_split('validation')
    test = load_split('test')
    train_ids = {d.metadata['source_id'] for d in train}
    val_ids = {d.metadata['source_id'] for d in val}
    test_ids = {d.metadata['source_id'] for d in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
