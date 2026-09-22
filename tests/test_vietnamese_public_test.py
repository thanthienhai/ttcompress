from ttcompress.vietnamese_public_test import load


def test_load_returns_requested_count_split_across_all_three_sources():
    samples = load(n=21, seed=1)
    assert len(samples) == 21
    from collections import Counter
    counts = Counter(s.metadata['dataset_source'] for s in samples)
    assert counts == {'uit_viquad': 7, 'xquad_vi': 7, 'vimqa': 7}


def test_load_doc_ids_unique():
    samples = load(n=20, seed=1)
    assert len({s.doc_id for s in samples}) == 20


def test_load_disjoint_from_tuning_pool():
    # The reserved uit_viquad test portion must never overlap the tuning
    # portion multilingual_sources.py's sweep draws from.
    from ttcompress.uit_viquad import split_dev_tune_test
    tune, _test = split_dev_tune_test(seed=1)
    tune_ids = {s.id for s in tune}

    samples = load(n=20, seed=1)
    uit_samples = [s for s in samples if s.metadata['dataset_source'] == 'uit_viquad']
    assert all(s.metadata['source_id'] not in tune_ids for s in uit_samples)


def test_vimqa_test_portion_disjoint_from_its_own_train_and_validation():
    # VIMQA's official 'test' split is reserved for this pool exclusively --
    # multilingual_sources.py's sweep only ever draws from 'train'/'validation'.
    from ttcompress.vimqa import load_split
    train_ids = {d.metadata['source_id'] for d in load_split('train')}
    val_ids = {d.metadata['source_id'] for d in load_split('validation')}

    samples = load(n=20, seed=1)
    vimqa_samples = [s for s in samples if s.metadata['dataset_source'] == 'vimqa']
    assert vimqa_samples
    vimqa_test_ids = {s.metadata['source_id'] for s in vimqa_samples}
    assert vimqa_test_ids.isdisjoint(train_ids)
    assert vimqa_test_ids.isdisjoint(val_ids)
