from ttcompress.uit_viquad import (
    ViquadSample,
    build_paragraph_pool,
    filter_by_excluded_titles,
    load_split,
    split_dev_tune_test,
)


def _sample(title, context, idx=0):
    return ViquadSample(id=f's{idx}', title=title, context=context, question='Q?', answer_text='A', answer_start=0)


def test_filter_by_excluded_titles_drops_matching_case_insensitive():
    samples = [_sample('Hà Nội', 'ctx1', 0), _sample('Paris', 'ctx2', 1), _sample('hà nội', 'ctx3', 2)]
    filtered = filter_by_excluded_titles(samples, ['Hà Nội'])
    assert [s.id for s in filtered] == ['s1']


def test_filter_by_excluded_titles_noop_when_no_match():
    samples = [_sample('Paris', 'ctx', 0)]
    assert filter_by_excluded_titles(samples, ['Hà Nội']) == samples


def test_build_paragraph_pool_dedups_shared_contexts():
    samples = [_sample('T', 'same context', 0), _sample('T', 'same context', 1), _sample('T', 'other', 2)]
    pool = build_paragraph_pool(samples)
    assert sorted(pool) == ['other', 'same context']


def test_official_test_split_has_no_usable_answers():
    # Real finding: all 7301 rows of the official test split have
    # answers=None (withheld, standard for a public leaderboard test set) --
    # load_split must not crash on it and must return nothing usable.
    samples = load_split('test')
    assert samples == []


def test_split_dev_tune_test_disjoint_and_covers_dev():
    tune, test = split_dev_tune_test(num_test=50, seed=1)
    dev_all = load_split('dev')
    assert len(test) == 50
    assert len(tune) + len(test) == len(dev_all)
    assert {s.id for s in tune}.isdisjoint({s.id for s in test})


def test_split_dev_tune_test_deterministic():
    tune1, test1 = split_dev_tune_test(num_test=50, seed=1)
    tune2, test2 = split_dev_tune_test(num_test=50, seed=1)
    assert [s.id for s in test1] == [s.id for s in test2]
    assert [s.id for s in tune1] == [s.id for s in tune2]
