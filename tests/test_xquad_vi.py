from ttcompress.xquad_vi import load_all, split_dev_test


def test_load_all_real_data():
    samples = load_all()
    assert len(samples) == 1190
    for s in samples[:5]:
        assert s.context.strip()
        assert s.question.strip()
        assert s.answer_text.strip()
        assert s.context[s.answer_start:s.answer_start + len(s.answer_text)] == s.answer_text


def test_split_dev_test_disjoint():
    dev, test = split_dev_test(num_dev=20, seed=1234)
    assert len(dev) == 20
    assert len(test) == 1170
    assert {s.id for s in dev}.isdisjoint({s.id for s in test})


def test_split_dev_test_deterministic():
    dev1, test1 = split_dev_test(num_dev=20, seed=1234)
    dev2, test2 = split_dev_test(num_dev=20, seed=1234)
    assert [s.id for s in dev1] == [s.id for s in dev2]
    assert [s.id for s in test1] == [s.id for s in test2]
