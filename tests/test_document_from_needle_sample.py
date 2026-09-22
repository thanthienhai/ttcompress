import pytest

from ttcompress.build_document import document_from_needle_sample
from ttcompress.data import NeedleSample
from ttcompress.public_datasets import build_dev_test_split


def test_adapter_on_real_ruler_kamradt_longbench_samples():
    dev, _test = build_dev_test_split(num_dev_essays=3, samples_per_essay=1, longbench_num_dev=3, seed=1)
    for sample in dev:
        doc = document_from_needle_sample(sample)
        assert doc.doc_id == sample.doc_id
        assert doc.chunks == sample.chunks
        assert doc.question == sample.query
        assert doc.answer_text == sample.reference_answer
        assert 0 <= doc.needle_index < doc.num_chunks


def test_adapter_raises_on_missing_chunks():
    sample = NeedleSample(sample_id='s', context='ctx', query='q', reference_answer='a', doc_id='d', chunks=[])
    with pytest.raises(ValueError):
        document_from_needle_sample(sample)


def test_adapter_raises_on_missing_needle_index():
    sample = NeedleSample(sample_id='s', context='a b', query='q', reference_answer='a', doc_id='d',
                           chunks=['a', 'b'], metadata={})
    with pytest.raises(ValueError):
        document_from_needle_sample(sample)
