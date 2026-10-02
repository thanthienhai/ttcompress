"""`--label-source loo_bin`: BCE on a binary leave-one-out label (LooComp/EnComp-style baseline)."""
import sys

import pytest
import torch

from ttcompress.attribution import ChunkLabels, save_record
from ttcompress.pruner_training import BINARY_SOURCES, example_loss, make_examples, ranking_metrics


class _NoGoldDoc(dict):
    """A document whose supporting-fact annotation must never be read."""

    def __getitem__(self, key):
        if key == 'gold_chunks':
            raise AssertionError("loo_bin read gold_chunks")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == 'gold_chunks':
            raise AssertionError("loo_bin read gold_chunks")
        return super().get(key, default)


def _loo(beta, informative=True, gold=(0,), estimator='loo', doc_id='d', doc_cls=dict):
    doc = doc_cls({'doc_id': doc_id, 'source': 's', 'question': 'q', 'chunks': [f'c{i}' for i in range(len(beta))],
                   'gold_chunks': list(gold), 'answers': ['a']})
    z = [float(b) for b in beta]   # z itself is irrelevant to the binary label
    return ChunkLabels(doc=doc, reader='r', target='f1', beta=list(beta), intercept=0.0, z=z,
                       informative=informative, alpha=0.0, cv_r2=float('nan'), full_f1=1.0, estimator=estimator)


def test_loo_bin_is_a_binary_source_with_bce():
    assert 'loo_bin' in BINARY_SOURCES
    ex = make_examples([_loo([0.0, 0.5, -0.2])], 'loo_bin')[0]
    parts = {}
    example_loss(torch.tensor([0.0, 1.0, 0.0]), ex, 'loo_bin', 1.0, 0.5, parts)
    assert set(parts) == {'bce'}


def test_loo_bin_label_is_positive_iff_f1_drop_exceeds_threshold():
    lab = _loo([0.0, 0.5, -0.2, 1.0, 0.25])
    ex = make_examples([lab], 'loo_bin')[0]                    # default threshold 0: strict, a zero drop is negative
    assert ex.target == [0.0, 1.0, 0.0, 1.0, 1.0]
    assert ex.gold == [1, 3, 4]
    ex = make_examples([lab], 'loo_bin', loo_threshold=0.25)[0]  # strict '>' at the threshold too
    assert ex.target == [0.0, 1.0, 0.0, 1.0, 0.0]
    ex = make_examples([lab], 'loo_bin', loo_threshold=-0.5)[0]  # negative threshold admits harmful chunks
    assert ex.target == [1.0] * 5


def test_loo_bin_refuses_ridge_records():
    with pytest.raises(SystemExit, match='estimator'):
        make_examples([_loo([0.0, 0.5]), _loo([0.3, 0.0], estimator='ridge', doc_id='r1')], 'loo_bin')
    with pytest.raises(SystemExit):   # old fit dirs load with the default estimator, which is ridge
        make_examples([ChunkLabels(doc={'doc_id': 'old', 'chunks': ['a']}, reader='r', target='f1', beta=[1.0],
                                   intercept=0.0, z=[0.0], informative=True, alpha=1.0, cv_r2=0.5, full_f1=1.0)],
                      'loo_bin')


def test_loo_bin_drops_uninformative_and_positive_free_documents():
    labs = [_loo([0.0, 0.5, 0.0], doc_id='keep'),
            _loo([0.0, 0.0, 0.0], informative=False, doc_id='flat'),   # all deltas equal: uninformative
            _loo([0.0, -0.3, 0.0], doc_id='nopos'),                     # informative, but no chunk's removal hurts
            _loo([0.1, 0.05, 0.0], doc_id='below')]                     # all drops under the threshold below
    assert [e.doc_id for e in make_examples(labs, 'loo_bin')] == ['keep', 'below']
    assert [e.doc_id for e in make_examples(labs, 'loo_bin', loo_threshold=0.2)] == ['keep']


def test_loo_bin_selection_reads_no_gold_chunks():
    # gold_chunks says chunk 0; the LOO label says chunk 2. Selection must count chunk 2 only.
    lab = _loo([0.0, -0.1, 0.4], gold=(0,), doc_cls=_NoGoldDoc)
    ex = make_examples([lab], 'loo_bin')[0]
    assert ex.gold == [2]
    assert ranking_metrics([0.0, 0.1, 1.0], ex)['gold_recall@25%'] == 1.0
    assert ranking_metrics([1.0, 0.1, 0.0], ex)['gold_recall@25%'] == 0.0   # ranking the gold chunk first earns nothing


def test_train_pruner_cli_refuses_ridge_fit_dirs_before_loading_a_model(tmp_path, monkeypatch):
    import train_pruner
    tr, dv = tmp_path / 'train', tmp_path / 'dev'
    save_record(_loo([0.0, 0.5], estimator='ridge', doc_id='a'), str(tr))
    save_record(_loo([0.0, 0.5], doc_id='b'), str(dv))
    monkeypatch.setattr(sys, 'argv', ['train_pruner.py', '--label-source', 'loo_bin', '--train-labels', str(tr),
                                      '--dev-labels', str(dv), '--loo-threshold', '0.1',
                                      '--out-dir', str(tmp_path / 'out')])
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained',
                        lambda *a, **k: pytest.fail("a model was loaded before the label check"))
    with pytest.raises(SystemExit, match='fit --estimator loo'):
        train_pruner.main()
