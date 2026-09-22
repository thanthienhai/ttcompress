import tempfile

import numpy as np

from ttcompress.outcome_labels import DocumentLabels
from ttcompress.position_decompose import (
    decompose_document,
    load_training_labels,
    save_training_labels,
)


def test_decompose_document_produces_normalized_labels():
    L = 20
    from ttcompress.pcs import g
    positions = list(range(L))
    beta_c = [0.4 + 0.3 * g(i, L) for i in positions]  # pure position signal, no content
    labels = DocumentLabels(
        doc_id='doc_x', chunks=[f'c{i}' for i in range(L)], num_chunks=L, positions=positions,
        beta=[0.1] + beta_c, ci_lo=[0.0] * (L + 1), ci_hi=[1.0] * (L + 1),
        low_confidence=[False] * L, alpha=1.0, metadata={},
    )
    training = decompose_document(labels)
    assert training.doc_id == 'doc_x'
    assert len(training.raw_label) == L
    assert len(training.residual_label) == L
    # pure position signal -> residual should be ~zero after normalization noise floor
    assert np.allclose(training.residual_label, 0.0, atol=1e-6)
    # raw_label normalized: mean ~0, std ~1
    assert abs(np.mean(training.raw_label)) < 1e-6
    assert abs(np.std(training.raw_label) - 1.0) < 1e-6


def test_training_labels_roundtrip():
    from ttcompress.position_decompose import TrainingLabels
    labels = TrainingLabels(
        doc_id='doc_y', chunks=['a', 'b'], positions=[0, 1],
        raw_label=[0.5, -0.5], residual_label=[0.1, -0.1],
        gamma_0=0.3, gamma_1=0.2, low_confidence=[False, True], metadata={'k': 'v'},
    )
    with tempfile.TemporaryDirectory() as d:
        path = save_training_labels(labels, d)
        loaded = load_training_labels(path)
    assert loaded == labels
