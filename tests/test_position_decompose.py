import numpy as np

from ttcompress.pcs import g
from ttcompress.position_decompose import decompose, position_scores_for_chunks


def test_position_scores_match_pcs_g():
    L = 12
    positions = list(range(L))
    scores = position_scores_for_chunks(positions, L)
    assert scores == [g(i, L) for i in positions]


def test_decompose_recovers_exact_linear_relationship():
    L = 15
    positions = list(range(L))
    gamma0_true, gamma1_true = 0.4, 0.35
    beta_c = [gamma0_true + gamma1_true * g(i, L) for i in positions]
    gamma0, gamma1, residual = decompose(beta_c, positions, L)
    np.testing.assert_allclose(gamma0, gamma0_true, atol=1e-9)
    np.testing.assert_allclose(gamma1, gamma1_true, atol=1e-9)
    np.testing.assert_allclose(residual, np.zeros(L), atol=1e-9)


def test_decompose_residual_nonzero_under_content_signal():
    # OLS's own gamma0/gamma1 estimates are themselves perturbed by the added
    # noise, so the residual isn't the injected content_signal exactly (that
    # would only hold if content_signal were orthogonal to the design matrix
    # by construction) -- this checks the decomposition still recovers
    # gamma1 close to the true slope and leaves a genuinely nonzero residual,
    # not that it reconstructs the noise verbatim.
    L = 200
    positions = list(range(L))
    rng = np.random.default_rng(0)
    content_signal = rng.normal(0, 0.2, size=L)
    beta_c = np.array([0.4 + 0.3 * g(i, L) for i in positions]) + content_signal
    gamma0, gamma1, residual = decompose(beta_c.tolist(), positions, L)
    assert abs(gamma1 - 0.3) < 0.05
    assert np.std(residual) > 0.1
