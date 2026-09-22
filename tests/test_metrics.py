from ttcompress.metrics import bootstrap_mean_ci, compute_token_f1


def test_token_f1_exact_match():
    assert compute_token_f1(["Hà Nội"], ["Hà Nội"]) == 1.0


def test_token_f1_no_overlap():
    assert compute_token_f1(["mèo"], ["chó"]) == 0.0


def test_token_f1_empty_predictions():
    assert compute_token_f1([], []) == 0.0


def test_bootstrap_ci_bounds_the_mean():
    values = [0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.5]
    mean, lo, hi = bootstrap_mean_ci(values, n_boot=500)
    assert lo <= mean <= hi


def test_cluster_bootstrap_runs_with_clusters():
    values = [1.0, 1.0, 0.0, 0.0, 0.5, 0.5]
    clusters = ['a', 'a', 'b', 'b', 'c', 'c']
    mean, lo, hi = bootstrap_mean_ci(values, n_boot=500, clusters=clusters)
    assert lo <= mean <= hi
