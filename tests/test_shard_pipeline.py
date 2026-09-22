import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.shard_pipeline import cells_for_shard, lambdas_for_shard, merge_eval_arms, merge_train_curves
from train import LAMBDA_GRID
from ttcompress.registry import ARM_RATIOS


def test_lambdas_for_shard_partitions_the_full_11_point_grid_without_overlap():
    num_shards = 4
    shards = [lambdas_for_shard(i, num_shards) for i in range(num_shards)]
    all_assigned = [l for shard in shards for l in shard]
    full_grid = [0.0] + LAMBDA_GRID + [1.0]

    assert sorted(all_assigned) == sorted(full_grid)  # every lambda assigned exactly once
    assert len(all_assigned) == len(set(all_assigned))  # no duplicates across shards
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1  # balanced


def test_lambdas_for_shard_more_shards_than_items_gives_some_shards_nothing():
    shards = [lambdas_for_shard(i, 20) for i in range(20)]
    nonempty = [s for s in shards if s]
    assert len(nonempty) == 11  # exactly one lambda per nonempty shard
    assert all(len(s) == 0 for s in shards[11:])


def test_cells_for_shard_default_arms_gives_one_cell_per_gpu_with_4_shards():
    arms = ['encoder_pcs', 'h2o', 'snapkv']
    all_cells = [(arm, ratio) for arm in arms for ratio in ARM_RATIOS[arm]]
    shards = [cells_for_shard(arms, i, 4) for i in range(4)]

    assert sorted(c for s in shards for c in s) == sorted(all_cells)
    assert all(len(s) == 1 for s in shards)  # exactly 4 cells total with default arms -> 1 each


def test_cells_for_shard_no_overlap_and_covers_every_arm_ratio():
    arms = ['encoder_pcs', 'h2o', 'snapkv']
    all_cells = {(arm, ratio) for arm in arms for ratio in ARM_RATIOS[arm]}
    shards = [cells_for_shard(arms, i, 3) for i in range(3)]
    seen = set()
    for s in shards:
        for cell in s:
            assert cell not in seen
            seen.add(cell)
    assert seen == all_cells


def test_merge_train_curves_recovers_the_full_curve_and_picks_the_true_best():
    shard_data = [
        {'curve': {'0.0': {'mean_token_f1': 0.1}, '0.4': {'mean_token_f1': 0.5}, '0.8': {'mean_token_f1': 0.3}},
         'ratios': [4.0, 8.0], 'relevance_source': 'synthetic'},
        {'curve': {'0.1': {'mean_token_f1': 0.2}, '0.5': {'mean_token_f1': 0.9}, '0.9': {'mean_token_f1': 0.4}}},
        {'curve': {'0.2': {'mean_token_f1': 0.35}, '0.6': {'mean_token_f1': 0.15}, '1.0': {'mean_token_f1': 0.05}}},
        {'curve': {'0.3': {'mean_token_f1': 0.25}, '0.7': {'mean_token_f1': 0.6}}},
    ]
    merged = merge_train_curves(shard_data)

    assert set(merged['curve'].keys()) == {str(l) for l in [0.0] + LAMBDA_GRID + [1.0]}
    assert merged['chosen_lambda'] == 0.5  # highest mean_token_f1 (0.9) among the 9 interior points
    assert merged['ratios'] == [4.0, 8.0]
    assert merged['relevance_source'] == 'synthetic'


def test_merge_train_curves_never_picks_an_anchor_as_chosen_lambda():
    # 0.0/1.0 are lambda=0 (pure relevance) / lambda=1 (pure truncation)
    # anchors, included in the curve for reporting but never eligible to be
    # picked as chosen_lambda -- same rule train.py's own single-process
    # sweep() enforces (max() over LAMBDA_GRID, not the anchors).
    shard_data = [{'curve': {'0.0': {'mean_token_f1': 0.99}, '1.0': {'mean_token_f1': 0.98},
                              '0.3': {'mean_token_f1': 0.1}, **{str(round(0.1 * i, 1)): {'mean_token_f1': 0.05}
                                                                 for i in range(1, 10) if i != 3}}}]
    merged = merge_train_curves(shard_data)
    assert merged['chosen_lambda'] == 0.3


def test_merge_train_curves_raises_when_no_interior_point_present():
    with pytest.raises(ValueError):
        merge_train_curves([{'curve': {'0.0': {'mean_token_f1': 0.5}, '1.0': {'mean_token_f1': 0.5}}}])


def test_merge_eval_arms_unions_disjoint_arm_ratio_cells():
    shard_data = [
        {'lam': 0.4, 'arms': {'encoder_pcs': {'4.0x': {'pooled': 1}}}},
        {'lam': 0.4, 'arms': {'encoder_pcs': {'8.0x': {'pooled': 2}}}},
        {'lam': 0.4, 'arms': {'h2o': {'8.0x': {'pooled': 3}}}},
        {'lam': 0.4, 'arms': {'snapkv': {'8.0x': {'pooled': 4}}}},
    ]
    merged = merge_eval_arms(shard_data)

    assert merged['lam'] == 0.4
    assert merged['arms'] == {
        'encoder_pcs': {'4.0x': {'pooled': 1}, '8.0x': {'pooled': 2}},
        'h2o': {'8.0x': {'pooled': 3}},
        'snapkv': {'8.0x': {'pooled': 4}},
    }


def test_merge_eval_arms_rejects_mismatched_lambda():
    shard_data = [{'lam': 0.4, 'arms': {'h2o': {'8.0x': {}}}}, {'lam': 0.9, 'arms': {'snapkv': {'8.0x': {}}}}]
    with pytest.raises(ValueError):
        merge_eval_arms(shard_data)


def test_merge_train_output_is_json_roundtrippable():
    shard_data = [{'curve': {str(round(0.1 * i, 1)): {'mean_token_f1': i / 10} for i in range(0, 11)},
                   'ratios': [4.0], 'relevance_source': 'e6'}]
    merged = merge_train_curves(shard_data)
    reloaded = json.loads(json.dumps(merged))
    assert reloaded == merged
