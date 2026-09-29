"""Failure modes that only show up on a multi-process cluster run: resumes after a
crash, a different process count between stages, settings drifting between runs."""
import json
import os
import subprocess
import sys

import pytest

from evaluate import _read_jsonl
from ttcompress.reader import fit_ids
from tests.conftest import TINY_CAUSAL_LM, make_doc

ENV = {**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONIOENCODING': 'utf-8'}


def test_fit_ids_cuts_the_middle_and_keeps_both_ends():
    ids = list(range(100))
    assert fit_ids(ids, None) == (ids, False)
    assert fit_ids(ids, 200) == (ids, False)
    out, truncated = fit_ids(ids, 10)
    assert truncated and out == [0, 1, 2, 3, 4, 95, 96, 97, 98, 99]


def test_torn_last_jsonl_line_is_dropped_but_a_torn_middle_line_raises(tmp_path):
    p = tmp_path / 'x.jsonl'
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": ', encoding='utf-8')
    assert _read_jsonl(str(p)) == [{'a': 1}, {'a': 2}]
    p.write_text('{"a": 1}\n{"a": \n{"a": 3}\n', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError):
        _read_jsonl(str(p))


def _write_selection_shards(out_dir, n_shards=4):
    for k in range(n_shards):
        doc = make_doc([f'chunk {k} a', f'chunk {k} b'], doc_id=f'doc{k}', answers=('a',), hop='single')
        with open(out_dir / f'documents_shard{k}.jsonl', 'w', encoding='utf-8') as f:
            f.write(json.dumps(doc.to_dict()) + '\n')
        row = {'doc_id': doc.doc_id, 'arm': 'lead', 'ratio': 4.0, 'source': doc.source, 'language': 'vi',
               'hop': 'single', 'cluster_id': doc.doc_id, 'needle_relpos': None, 'num_chunks': 2, 'full_tokens': 6,
               'kept': [0], 'text': doc.chunks[0], 'kept_tokens': 3, 'budget': 3, 'truncated': False,
               'gold_recall': 1.0, 'seconds': 0.0}
        with open(out_dir / f'selections_shard{k}.jsonl', 'w', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')


def test_answer_covers_every_selection_shard_with_fewer_processes(tmp_path):
    """select ran with 4 shards; a TP=2 reader answers with 2 processes."""
    _write_selection_shards(tmp_path)
    for shard in (0, 1):
        subprocess.run([sys.executable, 'evaluate.py', 'answer', '--out-dir', str(tmp_path),
                        '--reader-model', TINY_CAUSAL_LM, '--device', 'cpu', '--shard', str(shard),
                        '--num-shards', '2'], check=True, capture_output=True, env=ENV)
    tag = TINY_CAUSAL_LM.replace('/', '--')
    answered = sorted(p.name for p in tmp_path.glob(f'answers_{tag}_shard*.jsonl'))
    assert answered == [f'answers_{tag}_shard{k}.jsonl' for k in range(4)]
    # rerun = nothing left to do, no duplicate rows
    subprocess.run([sys.executable, 'evaluate.py', 'answer', '--out-dir', str(tmp_path), '--reader-model',
                    TINY_CAUSAL_LM, '--device', 'cpu'], check=True, capture_output=True, env=ENV)
    for k in range(4):
        assert len(_read_jsonl(str(tmp_path / f'answers_{tag}_shard{k}.jsonl'))) == 1


def test_measure_refuses_to_mix_settings_in_one_label_dir(tmp_path):
    out_dir = tmp_path / 'raw' / TINY_CAUSAL_LM.replace('/', '--') / 'vimqa_dev'
    out_dir.mkdir(parents=True)
    (out_dir / 'measure_config.json').write_text(json.dumps({'keep_rates': [0.9]}), encoding='utf-8')
    res = subprocess.run([sys.executable, 'generate_labels.py', 'measure', '--source', 'vimqa', '--split', 'dev',
                          '--n', '1', '--reader-model', TINY_CAUSAL_LM, '--device', 'cpu',
                          '--out-root', str(tmp_path / 'raw')], capture_output=True, env=ENV, text=True,
                         encoding='utf-8')
    assert res.returncode != 0 and 'use another --out-root' in res.stderr


def test_empty_measure_shard_records_the_sources_answer_budget(tmp_path):
    """n < num_shards leaves a shard with no documents; it must record the multi-hop budget (48), not the
    single-hop default, or the other shards of the same label dir refuse its measure_config.json. The shard
    that measures appends its run settings to measure_runs.jsonl (provenance, never compared)."""
    def measure(shard):
        subprocess.run([sys.executable, 'generate_labels.py', 'measure', '--source', 'vimqa', '--split', 'dev',
                        '--n', '1', '--shard', str(shard), '--num-shards', '2', '--reader-model', TINY_CAUSAL_LM,
                        '--device', 'cpu', '--backend', 'hf', '--k-min', '4', '--k-max', '4',
                        '--max-model-len', '512',   # the tiny test model's context
                        '--out-root', str(tmp_path)], check=True, capture_output=True, env=ENV)
    out = tmp_path / TINY_CAUSAL_LM.replace('/', '--') / 'vimqa_dev'
    measure(1)
    assert json.loads((out / 'measure_config.json').read_text(encoding='utf-8'))['max_new_tokens'] == 48
    assert not (out / 'measure_runs.jsonl').exists()   # nothing measured, nothing to account for
    measure(0)
    (run,) = [json.loads(line) for line in (out / 'measure_runs.jsonl').read_text(encoding='utf-8').splitlines()]
    assert run['shard'] == 0 and run['num_shards'] == 2 and run['tp'] == 1 and run['n_todo'] == 1
    assert len([p for p in out.glob('*.json') if p.name != 'measure_config.json']) == 1


def test_select_refuses_a_different_document_set_in_one_out_dir(tmp_path):
    (tmp_path / 'select_config.json').write_text(json.dumps({'num_shards': 8}), encoding='utf-8')
    res = subprocess.run([sys.executable, 'evaluate.py', 'select', '--sources', 'vimqa', '--split', 'test',
                          '--n', '1', '--arms', 'lead', '--budget-tokenizer', TINY_CAUSAL_LM,
                          '--out-dir', str(tmp_path)], capture_output=True, env=ENV, text=True, encoding='utf-8')
    assert res.returncode != 0 and 'use a new --out-dir' in res.stderr


def test_append_after_a_torn_line_repairs_the_file(tmp_path):
    from evaluate import _append_jsonl
    p = tmp_path / 'x.jsonl'
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": ', encoding='utf-8')     # killed mid-append
    _append_jsonl(str(p), [{'a': 3}])
    assert _read_jsonl(str(p)) == [{'a': 1}, {'a': 2}, {'a': 3}]  # no glued line, no crash on the next read
    _append_jsonl(str(p), [{'a': 4}])
    assert [r['a'] for r in _read_jsonl(str(p))] == [1, 2, 3, 4]


def _select_args(out_dir, arms, ratios):
    import argparse
    return argparse.Namespace(split='test', budget_tokenizer=TINY_CAUSAL_LM, sources='hotpotqa', n=1, num_shards=1,
                              shard=0, haystack_chars=30000, distractors='random', multihop_pad_chars=0, arms=arms,
                              ratios=ratios, oracle_beta_dir=None, device='cpu', out_dir=str(out_dir), flush_every=25,
                              retry_failed=False)


def _written_select_dir(tmp_path):
    """A select dir whose documents were already written (select reuses them: no dataset download)."""
    doc = make_doc(['Paris is in France.', 'Berlin is in Germany.', 'Rome is in Italy.'], doc_id='d0',
                   source='hotpotqa', hop='multi')
    meta = {'split': 'test', 'budget_tokenizer': TINY_CAUSAL_LM, 'sources': 'hotpotqa', 'n': 1, 'num_shards': 1,
            'haystack_chars': 30000, 'distractors': 'random', 'multihop_pad_chars': 0}
    (tmp_path / 'select_config.json').write_text(json.dumps(meta), encoding='utf-8')
    (tmp_path / 'documents_shard0.jsonl').write_text(json.dumps(doc.to_dict()) + '\n', encoding='utf-8')


def test_reselect_with_an_extra_ratio_adds_only_the_missing_rows(tmp_path):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from evaluate import cmd_select
    _written_select_dir(tmp_path)
    cmd_select(_select_args(tmp_path, 'lead', '4'))
    cmd_select(_select_args(tmp_path, 'lead', '4,8'))
    keys = [(r['doc_id'], r['arm'], r['ratio']) for r in _read_jsonl(str(tmp_path / 'selections_shard0.jsonl'))]
    assert sorted(keys) == [('d0', 'lead', 4.0), ('d0', 'lead', 8.0)]


def test_a_failing_published_compressor_is_recorded_not_fatal(tmp_path, monkeypatch):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import evaluate
    from ttcompress.selection import Arm

    class Broken:
        def compress(self, doc, ratio):
            raise RuntimeError('compressor exploded')

    real = evaluate.make_arm
    monkeypatch.setattr(evaluate, 'make_arm', lambda spec, *a: Arm(spec, 'text', text_compressor=Broken())
                        if spec == 'llmlingua2' else real(spec, *a))
    _written_select_dir(tmp_path)
    evaluate.cmd_select(_select_args(tmp_path, 'lead,llmlingua2', '4'))
    rows = _read_jsonl(str(tmp_path / 'selections_shard0.jsonl'))
    assert [r['arm'] for r in rows] == ['lead']
    fails = _read_jsonl(str(tmp_path / 'select_failures_shard0.jsonl'))
    assert fails[0]['arm'] == 'llmlingua2' and 'exploded' in fails[0]['error']
    evaluate.cmd_select(_select_args(tmp_path, 'lead,llmlingua2', '4'))      # resume: not retried by default
    assert len(_read_jsonl(str(tmp_path / 'select_failures_shard0.jsonl'))) == 1
    with pytest.raises(RuntimeError):                                        # our own arms still fail loudly
        monkeypatch.setattr(evaluate, 'make_arm', lambda spec, *a: Arm(spec, 'text', text_compressor=Broken()))
        evaluate.cmd_select(_select_args(tmp_path, 'mine=lead', '8'))
