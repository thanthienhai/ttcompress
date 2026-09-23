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


def test_select_refuses_a_different_document_set_in_one_out_dir(tmp_path):
    (tmp_path / 'select_config.json').write_text(json.dumps({'num_shards': 8}), encoding='utf-8')
    res = subprocess.run([sys.executable, 'evaluate.py', 'select', '--sources', 'vimqa', '--split', 'test',
                          '--n', '1', '--arms', 'lead', '--budget-tokenizer', TINY_CAUSAL_LM,
                          '--out-dir', str(tmp_path)], capture_output=True, env=ENV, text=True, encoding='utf-8')
    assert res.returncode != 0 and 'use a new --out-dir' in res.stderr
