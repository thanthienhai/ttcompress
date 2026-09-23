import os

import pytest
import torch
from transformers import AutoTokenizer

from ttcompress.pruner import ChunkPruner, PrunerScorer, document_scores, effective_max_len, pack_windows
from ttcompress.pruner_training import TrainExample, example_loss, listnet_loss, make_examples, ranking_metrics
from ttcompress.attribution import ChunkLabels
from tests.conftest import TINY_ENCODER

os.environ.setdefault('HF_HUB_OFFLINE', '1')


def _tok():
    return AutoTokenizer.from_pretrained(TINY_ENCODER)


def test_pack_windows_spans_are_exact_and_every_chunk_appears_once():
    tok = _tok()
    chunks = [f'đoạn văn số {i} ' * (3 + i % 5) for i in range(30)]
    windows = pack_windows(tok, 'câu hỏi', chunks, max_len=64)
    assert len(windows) > 1
    seen = [gi for w in windows for gi in w.chunk_ids]
    assert sorted(seen) == list(range(30))
    for w in windows:
        assert len(w.input_ids) <= 64 and len(w.input_ids) == len(w.chunk_slot)
        assert w.input_ids[0] == tok.cls_token_id and w.input_ids[-1] == tok.sep_token_id
        # every slot owns a contiguous, non-empty run of tokens
        for slot in range(len(w.chunk_ids)):
            pos = [i for i, s in enumerate(w.chunk_slot) if s == slot]
            assert pos and pos == list(range(pos[0], pos[-1] + 1))


def test_forward_scores_one_value_per_chunk_and_roundtrips(tmp_path):
    tok = _tok()
    model = ChunkPruner.from_backbone(TINY_ENCODER).eval()
    chunks = ['một hai ba', 'bốn năm', 'sáu bảy tám chín'] * 5
    windows = pack_windows(tok, 'hỏi', chunks, max_len=32)
    with torch.no_grad():
        scores = document_scores(model, windows, len(chunks), tok.pad_token_id, 'cpu')
    assert scores.shape == (len(chunks),)
    model.save_pretrained(str(tmp_path), tok, extra={'max_len': 32})
    scorer = PrunerScorer(str(tmp_path), device='cpu')
    again = scorer.score_chunks('hỏi', chunks)
    assert torch.allclose(scores, torch.tensor(again), atol=1e-5)


def test_training_step_moves_scores_toward_the_label():
    tok = _tok()
    torch.manual_seed(0)
    model = ChunkPruner.from_backbone(TINY_ENCODER)
    chunks = ['alpha beta', 'gamma delta', 'epsilon zeta', 'eta theta']
    ex = TrainExample('d', 'uit_viquad', 'q', chunks, target=[-1.0, 2.0, -0.5, -0.5], gold=[1])
    windows = pack_windows(tok, ex.question, chunks, max_len=64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(30):
        loss = example_loss(document_scores(model, windows, 4, tok.pad_token_id, 'cpu'), ex, 'beta', 1.0, 0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores = document_scores(model.eval(), windows, 4, tok.pad_token_id, 'cpu')
    assert int(scores.argmax()) == 1


def test_listnet_is_minimized_by_matching_order():
    target = torch.tensor([2.0, 0.0, -1.0])
    assert listnet_loss(target * 3, target) < listnet_loss(-target * 3, target)


def _label(z, informative=True, gold=(1,)):
    doc = {'doc_id': 'd', 'source': 's', 'question': 'q', 'chunks': ['a', 'b', 'c'], 'gold_chunks': list(gold)}
    return ChunkLabels(doc=doc, reader='r', target='f1', beta=z, intercept=0.0, z=z, informative=informative,
                       alpha=1.0, cv_r2=0.5, full_f1=1.0)


def test_make_examples_label_sources():
    labs = [_label([0.0, 1.2, -1.2]), _label([0.0, 0.0, 0.0], informative=False)]
    assert len(make_examples(labs, 'beta')) == 1           # uninformative dropped
    span = make_examples(labs, 'span')
    assert len(span) == 2 and span[0].target == [0.0, 1.0, 0.0]
    adj = make_examples(labs, 'beta', position_prior=[1.0] * 10)
    assert adj[0].target == pytest.approx([-1.0, 0.2, -2.2])
    m = ranking_metrics([0.1, 0.9, 0.0], make_examples(labs, 'beta')[0])
    assert m['gold_recall@25%'] == 1.0 and m['ndcg@3_beta'] == 1.0


def test_effective_max_len_respects_roberta_position_offset():
    model = ChunkPruner.from_backbone(TINY_ENCODER)
    limit = effective_max_len(model.encoder.config, 10_000)
    assert limit == 510
    tok = _tok()
    windows = pack_windows(tok, 'q', ['từ ' * 400] * 3, max_len=limit)
    with torch.no_grad():  # the longest window must run without an index error
        document_scores(model.eval(), windows, 3, tok.pad_token_id, 'cpu')
