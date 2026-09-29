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


def test_pack_windows_use_the_backbone_pair_frame():
    """RoBERTa / XLM-R cross-encoders read <s> q </s></s> passage </s>: the pruner frames windows the same way;
    pair_format=False keeps the old single-sequence frame for old checkpoints."""
    from ttcompress.pruner import pair_template
    tok = _tok()
    head, middle, tail = pair_template(tok)
    assert head == [tok.cls_token_id] and middle == [tok.sep_token_id] * 2 and tail == [tok.sep_token_id]
    q = tok.encode('hỏi', add_special_tokens=False)
    new = pack_windows(tok, 'hỏi', ['một hai ba'], max_len=64)[0]
    old = pack_windows(tok, 'hỏi', ['một hai ba'], max_len=64, pair_format=False)[0]
    assert list(new.input_ids[:len(q) + 3]) == [tok.cls_token_id] + q + [tok.sep_token_id] * 2
    assert list(old.input_ids[:len(q) + 2]) == [tok.cls_token_id] + q + [tok.sep_token_id]
    assert len(new.input_ids) == len(old.input_ids) + 1 and new.chunk_slot.count(0) == old.chunk_slot.count(0)


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
    model.save_pretrained(str(tmp_path), tok, extra={'max_len': 32, 'pair_format': True})   # as train_pruner.py
    scorer = PrunerScorer(str(tmp_path), device='cpu')
    again = scorer.score_chunks('hỏi', chunks)
    assert torch.allclose(scores, torch.tensor(again), atol=1e-5)
    # a checkpoint without the flag (trained before it existed) is scored in its own, old frame
    model.save_pretrained(str(tmp_path), tok, extra={'max_len': 32})
    old = PrunerScorer(str(tmp_path), device='cpu')
    assert not old.pair_format and len(old.score_chunks('hỏi', chunks)) == len(chunks)


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


def test_listnet_mse_gradient_balance_for_smooth_and_spiky_targets():
    """listnet (sum) + w_mse * mse (mean). For smooth (Gaussian) z-targets the gradient share of mse stays
    ~1 at any C; for spiky targets (one needle, the real single-hop case) it falls with C -- the documented
    behaviour in pruner_training.py, not a balance guarantee."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    ratios = {}
    for C in (2, 10, 40):
        vals = []
        for _ in range(200):
            t = torch.randn(C)
            t = (t - t.mean()) / (t.std() + 1e-6)
            s = (0.1 * torch.randn(C)).requires_grad_()
            g_listnet = torch.autograd.grad(listnet_loss(s, t), s)[0].norm()
            g_mse = torch.autograd.grad(0.5 * F.mse_loss(s, t), s)[0].norm()
            vals.append(float(g_mse / g_listnet))
        ratios[C] = sum(vals) / len(vals)
    assert all(0.7 < r < 1.4 for r in ratios.values()), ratios
    spiky = {}
    for C in (2, 10, 40):
        t = torch.full((C,), -1.0)
        t[0] = 1.0
        t = (t - t.mean()) / (t.std() + 1e-6)
        s = torch.zeros(C, requires_grad=True)
        g_listnet = torch.autograd.grad(listnet_loss(s, t), s)[0].norm()
        g_mse = torch.autograd.grad(0.5 * F.mse_loss(s, t), s)[0].norm()
        spiky[C] = float(g_mse / g_listnet)
    assert spiky[2] > spiky[10] > spiky[40], spiky


def test_ranking_metrics_refuse_non_finite_scores():
    """A NaN model must not score like `lead` (argsort keeps document order) and win model selection."""
    import math
    from ttcompress.pruner_training import TrainExample, ranking_metrics
    ex = TrainExample(doc_id='d', source='hotpotqa', question='q', chunks=['a', 'b', 'c', 'd'], target=[1.0, 0.0, 0.0, 0.0],
                      gold=[0], beta_z=[1.0, 0.0, 0.0, 0.0])
    ok = ranking_metrics([3.0, 1.0, 0.0, -1.0], ex)
    assert ok['ndcg@3_beta'] > 0.99 and ok['gold_recall@25%'] == 1.0
    bad = ranking_metrics([float('nan')] * 4, ex)
    assert all(math.isnan(v) for v in bad.values())


def test_accumulation_groups_average_over_their_own_documents():
    from ttcompress.pruner_training import accumulation_group_size
    sizes = [accumulation_group_size(pos, 19, 8) for pos in range(19)]
    assert sizes == [8] * 16 + [3] * 3            # the short last group divides by 3, not by 8
    assert [accumulation_group_size(p, 16, 8) for p in range(16)] == [8] * 16


def test_compact_windows_score_identically():
    from ttcompress.pruner import compact_windows
    tok = _tok()
    model = ChunkPruner.from_backbone(TINY_ENCODER).eval()
    chunks = ['một hai ba', 'bốn năm', 'sáu bảy tám chín'] * 5
    windows = pack_windows(tok, 'hỏi', chunks, max_len=32)
    with torch.no_grad():
        a = document_scores(model, windows, len(chunks), tok.pad_token_id, 'cpu')
        b = document_scores(model, compact_windows(windows), len(chunks), tok.pad_token_id, 'cpu')
    assert torch.equal(a, b)


def test_pruner_scorer_explains_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match='train stage'):
        PrunerScorer(str(tmp_path / 'models' / 'pruner_beta'), device='cpu')
    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError, match='not a finished pruner checkpoint'):
        PrunerScorer(str(tmp_path / 'empty'), device='cpu')


def test_example_loss_reports_components():
    ex = TrainExample('d', 's', 'q', ['a', 'b', 'c'], target=[1.0, 0.0, -1.0], gold=[0])
    parts = {}
    loss = example_loss(torch.tensor([0.5, 0.0, -0.5]), ex, 'beta', 1.0, 0.5, parts)
    assert set(parts) == {'listnet', 'mse'}
    assert abs(loss.item() - (parts['listnet'] + 0.5 * parts['mse'])) < 1e-6


def _label(z, informative=True, gold=(1,)):
    doc = {'doc_id': 'd', 'source': 's', 'question': 'q', 'chunks': ['a', 'b', 'c'], 'gold_chunks': list(gold)}
    return ChunkLabels(doc=doc, reader='r', target='f1', beta=z, intercept=0.0, z=z, informative=informative,
                       alpha=1.0, cv_r2=0.5, full_f1=1.0)


def test_make_examples_label_sources():
    labs = [_label([0.0, 1.2, -1.2]), _label([0.0, 0.0, 0.0], informative=False)]
    assert len(make_examples(labs, 'beta')) == 1           # uninformative dropped
    span = make_examples(labs, 'span')                     # same documents as beta (RQ2 control)
    assert len(span) == 1 and span[0].target == [0.0, 1.0, 0.0]
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
