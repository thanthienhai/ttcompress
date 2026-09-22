"""E6RelevanceProvider (classification) and RegressionRelevanceProvider
(Stage C, OUTCOME_SUPERVISED_RELEVANCE_SPEC.md) share a base class
(ttcompress.relevance._BaseEncoderRelevanceProvider) that only differs in how
one encoder window's logits become a per-token score. These tests exercise
that difference with a fake encoder (no real HF checkpoint needed) so the
refactor that introduced the shared base is locked in."""
import torch

from ttcompress.relevance import E6RelevanceProvider, RegressionRelevanceProvider


class FakeGenTokenizer:
    """Generation-side tokenizer: one token per character for simplicity."""

    def decode(self, ids, clean_up_tokenization_spaces=False, skip_special_tokens=True):
        return ''.join(chr(i) for i in ids)


class FakeEncTokenizer:
    """Encoder-side tokenizer: also one token per character, offsets 1:1."""

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False, truncation=False):
        ids = [ord(c) for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return {'input_ids': ids, 'offset_mapping': offsets}


class FakeLogitsOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeClassifierEncoder:
    """2-class classifier: logits = [drop_logit, keep_logit]. keep_logit is
    high (so softmax favors 'keep') for even-valued token ids, low otherwise."""

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, tensor):
        ids = tensor[0]
        keep_logit = torch.where(ids % 2 == 0, torch.tensor(5.0), torch.tensor(-5.0))
        drop_logit = -keep_logit
        logits = torch.stack([drop_logit, keep_logit], dim=-1).unsqueeze(0)
        return FakeLogitsOutput(logits)


class FakeRegressionEncoder:
    """1-unit regression head: raw logit = (token id - 100) / 10, so
    sigmoid(logit) varies smoothly with the character's ordinal value."""

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, tensor):
        ids = tensor[0].float()
        raw = (ids - 100.0) / 10.0
        logits = raw.unsqueeze(-1).unsqueeze(0)  # [1, S, 1]
        return FakeLogitsOutput(logits)


def _make_provider(cls, encoder, **kwargs):
    provider = cls(FakeGenTokenizer(), encoder_path='unused', device='cpu', **kwargs)
    provider._enc_tok = FakeEncTokenizer()
    provider._encoder = encoder
    return provider


def test_e6_classifier_prefers_even_char_codes():
    # 'a'=97 (odd), 'b'=98 (even) -- 'b' should score near 1.0 (keep), 'a' near 0.0
    provider = _make_provider(E6RelevanceProvider, FakeClassifierEncoder(), keep_label=1)
    scores = provider.relevance([ord('a'), ord('b')])
    assert scores[0] < 0.01
    assert scores[1] > 0.99


def test_e6_classifier_keep_label_0_inverts():
    provider = _make_provider(E6RelevanceProvider, FakeClassifierEncoder(), keep_label=0)
    scores = provider.relevance([ord('a'), ord('b')])
    assert scores[0] > 0.99
    assert scores[1] < 0.01


def test_regression_provider_sigmoid_matches_manual_computation():
    provider = _make_provider(RegressionRelevanceProvider, FakeRegressionEncoder())
    char_code = 150
    scores = provider.relevance([char_code])
    expected = torch.sigmoid(torch.tensor((char_code - 100) / 10.0)).item()
    assert abs(scores[0] - expected) < 1e-4


def test_regression_provider_scores_bounded_0_1():
    provider = _make_provider(RegressionRelevanceProvider, FakeRegressionEncoder())
    scores = provider.relevance([ord(c) for c in "Hello World"])
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_both_providers_return_one_score_per_input_token():
    text = "abcdef"
    ids = [ord(c) for c in text]
    for provider in (
        _make_provider(E6RelevanceProvider, FakeClassifierEncoder()),
        _make_provider(RegressionRelevanceProvider, FakeRegressionEncoder()),
    ):
        assert len(provider.relevance(ids)) == len(ids)
