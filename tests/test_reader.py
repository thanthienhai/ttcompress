import os

import pytest

from ttcompress.reader import HFReader, clean_answer, reader_tag, resolve_device
from tests.conftest import TINY_CAUSAL_LM

os.environ.setdefault('HF_HUB_OFFLINE', '1')


def test_reader_tag_and_clean_answer():
    assert reader_tag('Qwen/Qwen3-8B') == 'Qwen--Qwen3-8B'
    assert clean_answer('<think>\nhmm\n</think>\n\n Hà Nội ') == 'Hà Nội'
    assert clean_answer(None) == ''


def test_resolve_device_honours_indexed_cuda_and_cpu():
    import torch
    assert resolve_device('cpu') == 'cpu'
    expected = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    assert resolve_device('cuda:1') == expected


@pytest.fixture(scope='module')
def reader():
    return HFReader(TINY_CAUSAL_LM, device='cpu', batch_size=3)


def test_prompt_without_chat_template_contains_context_and_question(reader):
    p = reader.build_prompt('NGỮ CẢNH', 'CÂU HỎI', 'vi')
    assert 'NGỮ CẢNH' in p and 'CÂU HỎI' in p
    assert 'Context' in reader.build_prompt('c', 'q', 'en')


def test_batched_generation_equals_one_at_a_time(reader):
    prompts = [reader.build_prompt('ab ' * n, 'q', 'en') for n in (1, 7, 3, 12, 5)]
    batched = reader.generate(prompts, max_new_tokens=5)
    single = [reader.generate([p], max_new_tokens=5)[0] for p in prompts]
    assert batched == single


def test_batched_answer_logprob_equals_one_at_a_time(reader):
    prompts = [reader.build_prompt('xy ' * n, 'q', 'en') for n in (2, 9, 4)]
    answers = ['foo', 'bar baz', 'qux']
    batched = reader.answer_logprob(prompts, answers)
    single = [reader.answer_logprob([p], [a])[0] for p, a in zip(prompts, answers)]
    assert all(b < 0 for b in batched)
    assert batched == pytest.approx(single, abs=1e-4)
