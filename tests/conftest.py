import os

# Windows-only workaround: pyarrow's internal thread pool (pandas.read_parquet)
# intermittently crashes the interpreter with an access violation after
# repeated reads in one process. Single-threaded pyarrow/OpenMP avoids it.
os.environ.setdefault('PYARROW_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import pytest  # noqa: E402

from ttcompress.data import QADocument  # noqa: E402

TINY_ENCODER = 'hf-internal-testing/tiny-random-RobertaModel'
TINY_CAUSAL_LM = 'hf-internal-testing/tiny-random-gpt2'


class FakeTokenizer:
    """Whitespace tokenizer: one token per word, decode joins with spaces."""

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 1000 for w in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(f"tok{i}" for i in ids)


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


def make_doc(chunks, gold=(0,), question='q', answers=('a',), doc_id='d0', source='uit_viquad', hop='single'):
    return QADocument(doc_id=doc_id, source=source, language='vi', hop=hop, split='dev', question=question,
                      answers=list(answers), chunks=list(chunks), gold_chunks=list(gold), cluster_id=doc_id)
