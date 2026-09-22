import os

# Windows-only workaround: pyarrow's internal thread pool (used by
# pandas.read_parquet, in xquad_vi.py/uit_viquad.py) intermittently crashes
# the whole interpreter with "Windows fatal exception: access violation"
# after repeated reads within one process -- reproduced running the full
# suite, never in isolation. Forcing single-threaded pyarrow/OpenMP avoids it;
# must be set before pyarrow is imported anywhere; the datasets are tiny
# (a few hundred KB) so the perf cost is negligible.
os.environ.setdefault('PYARROW_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import pytest


class FakeTokenizer:
    """Minimal stand-in so compressor tests don't need to download a real HF
    tokenizer. Token ids are just ints; decode renders them as text."""

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ' '.join(f"tok{i}" for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 1000 for w in text.split()]


@pytest.fixture
def tokenizer():
    return FakeTokenizer()
