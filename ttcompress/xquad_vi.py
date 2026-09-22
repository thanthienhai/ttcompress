"""XQuAD Vietnamese (Artetxe, Ruder & Yogatama, ACL 2020, arXiv:1910.11856) --
a second, independent public monolingual-Vietnamese QA source, alongside
UIT-ViQuAD 2.0. XQuAD is the SQuAD 1.1 dev set (240 paragraphs, 1190
questions) professionally translated into 11 languages including Vietnamese;
verified on HuggingFace (`xquad`, config `xquad.vi`, plain parquet, no
loading script) 2026-09-22.

Same SQuAD-style schema as UIT-ViQuAD (context/question/answers with char
offsets), so this reuses uit_viquad.ViquadSample directly rather than a new
dataclass -- build_document.py/build_paragraph_pool work unchanged. No
`title` field in the source data (left empty) and no unanswerable questions
(XQuAD is extractive-only, unlike UIT-ViQuAD 2.0's SQuAD2.0-style negatives).

No official train/dev/test split of its own -- XQuAD is used purely as an
evaluation benchmark in the literature (it IS a translated SQuAD dev set),
never for training. split_dev_test() here does the same seeded-shuffle split
ttcompress.public_datasets' LongBench loader uses: a small slice reserved for
the lambda sweep, the rest for the one-time official test.
"""
from __future__ import annotations

import random
from typing import List, Tuple

from .uit_viquad import ViquadSample

HF_DATASET_ID = 'xquad'
_PARQUET_PATH = 'xquad.vi/validation-00000-of-00001.parquet'


def load_all() -> List[ViquadSample]:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download(repo_id=HF_DATASET_ID, repo_type='dataset', filename=_PARQUET_PATH)
    df = pd.read_parquet(path)
    samples = []
    for _, row in df.iterrows():
        answers = row['answers']
        samples.append(ViquadSample(
            id=str(row['id']), title='', context=row['context'], question=row['question'],
            answer_text=str(answers['text'][0]), answer_start=int(answers['answer_start'][0]),
        ))
    return samples


def split_dev_test(num_dev: int = 20, seed: int = 1234) -> Tuple[List[ViquadSample], List[ViquadSample]]:
    samples = load_all()
    rng = random.Random(seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    dev_idx, test_idx = set(order[:num_dev]), set(order[num_dev:])
    dev = [samples[i] for i in sorted(dev_idx)]
    test = [samples[i] for i in sorted(test_idx)]
    return dev, test
