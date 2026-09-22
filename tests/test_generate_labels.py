"""generate_labels.py cmd_measure saves one MaskOutcomeRecord file per
document, keyed by filename -- ConstructedDocument.doc_id is a cluster label
(RULER/Kamradt legitimately reuse one per essay, see
test_multilingual_sources.py), not a unique id, so cmd_measure must
disambiguate before using it as a filename. Real bug: with the default
samples_per_essay=2, two distinct RULER/Kamradt documents sharing an essay
used to silently overwrite one measured MaskOutcomeRecord with the other."""
import glob
import os
import tempfile
import types

import generate_labels as gl


def fake_generate_answer(model, tokenizer, compressed_ids, query, max_new_tokens=32, prompt_kind='generic'):
    return "an answer"


class FakeModel:
    device = 'cpu'


class FakeTok:
    chat_template = None

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ' '.join(str(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 5000 for w in text.split()]


def test_measure_saves_one_file_per_document_even_with_colliding_doc_ids(monkeypatch):
    monkeypatch.setattr(gl, 'load_reader', lambda model_name, device='cuda': (FakeModel(), FakeTok()))
    monkeypatch.setattr(gl, 'generate_answer', fake_generate_answer)

    with tempfile.TemporaryDirectory() as out_dir:
        args = types.SimpleNamespace(
            split='dev', sources='ruler,kamradt', reader_model='fake', n=12, k=3,
            max_new_tokens=None, device='cpu', seed=1, overlap_report=None, out_dir=out_dir,
        )
        gl.cmd_measure(args)
        files = glob.glob(os.path.join(out_dir, '*.jsonl'))
        assert len(files) == 12  # one per document, no silent overwrite
