"""make_compressor('h2o'/'snapkv', ...) used to hardcode SnapKVCompressor's
own device default ('cuda'), ignoring --device entirely -- invisible on a
single-GPU cuda:0 run, but a real mismatch on a CPU-only smoke test, a
non-cuda:0 device, or run_pipeline_multi_gpu.sh's per-shard cuda:i pinning
(every shard needs its own SnapKVCompressor actually running on ITS GPU, not
always cuda:0). Fixed by deriving the device from the model actually doing
the forward pass, not a hardcoded default."""
from ttcompress.registry import make_compressor


class _FakeModel:
    def __init__(self, device):
        self.device = device


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(str(i) for i in ids)


def test_h2o_device_follows_the_model_when_not_given_explicitly():
    model = _FakeModel('cuda:2')
    compressor = make_compressor('h2o', 8.0, _FakeTokenizer(), model)
    assert compressor.device == 'cuda:2'


def test_snapkv_device_follows_the_model_when_not_given_explicitly():
    model = _FakeModel('cpu')
    compressor = make_compressor('snapkv', 8.0, _FakeTokenizer(), model)
    assert compressor.device == 'cpu'


def test_h2o_explicit_device_overrides_the_model():
    model = _FakeModel('cuda:0')
    compressor = make_compressor('h2o', 8.0, _FakeTokenizer(), model, device='cuda:3')
    assert compressor.device == 'cuda:3'


def test_h2o_falls_back_to_cuda_when_model_has_no_device_attribute():
    class _NoDeviceModel:
        pass
    compressor = make_compressor('h2o', 8.0, _FakeTokenizer(), _NoDeviceModel())
    assert compressor.device == 'cuda'
