# A stand-in for torch, for tests/reference_dry.py (T357.4): what the reference tools (tests/reference_llama.py,
# tests/reference_qwen35.py, tests/reference_lfm2.py) ask of torch on the paths the dry run takes, on NumPy. Nothing
# here computes a model: transformers.py beside it answers with the engine itself. Not on any path but the dry run's.
import contextlib
import types

import numpy as np

__version__ = "stand-in"
float32, float16, bfloat16, long = np.float32, np.float16, np.float32, np.int64
_rng = np.random.default_rng(0)


class Tensor(np.ndarray):
    def float(self):
        return self

    def double(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def copy_(self, other):
        self[...] = other
        return self


def tensor(data, dtype=None):
    return np.array(data, dtype=dtype).view(Tensor)


def manual_seed(seed):
    global _rng
    _rng = np.random.default_rng(seed)


def randn_like(other):
    return _rng.standard_normal(other.shape).astype(np.float32).view(Tensor)


def rand_like(other):
    return _rng.random(other.shape).astype(np.float32).view(Tensor)


@contextlib.contextmanager
def no_grad():
    yield


class _Linear:
    forward = None


nn = types.SimpleNamespace(Linear=_Linear, functional=types.SimpleNamespace(linear=None))
