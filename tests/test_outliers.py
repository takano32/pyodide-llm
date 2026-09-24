# The classifier's outlier channels (T92, OUTLIER_CHANNELS): the channels with the largest final-norm weight, and
# their columns of the int8 classifier widened to float32, which the kernel forward multiplies separately.
import numpy as np

from llama2_numpy import OUTLIER_CHANNELS, OUTLIER_RATIO, outlier_channels, outlier_columns


def test_the_columns_are_those_of_the_widened_classifier():
    rng = np.random.default_rng(0)
    vocab, dim, group = 50, 96, 32
    values = rng.integers(-127, 128, size=(vocab, dim // group, group), dtype=np.int8)
    scales = rng.random((vocab, dim // group, 1), dtype=np.float32) + 0.5
    weight = rng.standard_normal(dim).astype(np.float32)
    weight[[7, 40, 95]] = [30.0, -25.0, 20.0]  # the three that a norm's weight blows up
    channels = outlier_channels(weight, 3)
    assert channels.tolist() == [7, 40, 95], "the largest of |weight|, in the order of the channels"
    columns = outlier_columns((values, scales), channels)
    widened = (values.astype(np.float32) * scales).reshape(vocab, dim)
    assert columns.shape == (3, vocab) and columns.dtype == np.float32 and columns.flags.c_contiguous
    assert np.array_equal(columns, widened[:, channels].T)


def test_a_norm_without_outliers_gets_no_channels():
    assert (OUTLIER_CHANNELS, OUTLIER_RATIO) == (8, 4.0)
    flat = np.ones(64, dtype=np.float32)
    flat[3] = 3.9  # tiny-lm's largest is 1.1 times its median, GPT-2's 13.9: the line is 4
    assert outlier_channels(flat).size == 0
    flat[3] = 4.0
    channels = outlier_channels(flat)
    assert channels.size == 8 and 3 in channels.tolist(), "eight channels, the outlier among them"


def test_never_more_columns_than_channels():
    values = np.ones((5, 1, 4), dtype=np.int8)
    scales = np.ones((5, 1, 1), dtype=np.float32)
    weight = np.array([1, 1, 1, 9], dtype=np.float32)
    channels = outlier_channels(weight, min(OUTLIER_CHANNELS, 4))
    assert channels.tolist() == [0, 1, 2, 3] and outlier_columns((values, scales), channels).shape == (4, 5)
