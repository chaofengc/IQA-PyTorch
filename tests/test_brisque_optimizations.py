import torch

from pyiqa.utils.color_util import rgb2yiq, to_y_channel


def test_y_channel_matches_full_yiq_first_channel_after_rounding():
    torch.manual_seed(29)
    image = torch.rand(8, 3, 64, 64)
    expected = rgb2yiq(image)[:, [0]] * 255
    expected = expected - expected.detach() + expected.round()

    actual = to_y_channel(image, 255)

    assert torch.equal(actual, expected)


def test_y_channel_matches_full_yiq_at_unit_range():
    image = torch.rand(2, 3, 16, 16)
    expected = rgb2yiq(image)[:, [0]]

    torch.testing.assert_close(to_y_channel(image, 1), expected)
