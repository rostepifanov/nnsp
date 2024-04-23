import pytest

import torch

from nnspt.blocks.layers.conv import CausualConv1d

@pytest.mark.layers
def test_CausaulConv1d_CASE_forward():
    IN_CHANNELS = 3
    OUT_CHANNELS = 2

    layer = CausualConv1d(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=3,
    )

    x = torch.randn(1, IN_CHANNELS, 64)
    y = layer(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == OUT_CHANNELS
    assert y.shape[2] == x.shape[2]
