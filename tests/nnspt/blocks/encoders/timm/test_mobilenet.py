import pytest

import torch

from nnspt.blocks.encoders import Encoder
from nnspt.blocks.encoders.timm import mobilenet_encoders

ENCODERS = mobilenet_encoders.keys()

@pytest.mark.mobilenet
@pytest.mark.encoders
@pytest.mark.parametrize('name', ENCODERS)
def test_MobileNetEncoder_CASE_creation(name):
    IN_CHANNELS = 3
    DEPTH = 5

    encoder = Encoder(
        in_channels=IN_CHANNELS,
        depth=DEPTH,
        name=name
    )

    assert len(encoder.out_channels) == DEPTH + 1

    encoder.eval()

    x = torch.randn(1, IN_CHANNELS, 32)

    with torch.no_grad():
        y = encoder(x)

    for nchannels, yi in zip(encoder.out_channels, y):
        assert yi.shape[0] == x.shape[0]
        assert yi.shape[1] == nchannels

@pytest.mark.mobilenet
@pytest.mark.encoders
@pytest.mark.parametrize('name', ENCODERS)
def test_MobileNetEncoder_CASE_unused_layer_check(name):
    IN_CHANNELS = 3
    DEPTH = 5

    encoder = Encoder(
        in_channels=IN_CHANNELS,
        depth=DEPTH,
        name=name
    )

    encoder.train()

    x = torch.randn(2, IN_CHANNELS, 32)
    y = encoder(x)[-1]

    loss = y.sum()
    loss.backward()

    for n, p in encoder.named_parameters():
        assert p.grad is not None, f'{n} parameter has not a grad'
