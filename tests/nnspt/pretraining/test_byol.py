import pytest

import torch

from nnspt.pretraining.byol import BYOL
from nnspt.blocks.encoders.builder import nnspt_encoders

ENCODERS = nnspt_encoders.keys()

@pytest.mark.byol
@pytest.mark.pretraining
def test_BYOL_CASE_creation_AND_defaul_parameters():
    model = BYOL()

@pytest.mark.byol
@pytest.mark.pretraining
def test_BYOL_CASE_wrong_encoder_name():
    with pytest.raises(KeyError) as e:
        model = BYOL(encoder='wrong_name')

@pytest.mark.byol
@pytest.mark.pretraining
@pytest.mark.parametrize('name', ENCODERS)
def test_Autoencoder_CASE_creation(name):
    NCHANNELS = 12
    DEPTH = 4
    REDUCTION = 8

    model = BYOL(
        nchannels=NCHANNELS,
        depth=DEPTH,
        encoder=name,
        reduction=REDUCTION
    )

    model.eval()

    x = torch.randn(1, NCHANNELS, 32)

    with torch.no_grad():
        y1, y2 = model(x, x)

    assert y1.shape[0] == x.shape[0]
    assert y1.shape[1] == (model.encoder.out_channels[-1] + REDUCTION - 1) // REDUCTION

    assert y2.shape[0] == x.shape[0]
    assert y2.shape[1] == (model.encoder.out_channels[-1] + REDUCTION - 1) // REDUCTION

    assert y1.shape[0] == y2.shape[0]
    assert y1.shape[1] == y2.shape[1]
