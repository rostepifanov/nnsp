import pytest

import torch

from nnspt.pretraining.autoencoder import Autoencoder
from nnspt.blocks.encoders.builder import nnspt_encoders

ENCODERS = nnspt_encoders.keys()

@pytest.mark.autoencoder
@pytest.mark.pretraining
def test_Autoencoder_CASE_creation_AND_defaul_parameters():
    model = Autoencoder()

@pytest.mark.autoencoder
@pytest.mark.pretraining
def test_Autoencoder_CASE_wrong_encoder_name():
    with pytest.raises(KeyError) as e:
        model = Autoencoder(encoder='wrong_name')

@pytest.mark.autoencoder
@pytest.mark.pretraining
@pytest.mark.parametrize('name', ENCODERS)
def test_Autoencoder_CASE_creation(name):
    NCHANNELS = 12
    DEPTH = 4

    model = Autoencoder(
        nchannels=NCHANNELS,
        depth=DEPTH,
        encoder=name,
    )

    model.eval()

    x = torch.randn(1, NCHANNELS, 32)

    with torch.no_grad():
        y = model(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1]
    assert y.shape[2] == x.shape[2]
