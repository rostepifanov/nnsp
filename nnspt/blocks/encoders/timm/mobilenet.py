import torch.nn as nn

from functools import partial
from timm.models.efficientnet import decode_arch_def, round_channels

from nnspt.blocks.encoders.timm.efficientnet import EfficientNetEncoder
from nnspt.blocks.encoders.converters import Converter1d, ConverterTimm

def get_mobilenet_v2_kwargs(channel_multiplier=1.0, depth_multiplier=1.0, fix_stem_head=False):
    """
        :NOTE:
            it is a modified copy of _gen_mobilenet_v2 from timm package
    """
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r4_k3_s2_e6_c64'],
        ['ir_r3_k3_s1_e6_c96'],
        ['ir_r3_k3_s2_e6_c160'],
        ['ir_r1_k3_s1_e6_c320'],
    ]

    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)

    kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier=depth_multiplier, fix_first_last=fix_stem_head),
        num_features=1280 if fix_stem_head else max(1280, round_chs_fn(1280)),
        stem_size=32,
        fix_stem=fix_stem_head,
        round_chs_fn=round_chs_fn,
        act_layer=nn.ReLU6,
    )

    return kwargs

class MobileNetV2Encoder(EfficientNetEncoder):
    """Builder for MobileNetV2 encoders
    """
    def __init__(
        self,
        stage_idxs,
        out_channels,
        depth=5,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        fix_stem_head=False,
    ):
        kwargs = get_mobilenet_v2_kwargs(channel_multiplier, depth_multiplier, fix_stem_head)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)

        del self.conv_head, self.bn2

        ConverterTimm.convert(self)
        Converter1d.convert(self)

mobilenet_encoders = {
    'timm-mobilenetv2-035': {
        'encoder': MobileNetV2Encoder,
        'params': {
            'out_channels': (3, 16, 8, 16, 32, 112),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 0.35,
            'depth_multiplier': 1.0,
            'fix_stem_head': False,
        },
    },
    'timm-mobilenetv2-050': {
        'encoder': MobileNetV2Encoder,
        'params': {
            'out_channels': (3, 16, 16, 16, 48, 160),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 0.5,
            'depth_multiplier': 1.0,
            'fix_stem_head': False,
        },
    },
    'timm-mobilenetv2-075': {
        'encoder': MobileNetV2Encoder,
        'params': {
            'out_channels': (3, 24, 24, 24, 72, 240),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 0.75,
            'depth_multiplier': 1.0,
            'fix_stem_head': False,
        },
    },
    'timm-mobilenetv2-100': {
        'encoder': MobileNetV2Encoder,
        'params': {
            'out_channels': (3, 32, 24, 32, 96, 320),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.00,
            'depth_multiplier': 1.0,
            'fix_stem_head': False,
        },
    },
    'timm-mobilenetv2-140': {
        'encoder': MobileNetV2Encoder,
        'params': {
            'out_channels': (3, 48, 32, 48, 136, 448),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.40,
            'depth_multiplier': 1.0,
            'fix_stem_head': False,
        },
    },
}
