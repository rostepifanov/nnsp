import torch.nn as nn

from functools import partial
from timm.layers.activations import Swish
from timm.models.efficientnet import EfficientNet, decode_arch_def, round_channels

from nnspt.blocks.encoders.base import EncoderBase
from nnspt.blocks.encoders.converters import Converter1d, ConverterTimm

def get_efficientnet_v1_kwargs(channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
    """
        :NOTE:
            it is a modified copy of _gen_efficientnet from timm package
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]

    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, divisor=8)

    kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=Swish,
        drop_rate=drop_rate,
        drop_path_rate=0.2,
    )

    return kwargs

def gen_efficientnet_lite_kwargs(channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
    """
        :NOTE:
            it is a modified copy of _gen_efficientnet_lite from timm package
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r2_k5_s2_e6_c40'],
        ['ir_r3_k3_s2_e6_c80'],
        ['ir_r3_k5_s1_e6_c112'],
        ['ir_r4_k5_s2_e6_c192'],
        ['ir_r1_k3_s1_e6_c320'],
    ]

    kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
        num_features=1280,
        stem_size=32,
        fix_stem=True,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=nn.ReLU6,
        drop_rate=drop_rate,
        drop_path_rate=0.2,
    )

    return kwargs

def gen_efficientnetv2_base_kwargs(channel_multiplier=1.0, depth_multiplier=1.0):
    """
        :NOTE:
            it is a modified copy of _gen_efficientnetv2_base from timm package
    """
    arch_def = [
        ['cn_r1_k3_s1_e1_c16_skip'],
        ['er_r2_k3_s2_e4_c32'],
        ['er_r2_k3_s2_e4_c48'],
        ['ir_r3_k3_s2_e4_c96_se0.25'],
        ['ir_r5_k3_s1_e6_c112_se0.25'],
        ['ir_r8_k3_s2_e6_c192_se0.25'],
    ]

    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, round_limit=0.)

    kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=nn.SiLU,
    )

    return kwargs

def gen_efficientnetv2_small_kwargs(channel_multiplier=1.0, depth_multiplier=1.0):
    """
        :NOTE:
            it is a modified copy of _gen_efficientnetv2_s from timm package
    """
    arch_def = [
        ['cn_r2_k3_s1_e1_c24_skip'],
        ['er_r4_k3_s2_e4_c48'],
        ['er_r4_k3_s2_e4_c64'],
        ['ir_r6_k3_s2_e4_c128_se0.25'],
        ['ir_r9_k3_s1_e6_c160_se0.25'],
        ['ir_r15_k3_s2_e6_c256_se0.25'],
    ]

    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)

    kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=24,
        round_chs_fn=round_chs_fn,
        act_layer=nn.SiLU,
    )

    return kwargs

def gen_efficientnetv2_medium_kwargs(channel_multiplier=1.0, depth_multiplier=1.0):
    """
        :NOTE:
            it is a modified copy of _gen_efficientnetv2_m from timm package
    """
    arch_def = [
        ['cn_r3_k3_s1_e1_c24_skip'],
        ['er_r5_k3_s2_e4_c48'],
        ['er_r5_k3_s2_e4_c80'],
        ['ir_r7_k3_s2_e4_c160_se0.25'],
        ['ir_r14_k3_s1_e6_c176_se0.25'],
        ['ir_r18_k3_s2_e6_c304_se0.25'],
        ['ir_r5_k3_s1_e6_c512_se0.25'],
    ]

    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)

    kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=24,
        round_chs_fn=round_chs_fn,
        act_layer=nn.SiLU,
    )

    return kwargs

def gen_efficientnetv2_large_kwargs(channel_multiplier=1.0, depth_multiplier=1.0):
    """
        :NOTE:
            it is a modified copy of _gen_efficientnetv2_l from timm package
    """
    arch_def = [
        ['cn_r4_k3_s1_e1_c32_skip'],
        ['er_r7_k3_s2_e4_c64'],
        ['er_r7_k3_s2_e4_c96'],
        ['ir_r10_k3_s2_e4_c192_se0.25'],
        ['ir_r19_k3_s1_e6_c224_se0.25'],
        ['ir_r25_k3_s2_e6_c384_se0.25'],
        ['ir_r7_k3_s1_e6_c640_se0.25'],
    ]

    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)

    kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=nn.SiLU,
    )

    return kwargs

def gen_efficientnetv2_extralarge_kwargs(channel_multiplier=1.0, depth_multiplier=1.0):
    """
        :NOTE:
            it is a modified copy of _gen_efficientnetv2_xl from timm package
    """
    arch_def = [
        ['cn_r4_k3_s1_e1_c32_skip'],
        ['er_r8_k3_s2_e4_c64'],
        ['er_r8_k3_s2_e4_c96'],
        ['ir_r16_k3_s2_e4_c192_se0.25'],
        ['ir_r24_k3_s1_e6_c256_se0.25'],
        ['ir_r32_k3_s2_e6_c512_se0.25'],
        ['ir_r8_k3_s1_e6_c640_se0.25'],
    ]

    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)

    kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=nn.SiLU,
    )

    return kwargs

class EfficientNetEncoder(EfficientNet, EncoderBase):
    """Builder for encoder from EfficientNet family
    """
    def __init__(self, stage_idxs, out_channels, depth=5, **kwargs):
        """
            :NOTE:

            :args:
                stage_idxs: list of int
                    nested parameters for timm efficientnet
                out_channels: list of int
                    channel number of output tensors, including intermediate ones
                depth: int
                    depth of encoder
        """
        super().__init__(**kwargs)

        self.in_channels = 3
        self.out_channels_ = out_channels

        self.depth = depth
        self._stage_idxs = stage_idxs

        del self.classifier

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv_stem, self.bn1),
            self.blocks[:self._stage_idxs[0]],
            self.blocks[self._stage_idxs[0]:self._stage_idxs[1]],
            self.blocks[self._stage_idxs[1]:self._stage_idxs[2]],
            self.blocks[self._stage_idxs[2]:],
        ]

    def forward(self, x):
        """
            :args:
                x: [batch_size, in_channels, length] torch.tensor
                    input tensor

            :return:
                output: list of [batch_size, schannels, slength] torch.tensor
                    latent representations of input
        """
        stages = self.get_stages()
        features = []

        for i in range(self.depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

class EfficientNetV1Encoder(EfficientNetEncoder):
    """Builder for EfficientNetV1 encoders
    """
    def __init__(
        self,
        stage_idxs,
        out_channels,
        depth=5,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        drop_rate=0.2,
    ):
        kwargs = get_efficientnet_v1_kwargs(channel_multiplier, depth_multiplier, drop_rate)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)

        del self.conv_head, self.bn2

        ConverterTimm.convert(self)
        Converter1d.convert(self)

class EfficientNetLiteEncoder(EfficientNetEncoder):
    """Builder for EfficientNetLite encoders
    """
    def __init__(
        self,
        stage_idxs,
        out_channels,
        depth=5,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        drop_rate=0.2,
    ):
        kwargs = gen_efficientnet_lite_kwargs(channel_multiplier, depth_multiplier, drop_rate)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)

        del self.conv_head, self.bn2

        ConverterTimm.convert(self)
        Converter1d.convert(self)

class EfficientNetV2BaseEncoder(EfficientNetEncoder):
    """Builder for EfficientNetV2BaseEncoder encoders
    """
    def __init__(
        self,
        stage_idxs,
        out_channels,
        depth=5,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
    ):
        kwargs = gen_efficientnetv2_base_kwargs(channel_multiplier, depth_multiplier)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)

        del self.conv_head, self.bn2

        ConverterTimm.convert(self)
        Converter1d.convert(self)

class EfficientNetV2SmallEncoder(EfficientNetEncoder):
    """Builder for EfficientNetV2SmallEncoder encoder
    """
    def __init__(
        self,
        stage_idxs,
        out_channels,
        depth=5,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
    ):
        kwargs = gen_efficientnetv2_small_kwargs(channel_multiplier, depth_multiplier)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)

        del self.conv_head, self.bn2

        ConverterTimm.convert(self)
        Converter1d.convert(self)

class EfficientNetV2MediumEncoder(EfficientNetEncoder):
    """Builder for EfficientNetV2MediumEncoder encoder
    """
    def __init__(
        self,
        stage_idxs,
        out_channels,
        depth=5,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
    ):
        kwargs = gen_efficientnetv2_medium_kwargs(channel_multiplier, depth_multiplier)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)

        del self.conv_head, self.bn2

        ConverterTimm.convert(self)
        Converter1d.convert(self)

class EfficientNetV2LargeEncoder(EfficientNetEncoder):
    """Builder for EfficientNetV2LargeEncoder encoder
    """
    def __init__(
        self,
        stage_idxs,
        out_channels,
        depth=5,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
    ):
        kwargs = gen_efficientnetv2_large_kwargs(channel_multiplier, depth_multiplier)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)

        del self.conv_head, self.bn2

        ConverterTimm.convert(self)
        Converter1d.convert(self)

class EfficientNetV2ExtraLargeEncoder(EfficientNetEncoder):
    """Builder for EfficientNetV2ExtraLargeEncoder encoder
    """
    def __init__(
        self,
        stage_idxs,
        out_channels,
        depth=5,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
    ):
        kwargs = gen_efficientnetv2_extralarge_kwargs(channel_multiplier, depth_multiplier)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)

        del self.conv_head, self.bn2

        ConverterTimm.convert(self)
        Converter1d.convert(self)

efficientnet_encoders = {
    'timm-efficientnet-b0': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 32, 24, 40, 112, 320),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.0,
            'drop_rate': 0.2,
        },
    },
    'timm-efficientnet-b1': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 32, 24, 40, 112, 320),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.1,
            'drop_rate': 0.2,
        },
    },
    'timm-efficientnet-b2': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 32, 24, 48, 120, 352),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.1,
            'depth_multiplier': 1.2,
            'drop_rate': 0.3,
        },
    },
    'timm-efficientnet-b3': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 40, 32, 48, 136, 384),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.2,
            'depth_multiplier': 1.4,
            'drop_rate': 0.3,
        },
    },
    'timm-efficientnet-b4': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 48, 32, 56, 160, 448),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.4,
            'depth_multiplier': 1.8,
            'drop_rate': 0.4,
        },
    },
    'timm-efficientnet-b5': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 48, 40, 64, 176, 512),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.6,
            'depth_multiplier': 2.2,
            'drop_rate': 0.4,
        },
    },
    'timm-efficientnet-b6': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 56, 40, 72, 200, 576),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.8,
            'depth_multiplier': 2.6,
            'drop_rate': 0.5,
        },
    },
    'timm-efficientnet-b7': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 64, 48, 80, 224, 640),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 2.0,
            'depth_multiplier': 3.1,
            'drop_rate': 0.5,
        },
    },
    'timm-efficientnet-b8': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 72, 56, 88, 248, 704),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 2.2,
            'depth_multiplier': 3.6,
            'drop_rate': 0.5,
        },
    },
    'timm-efficientnet-l2': {
        'encoder': EfficientNetV1Encoder,
        'params': {
            'out_channels': (3, 136, 104, 176, 480, 1376),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 4.3,
            'depth_multiplier': 5.3,
            'drop_rate': 0.5,
        },
    },
    'timm-efficientnet-lite0': {
        'encoder': EfficientNetLiteEncoder,
        'params': {
            'out_channels': (3, 32, 24, 40, 112, 320),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.0,
            'drop_rate': 0.2,
        },
    },
    'timm-efficientnet-lite1': {
        'encoder': EfficientNetLiteEncoder,
        'params': {
            'out_channels': (3, 32, 24, 40, 112, 320),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.1,
            'drop_rate': 0.2,
        },
    },
    'timm-efficientnet-lite2': {
        'encoder': EfficientNetLiteEncoder,
        'params': {
            'out_channels': (3, 32, 24, 48, 120, 352),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.1,
            'depth_multiplier': 1.2,
            'drop_rate': 0.3,
        },
    },
    'timm-efficientnet-lite3': {
        'encoder': EfficientNetLiteEncoder,
        'params': {
            'out_channels': (3, 32, 32, 48, 136, 384),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.2,
            'depth_multiplier': 1.4,
            'drop_rate': 0.3,
        },
    },
    'timm-efficientnet-lite4': {
        'encoder': EfficientNetLiteEncoder,
        'params': {
            'out_channels': (3, 32, 32, 56, 160, 448),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.4,
            'depth_multiplier': 1.8,
            'drop_rate': 0.4,
        },
    },
    'timm-efficientnetv2-b0': {
        'encoder': EfficientNetV2BaseEncoder,
        'params': {
            'out_channels': (3, 32, 32, 48, 112, 192),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.0,
        },
    },
    'timm-efficientnetv2-b1': {
        'encoder': EfficientNetV2BaseEncoder,
        'params': {
            'out_channels': (3, 32, 32, 48, 112, 192),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.1,
        },
    },
    'timm-efficientnetv2-b2': {
        'encoder': EfficientNetV2BaseEncoder,
        'params': {
            'out_channels': (3, 32, 32, 56, 120, 208),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.1,
            'depth_multiplier': 1.2,
        },
    },
    'timm-efficientnetv2-b3': {
        'encoder': EfficientNetV2BaseEncoder,
        'params': {
            'out_channels': (3, 40, 40, 56, 136, 232),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.2,
            'depth_multiplier': 1.4,
        },
    },
    'timm-efficientnetv2-b4': {
        'encoder': EfficientNetV2BaseEncoder,
        'params': {
            'out_channels': (3, 48, 48, 64, 160, 272),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.4,
            'depth_multiplier': 1.8,
        },
    },
    'timm-efficientnetv2-b5': {
        'encoder': EfficientNetV2BaseEncoder,
        'params': {
            'out_channels': (3, 48, 48, 80, 176, 304),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.6,
            'depth_multiplier': 2.2,
        },
    },
    'timm-efficientnetv2-b6': {
        'encoder': EfficientNetV2BaseEncoder,
        'params': {
            'out_channels': (3, 56, 56, 88, 200, 344),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.8,
            'depth_multiplier': 2.6,
        },
    },
    'timm-efficientnetv2-b7': {
        'encoder': EfficientNetV2BaseEncoder,
        'params': {
            'out_channels': (3, 64, 64, 96, 224, 384),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 2.0,
            'depth_multiplier': 3.1,
        },
    },
    'timm-efficientnetv2-b8': {
        'encoder': EfficientNetV2BaseEncoder,
        'params': {
            'out_channels': (3, 72, 72, 104, 248, 424),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 2.2,
            'depth_multiplier': 3.6,
        },
    },
    'timm-efficientnetv2-s': {
        'encoder': EfficientNetV2SmallEncoder,
        'params': {
            'out_channels': (3, 24, 48, 64, 160, 256),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.0,
        },
    },
    'timm-efficientnetv2-m': {
        'encoder': EfficientNetV2MediumEncoder,
        'params': {
            'out_channels': (3, 24, 48, 80, 176, 512),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.0,
        },
    },
    'timm-efficientnetv2-l': {
        'encoder': EfficientNetV2LargeEncoder,
        'params': {
            'out_channels': (3, 32, 64, 96, 224, 640),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.0,
        },
    },
    'timm-efficientnetv2-xl': {
        'encoder': EfficientNetV2ExtraLargeEncoder,
        'params': {
            'out_channels': (3, 32, 64, 96, 256, 640),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.0,
        },
    },
}
