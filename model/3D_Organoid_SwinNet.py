
from model.fusion import BiFusion_block_3d
from model.SwinViT.SwinViT import SwinTransformer
import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg

rearrange, _ = optional_import("einops", name="rearrange")

class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class SegformerDecodeHead(nn.Module):
    def __init__(self,
                 hidden_sizes=[4*48 ,48,96 ,192 ,384 ],
                 num_encoder_blocks = 5,
                 target_size = [96,96, 96],
                 decoder_hidden_size = 384,
                 dropout_prob = 0.5,
                 num_labels= 14
                 ):
        super(SegformerDecodeHead, self).__init__()
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(num_encoder_blocks):
            # print("i:", i)
            # print("input dim: ",config.hidden_sizes[i] )
            mlp = SegformerMLP(decoder_hidden_size, input_dim = hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv3d(
            in_channels=decoder_hidden_size * num_encoder_blocks,
            out_channels=decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm3d(decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Conv3d(decoder_hidden_size, num_labels, kernel_size=1)

        # self.config = config
        self.target_size = target_size
    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            # if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
            #     height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
            #     encoder_hidden_state = (
            #         encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            #     )

            # unify channel dimension
            height, width, depth = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3], encoder_hidden_state.shape[4]
            # print("input shape mlp", encoder_hidden_state.shape)
            encoder_hidden_state = mlp(encoder_hidden_state)
            # print("output shape mlp", encoder_hidden_state.shape)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width, depth)
            # print("shape after reshaping", encoder_hidden_state.shape)
            # upsample we should map the data to 96 instead of 56
            # print("target shape: ", encoder_hidden_states[0].size()[2:])
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size= self.target_size, mode="trilinear", align_corners=False
            )
            # print("shape after interpolation", encoder_hidden_state.shape)

            all_hidden_states += (encoder_hidden_state,)
        # print("all_hidden_states----:", all_hidden_states[0].shape, all_hidden_states[1].shape, all_hidden_states[2].shape, all_hidden_states[3].shape)
        # print("torch.cat(all_hidden_states[::-1], dim=1)", torch.cat(all_hidden_states[::-1], dim=1).shape)
        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        # print("shape after concate:",hidden_states.shape )
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)
        # print("shape logits:",logits.shape )
        return logits


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}




__all__ = [
    "SimpleUNET",
    "SwinUNETR",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]


class 3D_Organoid_SwinNet(nn.Module):

    patch_size: Final[int] = 2

    @deprecated_arg(
        name="img_size",
        since="1.3",
        removed="1.5",
        msg_suffix="The img_size argument is not required anymore and "
        "checks on the input size are run during forward().",
    )
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:


        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )
        
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8* feature_size,
            out_channels=8* feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.SegformerDecodeHead = SegformerDecodeHead()


    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, x_in):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out_1 = self.swinViT_1(x_in, self.normalize)
        hidden_states_out_2 = self.swinViT_2(x_in[:,:,32:96,32:96,32:96], self.normalize)
        hidden_states_out_3 = self.swinViT_3(x_in[:,:,48:80,48:80,48:80], self.normalize)
        hidden_states_out_4 = self.swinViT_4(x_in[:,:,56:72,56:72,56:72], self.normalize)

        first_sum = torch.cat((hidden_states_out_1[1] , hidden_states_out_2[0]), dim =1 )
        second_sum= torch.cat((hidden_states_out_1[2] , hidden_states_out_2[1], hidden_states_out_3[0]), dim =1 )
        third_sum= torch.cat((hidden_states_out_1[3] , hidden_states_out_2[2], hidden_states_out_3[1], hidden_states_out_4[0] ), dim =1 )

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out_1[0])
        enc1 = self.fusion2(enc1,hidden_states_out_1[0])
        enc2 = self.encoder3(first_sum)


        enc2 = self.fusion3(enc2,first_sum)
        enc3 = self.encoder4(second_sum)
        enc3 = self.fusion4(enc3,second_sum)
        dec4 = self.Conv3D_1D(third_sum)
        dec4 = self.fusion5(dec4,third_sum)
        dec2 = self.decoder4(dec4, enc3)

        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits


