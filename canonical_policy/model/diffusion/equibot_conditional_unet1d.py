from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from canonical_policy.model.diffusion.equibot_conv1d_components import (
    VecDownsample1d,
    VecUpsample1d,
    VecConv1dBlock,
)
from canonical_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from canonical_policy.model.vision.canonical_utils.vec_layers import VecLinNormAct as VecLNA
from canonical_policy.model.vision.canonical_utils.vec_layers import VecLinear
logger = logging.getLogger(__name__)

class VecConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        vec_cond_dim,
        scalar_cond_dim=0,
        kernel_size=3,
        cond_predict_scale=False,
    ):
        # input dimensionality: ()
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                VecConv1dBlock(in_channels, out_channels, kernel_size),
                VecConv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        act_func = nn.Mish()
        vnla_cfg = dict(mode="so3", s_in=scalar_cond_dim, return_tuple=False)
        self.cond_encoder_l1 = VecLNA(vec_cond_dim, cond_channels, act_func, **vnla_cfg)
        self.cond_encoder_l2 = VecLinear(
            cond_channels, out_channels, s_out=out_channels if cond_predict_scale else 0
        )

        # make sure dimensions compatible
        self.residual_conv = (
            VecDownsample1d(in_channels, out_channels, 1, 1, 0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, vec_cond, scalar_cond=None):
        """
        x : [ batch_size x in_channels x 3 x horizon ]
        vec_cond : [ batch_size x vec_cond_dim x 3]
        scalar_cond : [ batch_size x scalar_cond_dim ]

        returns:
        out : [ batch_size x out_channels x 3 x horizon ]
        """
        out = self.blocks[0](x)
        bias, scale = self.cond_encoder_l2(
            self.cond_encoder_l1(vec_cond, s=scalar_cond)
        )
        if self.cond_predict_scale:
            out = scale[..., None, None] * out + bias[..., None]
        else:
            out = out + bias[..., None]
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class VecConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        n_obs_steps=2,
        scalar_input_dim=0,
        scalar_cond_dim=0,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        cond_predict_scale=False,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        self.n_obs_steps = n_obs_steps

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        vec_cond_dim = cond_dim
        scalar_cond_dim = dsed + scalar_cond_dim

        if scalar_input_dim > 0:
            self.scalar_fusion_module = VecLNA(
                input_dim,
                down_dims[0],
                act_func=nn.Mish(),
                s_in=scalar_input_dim,
                mode="so3",
                return_tuple=False,
            )
            all_dims[0] = down_dims[0]

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        make_res_block = lambda din, dout: VecConditionalResidualBlock1D(
            din,
            dout,
            vec_cond_dim,
            scalar_cond_dim=scalar_cond_dim,
            kernel_size=kernel_size,
            cond_predict_scale=cond_predict_scale,
        )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [make_res_block(mid_dim, mid_dim), make_res_block(mid_dim, mid_dim)]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        make_res_block(dim_in, dim_out),
                        make_res_block(dim_out, dim_out),
                        VecDownsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        make_res_block(dim_out * 2, dim_in),
                        make_res_block(dim_in, dim_in),
                        VecUpsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            VecConv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            VecDownsample1d(start_dim, input_dim, 1, 1, 0),
        )

        if scalar_input_dim > 0:
            self.final_linear = VecLinear(start_dim, 0, s_out=scalar_input_dim)

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        obs_cond_vec=None,
        obs_cond_scalar=None,
        points_center=None,
        points_scale=None,
        **kwargs
    ):
        """
        Forward pass of the EquiBot Conditional UNet1D model.
        Args:
            sample (torch.Tensor): Input tensor of shape (B, T, input_dim), where:
                - B: Batch size
                - T: Sequence length
                - input_dim: Dimensionality of the input
            timestep (Union[torch.Tensor, float, int]): Diffusion step, either as:
                - A tensor of shape (B,) representing the batch-wise timesteps.
                - A single integer or float value.
            obs_cond_vec (torch.Tensor): Observation conditioning vector of shape (B, cond_dim, 3), where:
                - cond_dim: Dimensionality of the conditioning vector.
            obs_cond_scalar (torch.Tensor): Observation conditioning scalar of shape (B, cond_dim), where:
                - cond_dim: Dimensionality of the conditioning scalar.
            points_center (torch.Tensor): Tensor of shape (BTo, 1, 3) representing the center points for normalization.
            points_scale (torch.Tensor): Tensor of shape (BTo) representing the scaling factors for normalization.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: Output tensor of shape (B, T, output_dim), where:
                - output_dim: Dimensionality of the output, including positional, rotational, and scalar components.
        """

        B, T = sample.shape[:2]

        points_center = points_center.reshape(B, self.n_obs_steps, 3)[:, [-1]].repeat(1, T, 1)  # [B, T, 3]
        points_scale = points_scale.reshape(B, self.n_obs_steps, 1)[:, [-1]].repeat(1, T, 1)  # [B, T, 1]

        sample_pos, sample_sixd, sample_gripper = sample[..., :3], sample[..., 3:9], sample[..., [-1]]  # [B, T, 3], [B, T, 6], [B, T, 1]
        sample_dir1, sample_dir2 = sample_sixd[..., :3], sample_sixd[..., 3:]

        sample_pos = (sample_pos - points_center) / points_scale   # [B, T, 3]
        sample = torch.cat((sample_pos.unsqueeze(2), sample_dir1.unsqueeze(2), sample_dir2.unsqueeze(2)), dim=2)    # [B, T, 3, 3]

        sample = einops.rearrange(sample, "b h t v -> b t v h")
        scalar_sample = einops.rearrange(sample_gripper, "b h t -> b t h")

        # 1. encode timestep
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        # 2. conditioning
        assert obs_cond_vec is not None
        vec_feature = obs_cond_vec
        scalar_feature = self.diffusion_step_encoder(timesteps)  # (B, dsed)
        scalar_feature = torch.cat([scalar_feature, obs_cond_scalar], dim=-1)

        # 3. forward pass through the unet
        # 3.1 fuse vector and scalar samples if needed
        assert hasattr(self, "scalar_fusion_module")
        sample = self.scalar_fusion_module(sample, scalar_sample)

        # 3.2 unet
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, vec_feature, scalar_feature)
            x = resnet2(x, vec_feature, scalar_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, vec_feature, scalar_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, vec_feature, scalar_feature)
            x = resnet2(x, vec_feature, scalar_feature)
            x = upsample(x)

        assert hasattr(self, "final_linear")
        x_scalar = self.final_linear(x)[1]
        x_scalar = einops.rearrange(x_scalar, "b t h -> b h t")

        x = self.final_conv(x)
        x = einops.rearrange(x, "b t v h -> b h t v")  # [B, T, 3, 3]

        x_pos, x_dir1, x_dir2 = x[:, :, 0], x[:, :, 1], x[:, :, 2]  # [B, T, 3]

        x_pos = x_pos * points_scale + points_center  # [B, T, 3]
        x_sixd = torch.cat([x_dir1, x_dir2], dim=-1)   # [B, T, 6]

        x = torch.cat((x_pos, x_sixd, x_scalar), dim=-1)

        return x
