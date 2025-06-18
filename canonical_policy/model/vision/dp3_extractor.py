import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import math
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from einops import rearrange
from pytorch3d.ops import sample_farthest_points
    
def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_
        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')


        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
    

    def forward(self, x):

        x = self.mlp(x)  # [B, N, 512]

        feat = torch.max(x, 1)[0]  # [B, 64]

        feat = self.final_projection(feat)  # [B, 64]

        return feat


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 out_channel=256,
                 num_points=256,
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 use_pc_color=False,
                 pointnet_type='dp3',
                 state_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_keys = state_keys
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_size = sum([observation_space[key][0] for key in self.state_keys])
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None

        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_size}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        self.num_points = num_points
        self.obs_dim = out_channel

        if pointnet_type == 'dp3':
            self.extractor = PointNetEncoderXYZRGB(out_channels=self.obs_dim, use_layernorm=True, final_norm='layernorm')
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_size, output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]  # [B*To, N, 6]
        # sampling
        points, _ = sample_farthest_points(points, K=self.num_points)  # [BTo, num_samples, 6]
        points_xyz = points[..., :3]    # [BTo, num_samples, 3]

        # extract state
        state_pos = observations[self.state_keys[0]]  # [BTo, 3]
        state_quat = observations[self.state_keys[1]]  # [BTo, 4]  ijkw
        state_gripper = observations[self.state_keys[2]]  # [BTo, 2]

        if self.pointnet_type == "dp3":
            points_feat = self.extractor(points_xyz)    # [BTo, 64]
            state = torch.cat([state_pos, state_quat, state_gripper], dim=-1)  # [B, Ds]
            state_feat = self.state_mlp(state)  # B * 64
            final_feat = torch.cat([points_feat, state_feat], dim=-1)
        else:
            raise NotImplementedError(f"pointnet_type: {self.pointnet_type}")

        return final_feat

    def output_shape(self):
        return self.n_output_channels