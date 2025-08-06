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
from pytorch3d.transforms import quaternion_apply, quaternion_invert, quaternion_multiply, matrix_to_quaternion
from canonical_policy.model.vision.canonical_utils.vec_pointnet import VecPointNet, VN_Regressor
from canonical_policy.model.vision.canonical_utils.agg_encoder import AggEncoder
from canonical_policy.model.vision.canonical_utils.utils import construct_rotation_matrix, extract_z_rotation
    
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

class CanonicalEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict,
                 canonical_encoder_cfg: Optional[Dict],
                 out_channel=256,
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 use_pc_color=False,
                 pointnet_type='cp_so2',
                 state_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
                 point_cloud_key='point_cloud',
                 ):
        super().__init__()
        self.state_keys = state_keys
        self.point_cloud_key = point_cloud_key

        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_size = sum([observation_space[key][0] for key in self.state_keys])

        cprint(f"[CanonicalEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[CanonicalEncoder] state shape: {self.state_size}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.obs_dim = out_channel

        self.num_points = canonical_encoder_cfg.num_points  
        self.ksize = canonical_encoder_cfg.neighbor_num
        self.rot_hidden_dim = canonical_encoder_cfg.rot_hidden_dim
        self.rot_layers = canonical_encoder_cfg.rot_layers

        if pointnet_type == "cp_so2" or pointnet_type == "cp_so3":
            self.xyz_extractor = AggEncoder(hidden_dim=self.obs_dim, ksize=self.ksize)
            self.equiv_extractor = VecPointNet(h_dim=self.rot_hidden_dim, c_dim=self.rot_hidden_dim, num_layers=self.rot_layers, ksize=self.ksize)
            self.rot_predictor = VN_Regressor(pc_feat_dim=self.rot_hidden_dim)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        state_dim = state_mlp_size[-1]
        self.state_mlp = nn.Sequential(*create_mlp(self.state_size, state_dim, net_arch, state_mlp_activation_fn))

        self.n_output_channels = self.obs_dim + state_dim
        cprint(f"[CanonicalEncoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:

        # extract state
        state_pos = observations[self.state_keys[0]]  # [BTo, 3]
        state_quat = observations[self.state_keys[1]]  # [BTo, 4]  ijkw
        state_quat = state_quat[:, [3, 0, 1, 2]]  # [BTo, 4]  wijk
        state_gripper = observations[self.state_keys[2]]  # [BTo, 2]

        # extract point cloud
        points = observations[self.point_cloud_key]  # [B*To, N, 6]
        points, _ = sample_farthest_points(points, K=self.num_points)  # [BTo, num_samples, 6]
        points_xyz = points[..., :3]    # [BTo, num_samples, 3]

        # 1. Point Cloud Decenterization
        points_center = torch.mean(points_xyz, dim=1, keepdim=True)  # [BTo, 1, 3]
        input_pcl = points_xyz - points_center  # [BTo, N, 3]

        # 2. Using SO3-equivariant Network to predict rotation
        equiv_feat = self.equiv_extractor(input_pcl)  # [B, 32, 3], [B, 64]
        v1, v2 = self.rot_predictor(equiv_feat)
        rot_mat = construct_rotation_matrix(v1, v2)  # [B, 3, 3]  SO3

        if self.pointnet_type == "cp_so2":
            est_rot = extract_z_rotation(rot_mat)  # [BTo, 3, 3]  SO2
        elif self.pointnet_type == "cp_so3":
            est_rot = rot_mat  # [BTo, 3, 3]  SO3
        else:   
            raise NotImplementedError(f"pointnet_type: {self.pointnet_type}")
        
        # Use quaternion to represent the rotation
        est_quat = matrix_to_quaternion(est_rot)    # [BTo, 4]  wijk
        est_quat_inv = quaternion_invert(est_quat)      # [BTo, 4]  wijk

        # 3. SE3-inverse for state
        state_pos = state_pos - points_center.squeeze(1)  # [BTo, 3]
        state_pos = quaternion_apply(est_quat_inv, state_pos)   # [BTo, 3]
        state_quat = quaternion_multiply(est_quat_inv, state_quat)  # [BTo, 4]  wijk

        # 4. SO3-inverse for point cloud, (R^(-1)*x.T).T = x*R
        rot_input_pcl = torch.matmul(input_pcl, est_rot)

        # 5. Extract features for observations
        xyz_feat = self.xyz_extractor(rot_input_pcl)  # [BTo, obs_dim]
        state = torch.cat([state_pos, state_quat, state_gripper], dim=-1)   # [BTo, Ds]
        state_feat = self.state_mlp(state)  # [BTo, state_dim]

        final_feat = torch.cat([xyz_feat, state_feat], dim=-1)  # [BTo, self.n_output_channels]

        ret = {}
        ret['final_feat'] = final_feat
        ret['points_center'] = points_center
        ret['est_quat'] = est_quat

        return ret

    def output_shape(self):
        return self.n_output_channels