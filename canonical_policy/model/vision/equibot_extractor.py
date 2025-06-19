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
from canonical_policy.model.vision.canonical_utils.vec_pointnet import VecPointNet
from canonical_policy.model.common.rotation_transformer import RotationTransformer

class EquibotEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 equibot_encoder_cfg: Optional[Dict],
                 use_pc_color=False,
                 pointnet_type='equibot',
                 state_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_keys = state_keys
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_size = sum([observation_space[key][0] for key in self.state_keys])
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None

        cprint(f"[EquibotEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[EquibotEncoder] state shape: {self.state_size}", "yellow")
        cprint(f"[EquibotEncoder] imagination point shape: {self.imagination_shape}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        self.num_points = equibot_encoder_cfg.num_points
        self.k = equibot_encoder_cfg.neighbor_num
        self.rot_hidden_dim = equibot_encoder_cfg.rot_hidden_dim
        self.rot_layers = equibot_encoder_cfg.rot_layers

        if pointnet_type == "equibot":
            self.extractor = VecPointNet(h_dim=self.rot_hidden_dim, c_dim=self.rot_hidden_dim, num_layers=self.rot_layers, knn=self.k)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        self.n_output_channels = self.rot_hidden_dim + 3  # rot_hidden_dim + 3 (pos, dir1, dir2)

        cprint(f"[EquibotEncoder] output dim: {self.n_output_channels}", "red")

        self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')

    def get6DRotation(self, quat):
        return self.quaternion_to_sixd.forward(quat)  # wijk
    
    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]  # [B*To, N, 6]
        # sampling
        points, _ = sample_farthest_points(points, K=self.num_points)  # [BTo, num_samples, 6]
        points_xyz = points[..., :3]    # [BTo, num_samples, 3]
        BTo, N = points.shape[:2]
        To=2

        # extract state
        state_pos = observations[self.state_keys[0]]  # [BTo, 3]
        state_quat = observations[self.state_keys[1]]  # [BTo, 4]  ijkw
        state_quat = state_quat[:, [3, 0, 1, 2]]  # [BTo, 4]  wijk
        state_gripper = observations[self.state_keys[2]]  # [BTo, 2]

        if self.pointnet_type == "equibot":
            state_sixd = self.get6DRotation(state_quat)  # [BTo, 6]
            state_dir1 = state_sixd[:, :3]  # [BTo, 3]
            state_dir2 = state_sixd[:, 3:]  # [BTo, 3]

            # Parameter Normalization
            points_center = torch.mean(points_xyz, dim=1, keepdim=True)  # [BTo, 1, 3]
            points_shift = points_xyz - points_center  # [BTo, N, 3]
            points_scale = points_shift.norm(dim=-1).mean(-1)  # [BTo]
            input_pcl = points_shift / points_scale[:, None, None]  # [BTo, N, 3]

            points_center = points_center.reshape(BTo//To, To, 3)[:, [-1]].repeat(1, To, 1).reshape(BTo, 1, 3)
            points_scale = points_scale.reshape(BTo//To, To, 1)[:, [-1]].repeat(1, To, 1).reshape(BTo)

            state_pos = (state_pos - points_center.squeeze(1)) / points_scale[:, None]  # [BTo, 3]

            state_vec = torch.cat((state_pos.unsqueeze(1), state_dir1.unsqueeze(1), state_dir2.unsqueeze(1)), dim=1)  # [BTo, 3, 3]

            z, _ = self.extractor(input_pcl.transpose(1, 2)) # [BTo, 32, 3] 
            obs_cond_vec = torch.cat((z, state_vec), dim=1)  # [BTo, 32 + 3, 3]
            obs_cond_vec = obs_cond_vec.reshape(BTo//2, -1, 3)  # [B, 2 * (32 + 3), 3]

            obs_cond_scalar = state_gripper.reshape(BTo//To, -1)  # [B, 2 * 2]

            ret = {}
            ret['obs_cond_vec'] = obs_cond_vec
            ret['obs_cond_scalar'] = obs_cond_scalar
            ret['points_scale'] = points_scale
            ret['points_center'] = points_center

        else:
            raise NotImplementedError(f"pointnet_type: {self.pointnet_type}")

        return ret

    def output_shape(self):
        return self.n_output_channels