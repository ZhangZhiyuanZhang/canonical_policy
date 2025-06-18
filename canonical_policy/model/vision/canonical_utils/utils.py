import torch
import torch.nn.functional as F

def construct_rotation_matrix(v1, v2):
    # Normalize the first vector
    v1_norm = F.normalize(v1, dim=-1)  # [BTo, 3]

    # Orthogonalize and normalize the second vector
    v2_proj = torch.sum(v2 * v1_norm, dim=-1, keepdim=True) * v1_norm
    v2_orthogonal = v2 - v2_proj
    v2_norm = F.normalize(v2_orthogonal, dim=-1)  # [BTo, 3]

    # Compute the third vector using cross product
    v3 = torch.cross(v1_norm, v2_norm, dim=-1)  # [BTo, 3]

    # Stack vectors to form the rotation matrix
    rot_mat = torch.stack([v1_norm, v2_norm, v3], dim=-1)  # [BTo, 3, 3]

    return rot_mat

def extract_z_rotation(rot_mat):
    """
    Extracts the Z-axis rotation component from a batch of 3D rotation matrices.
    :param rot_mat: (B, 3, 3) Batch of rotation matrices
    :return: (B, 3, 3) Batch of rotation matrices containing only Z-axis rotation
    """
    # Extract Z-axis rotation angle
    theta = torch.atan2(rot_mat[:, 1, 0], rot_mat[:, 0, 0])  # (B,)

    # Compute cos and sin
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Construct new Z-axis rotation matrices (B, 3, 3)
    rot_mat_z = torch.zeros_like(rot_mat)
    rot_mat_z[:, 0, 0] = cos_theta
    rot_mat_z[:, 0, 1] = -sin_theta
    rot_mat_z[:, 1, 0] = sin_theta
    rot_mat_z[:, 1, 1] = cos_theta
    rot_mat_z[:, 2, 2] = 1

    return rot_mat_z