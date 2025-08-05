import torch
from canonical_policy.model.vision.canonical_utils.utils import construct_rotation_matrix
from canonical_policy.model.vision.canonical_utils.vec_pointnet import VecPointNet, VN_Regressor

# encapulate the rotation matrix estimation
def get_canonical_rot(input_pcl, extractor, predictor):
    """
    input_pcl: [B, N, 3]
    """
    equiv_feat = extractor(input_pcl)   # [B, D, 3, N] mean pooling -> [B, D, 3]
    v1, v2 = predictor(equiv_feat)  # [B, 3], [B, 3]
    rot = construct_rotation_matrix(v1, v2)  # [B, 3, 3]
    return rot

# generate random SO3 rotation matrix
def random_rotation_matrix():
    q = torch.randn(4)
    q = q / q.norm()
    qw, qx, qy, qz = q
    return torch.tensor([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
    ])

# SO3-equivariant model
extractor = VecPointNet(h_dim=32, c_dim=32, num_layers=2, ksize=5)
predictor = VN_Regressor(pc_feat_dim=32)


# Generate random point cloud X ∈ ℝ^{N×3}
B = 1
N = 12
X = torch.rand(B, N, 3)

# X and Y are within the same equivariant group
R = random_rotation_matrix()
t = torch.randn(B, 1, 3)
Y = X @ R.T + t  # equal to (R @ X.T).T

# 1. Point Cloud Decenterization
X_decentered = X - X.mean(dim=1, keepdim=True)   # [B, N, 3]
Y_decentered = Y - Y.mean(dim=1, keepdim=True)   # [B, N, 3]

# 2. Compute inverse rotation matrix for each element
rot_X = get_canonical_rot(X_decentered, extractor, predictor)  # [B, 3, 3]
rot_Y = get_canonical_rot(Y_decentered, extractor, predictor)  # [B, 3, 3]

# 3. Transform to canonical space
X_canonical = X_decentered @ rot_X
Y_canonical = Y_decentered @ rot_Y

# Print results
print("X_canonical:", X_canonical)
print("Y_canonical:", Y_canonical)