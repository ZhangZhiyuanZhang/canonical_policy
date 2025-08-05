import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def get_activation(activation):
    activations = {
        'gelu': nn.GELU(),
        'rrelu': nn.RReLU(inplace=True),
        'selu': nn.SELU(inplace=True),
        'silu': nn.SiLU(inplace=True),
        'hardswish': nn.Hardswish(inplace=True),
        'leakyrelu': nn.LeakyReLU(inplace=True),
        'leakyrelu0.2': nn.LeakyReLU(negative_slope=0.2, inplace=True),
    }
    return activations.get(activation.lower(), nn.ReLU(inplace=True))

def square_distance(src, dst):
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, -1).unsqueeze(1)
    return dist

def index_points(points, idx):
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(-1, 1, 1)
    return points[batch_indices, idx, :]

def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

class FeatureAggregation(nn.Module):
    def __init__(self, channel, kneighbors, normalize="anchor"):
        super(FeatureAggregation, self).__init__()
        self.kneighbors = kneighbors
        self.normalize = normalize
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel]))

    def forward(self, xyz, points):
        """
        x: [B, G, 3]
        points: [B, G, D]
        """
        B, G = xyz.shape[:2]
        idx = knn_point(self.kneighbors, xyz, xyz)
        grouped_points = index_points(points, idx)  # [B, G, K, D]
        mean = points.unsqueeze(dim=-2) if self.normalize == "anchor" else torch.mean(grouped_points, dim=2, keepdim=True)  # [B, G, 1, D]
        std = torch.std((grouped_points - mean).reshape(grouped_points.size(0), -1), dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        grouped_points = (grouped_points - mean) / (std + 1e-5)
        grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, points.view(B, G, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_points  # [B, G, K, 2D]

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.mlp(x))

class AggEncoder(nn.Module):
    def __init__(self, hidden_dim=64, ksize=10):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.local_grouper = FeatureAggregation(
            channel=hidden_dim, kneighbors=ksize, normalize='anchor'
        )

        self.pre_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 4 * hidden_dim),
            nn.LayerNorm(4 * hidden_dim),
        )
        self.pre_block = ResidualMLPBlock(4 * hidden_dim, 8 * hidden_dim)

        self.pos_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, 4 * hidden_dim),
            nn.LayerNorm(4 * hidden_dim),
        )
        self.pos_block = ResidualMLPBlock(4 * hidden_dim, 8 * hidden_dim)

        self.final_projection = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        B, N = x.shape[:2]
        xyz = x.clone()

        x = self.embedding(x)                     # [B, G, D]
        x = self.local_grouper(xyz, x)            # [B, G, K, 2D]
        x = rearrange(x, 'b g k d -> (b g) k d')  # [BG, K, 2D]

        x = self.pre_mlp(x)                       # [BG, K, 4D]
        x = self.pre_block(x)                     # [BG, K, 4D]

        x = x.transpose(1, 2)                     # [BG, 4D, K]
        x = F.adaptive_max_pool1d(x, 1).reshape(B, N, -1)   # [B, G, 4D]

        x = self.pos_mlp(x)                       # [B, G, 4D]
        x = self.pos_block(x)                     # [B, G, 4D]

        x = torch.max(x, dim=1)[0]                # [B, 4D]
        x = self.final_projection(x)              # [B, D]

        return x
