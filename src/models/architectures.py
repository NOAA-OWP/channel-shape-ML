import torch
import torch.nn as nn
import torch.nn.functional as F


class PiecewiseLinearEncoding(nn.Module):
    """Piecewise Linear Quantile Binning Encoding Layer."""
    def __init__(self, in_features: int, num_bins: int = 16):
        super().__init__()
        self.in_features = in_features
        self.num_bins = num_bins
        bins = torch.linspace(-3.0, 3.0, num_bins + 1).repeat(in_features, 1)
        self.register_buffer("bins", bins)

    def forward(self, x):
        x_exp = x.unsqueeze(-1)
        b_left = self.bins[:, :-1]
        b_right = self.bins[:, 1:]
        denom = torch.clamp(b_right - b_left, min=1e-5)
        enc = torch.clamp((x_exp - b_left) / denom, min=0.0, max=1.0)
        return enc.flatten(start_dim=1)


class CrossFeatureInteraction(nn.Module):
    """Explicit Pairwise Cross Feature Interaction Layer."""
    def __init__(self, in_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(in_features))

    def forward(self, x):
        interaction = torch.matmul(x, self.weight) + self.bias
        return x * interaction + x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SqueezeAndExcitation1D(nn.Module):
    """Channel/Feature Recalibration Attention Module."""
    def __init__(self, dim: int, reduction_ratio: int = 4):
        super().__init__()
        reduced_dim = max(16, dim // reduction_ratio)
        self.fc1 = nn.Linear(dim, reduced_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(reduced_dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weights = self.sigmoid(self.fc2(self.act(self.fc1(x))))
        return x * weights


class ModernResTabBlockV4(nn.Module):
    """SwiGLU + SE + RMSNorm Block."""
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15, se_ratio: int = 4):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.fc_in = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout_rate)
        self.se = SqueezeAndExcitation1D(hidden_dim, reduction_ratio=se_ratio)
        self.gamma = nn.Parameter(torch.full((hidden_dim,), 0.1))

    def forward(self, x):
        residual = x
        out = self.norm(x)
        x_proj = self.fc_in(out)
        x1, x2 = x_proj.chunk(2, dim=-1)
        out = F.silu(x1) * x2
        out = self.drop(self.fc_out(out))
        out = self.se(out)
        return residual + (self.gamma * out)


class PhysicalResTabNet(nn.Module):
    """Physical-Scale Aware Tabular Neural Network Architecture."""
    def __init__(
        self, 
        in_features: int, 
        hidden_dim: int = 256, 
        num_blocks: int = 4, 
        dropout_rate: float = 0.15, 
        head_hidden_dims: list = None,
        se_ratio: int = 4,
        num_bins: int = 16
    ):
        super().__init__()
        head_hidden_dims = head_hidden_dims or [128, 64]
        
        self.ple = PiecewiseLinearEncoding(in_features, num_bins=num_bins)
        self.cross = CrossFeatureInteraction(in_features)
        
        ple_out_dim = in_features * num_bins
        combined_in_dim = ple_out_dim + in_features
        
        self.in_proj = nn.Linear(combined_in_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            ModernResTabBlockV4(hidden_dim, dropout_rate=dropout_rate, se_ratio=se_ratio) 
            for _ in range(num_blocks)
        ])
        
        head_in_dim = hidden_dim + in_features
        head_layers = [RMSNorm(head_in_dim)]
        curr_dim = head_in_dim
        
        for h_dim in head_hidden_dims:
            head_layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.SiLU(),
                nn.Dropout(dropout_rate)
            ])
            curr_dim = h_dim
        head_layers.append(nn.Linear(curr_dim, 1))
        
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        raw_x = x
        ple_feat = self.ple(x)
        cross_feat = self.cross(x)
        
        combined = torch.cat([ple_feat, cross_feat], dim=-1)
        out = self.in_proj(combined)
        
        for block in self.blocks:
            out = block(out)
            
        out = torch.cat([out, raw_x], dim=-1)
        return self.head(out).squeeze(-1)