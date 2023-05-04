import torch
from torch import nn
import torch.nn.functional as F


class ProductQuantizer(nn.Module):
    def __init__(self, z_dim, n_groups, n_entries, q_dim, temperature=1.):
        super().__init__()
        self.z_dim = z_dim
        self.n_groups = n_groups
        self.n_entries = n_entries
        self.q_dim = q_dim
        assert q_dim % n_groups == 0, f"Invalid arguments: q_dim must be divisable by n_groups. Got z_dim={z_dim}, n_groups={n_groups}"
        self.v_dim = q_dim // n_groups
        self.temperature = temperature
        self.codevectors = nn.Parameter(torch.FloatTensor(1, n_groups*n_entries, self.v_dim))
        self.proj = nn.Linear(in_features=z_dim, out_features=n_groups*n_entries)
    
    def forward(self, x):
        # x has shape (B, C, T)
        x = x.transpose(-2, -1) # now x has shape (B, T, C)
        B, T, C = x.shape
        x = self.proj(x) # (B, T, n_groups*n_entries)
        
        result = {"codebook_logits": x.view(B, T, self.n_groups, -1)}
        x = x.view(B*T*self.n_groups, -1)
        x = F.gumbel_softmax(x.float(), tau=self.temperature, hard=True)
        x = x.view(B*T, -1).unsqueeze(-1)
        x = x * self.codevectors
        x = x.view(B*T, self.n_groups, self.n_entries, -1)
        x = x.sum(axis=-2)
        x = x.view(B, T, -1)
        result["q"] = x
        
        return result