from torch import nn
from wav2vec2.utils import TransposeLastDim

class LengthCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x[:, :, :-1]

class RelativePositionalEmbedding(nn.Module):
    """ A CNN used as relative positional embedding
    """
    def __init__(self, embed_dim, kernel_size, padding, groups):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, padding=padding, groups=groups),
            LengthCorrection(),
            TransposeLastDim(),
            nn.LayerNorm(normalized_shape=embed_dim),
            TransposeLastDim(),
            nn.GELU()
        )
    
    def forward(self, x):
        # x has shape (B, C, T)
        x = self.conv(x)
        return x