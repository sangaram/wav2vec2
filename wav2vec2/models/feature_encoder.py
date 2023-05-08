import torch
from torch import nn
from wav2vec2.utils import TransposeLastDim


class FeatureEncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, drop_prob, activation):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride),
            nn.Dropout(p=drop_prob),
            TransposeLastDim(),
            nn.LayerNorm(normalized_shape=out_channel),
            TransposeLastDim(),
            activation
        )
        
    def forward(self, x):
        return self.conv_block(x)

class FeatureEncoder(nn.Module):
    def __init__(
        self,
        dim,
        kernel_sizes,
        strides,
        p,
        with_mask,
        mask_span,
        drop_prob,
        activation='gelu'
    ):
        super().__init__()
        self.dim = dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.drop_prob = drop_prob
        self.activation = nn.GELU()
        self.conv_blocks = nn.ModuleList()
        in_channel = 1
        for (kernel_size, stride) in zip(kernel_sizes, strides):
            self.conv_blocks.append(
                FeatureEncoderBlock(
                    in_channel=in_channel,
                    out_channel=dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    drop_prob=drop_prob,
                    activation=self.activation
                )
            )
            in_channel = dim
        
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # adding the channel dimension
        
        B, _, T = x.shape
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            
        return x