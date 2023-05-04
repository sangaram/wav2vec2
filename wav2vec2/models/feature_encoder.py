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
        self.conv_bloc = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride),
            nn.Dropout(p=drop_prob),
            TransposeLastDim(),
            nn.LayerNorm(normalized_shape=out_channel),
            TransposeLastDim(),
            activation
        )
        
    def forward(self, x):
        return self.conv_bloc(x)

class FeatureEncoder(nn.Module):
    def __init__(
        self,
        n_blocks,
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
        self.p = p
        self.with_mask = with_mask
        self.mask_span = mask_span
        self.mask_vector = nn.Parameter(torch.FloatTensor(dim).uniform_()) if with_mask else None
        self.drop_prob = drop_prob
        self.activation = nn.GELU()
        self.conv_blocks = nn.ModuleList()
        assert len(kernel_sizes) == n_blocks, "Invalid arguments: block_kernel_sizes must have length n_blocks"
        assert len(strides) == n_blocks, "Invalid arguments: block_strides must have length n_blocks"
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
            x = x.unsqueeze(1)
        
        B, _, T = x.shape
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        if self.with_mask:
            with torch.no_grad():
                num_samples = int(self.p*T)
                indices = torch.arange(T).float()
                start_ids = torch.multinomial(indices, num_samples).to(torch.long)
                masked_x = x.clone()
                mask = torch.full(x.shape, False)
                for idx in start_ids:
                    masked_x[:,:,idx:(idx+self.mask_span)%T] = 0
                    masked_x[:,:,idx:(idx+self.mask_span)%T] += self.mask_vector.unsqueeze(0).unsqueeze(-1)
                    mask[:,:,idx:(idx+self.mask_span)%T] = True

                result = {
                    "x": x,
                    "masked_x": masked_x,
                    "mask": mask
                }
            
            return result
        else:
            return x