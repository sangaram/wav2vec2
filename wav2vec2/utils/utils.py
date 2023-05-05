import torch
from torch import nn

class TransposeLastDim(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.transpose(-2, -1)
    

def resize_audio(audio, max_size):
    assert len(audio.shape) == 2, f"Error: audio must have shape (n_channels, audio_size) but got {audio.shape}"
    size = audio.size(1) # Assuming audio is of shape (n_channels, audio_size)
    diff = max_size - size
    if diff <= 0:
        return audio[:, :max_size]
    else:
        # padding with silence
        start_pad_len = torch.randint(0, diff, (1,)).item()
        end_pad_len = diff - start_pad_len
        n_channels = audio.size(0)
        start_pad = torch.zeros((n_channels, start_pad_len), dtype=audio.dtype)
        end_pad = torch.zeros((n_channels, end_pad_len), dtype=audio.dtype)
        result = torch.cat((start_pad, audio, end_pad), dim=1)
        return result