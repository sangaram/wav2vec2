import torch
from torch.utils.data import DataLoader
from wav2vec2.data import LibriSpeechDatasetWrapper


dataset = LibriSpeechDatasetWrapper(
    root="..",
    max_sample_size=200
)

dataloader = DataLoader(
    dataset,
    batch_size=20,
    shuffle=False
)

batch = next(iter(dataloader))
print(f"Batch of shape: {batch.shape}")

"""
from wav2vec2.models import Wav2Vec2Model

model = Wav2Vec2Model(
    fe_n_blocks=3,
    fe_dim=16,
    fe_kernel_sizes=[4, 4, 2],
    fe_strides=[1, 1, 1],
    p=0.063,
    with_mask=True,
    mask_span=4,
    drop_prob=0.05,
    rpe_kernel_size=4,
    rpe_groups=4,
    qt_n_groups=4,
    qt_n_entries=16,
    final_dim=16,
    temperature=1.3,
    tfe_dff=32,
    tfe_num_heads=2,
    tfe_num_layers=4,
    tfe_activation='relu',
    activation='gelu'
)

print(model.__dict__)
"""