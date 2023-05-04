from dataclasses import dataclass
import torch
from torch import nn
from .feature_encoder import FeatureEncoder
from .transformer_encoder import TransformerEncoder
from wav2vec2.modules import (
    RelativePositionalEmbedding,
    ProductQuantizer
)

@dataclass
class Wav2Vec2ModelConfig:
    fe_n_blocks=7,
    fe_dim=512,
    fe_kernel_sizes=[10,3,3,3,3,2,2],
    fe_strides=[5,2,2,2,2,2,2],
    p=0.065,
    with_mask=True,
    mask_span=10,
    drop_prob=0.05,
    rpe_kernel_size=128,
    rpe_groups=16,
    qt_n_groups=2,
    qt_n_entries=320,
    final_dim=768,
    temperature=0.5,
    tfe_dff=3072,
    tfe_num_heads=8,
    tfe_num_layers=12,
    tfe_activation='relu',
    activation='gelu'


class Wav2Vec2Model(nn.Module):
    def __init__(
        self,
        fe_n_blocks,
        fe_dim,
        fe_kernel_sizes,
        fe_strides,
        p,
        with_mask,
        mask_span,
        drop_prob,
        rpe_kernel_size,
        rpe_groups,
        qt_n_groups,
        qt_n_entries,
        final_dim,
        temperature,
        tfe_dff,
        tfe_num_heads,
        tfe_num_layers,
        tfe_activation='relu',
        activation='gelu'
    ):
        super().__init__()
        self.with_mask = with_mask
        self.mask_span = mask_span
        self.drop_prob = drop_prob
        
        self.feature_encoder = FeatureEncoder(
            n_blocks=fe_n_blocks,
            dim=fe_dim,
            kernel_sizes=fe_kernel_sizes,
            strides=fe_strides,
            p=p,
            with_mask=with_mask,
            mask_span=mask_span,
            drop_prob=drop_prob,
            activation=activation
        )
        
        self.post_extract_proj = nn.Linear(in_features=fe_dim, out_features=final_dim) if fe_dim != final_dim else None
        
        self.positional_embedding = RelativePositionalEmbedding(
            embed_dim=fe_dim if fe_dim == final_dim else final_dim,
            kernel_size=rpe_kernel_size,
            padding=rpe_kernel_size // 2,
            groups=rpe_groups
        )
        
        self.quantizer = ProductQuantizer(
            z_dim=fe_dim,
            n_groups=qt_n_groups,
            n_entries=qt_n_entries,
            q_dim=final_dim,
            temperature=temperature
        )
        
        self.transformer_encoder = TransformerEncoder(
            embed_dim=final_dim,
            d_model=final_dim,
            num_heads=tfe_num_heads,
            dff=tfe_dff,
            num_layers=tfe_num_layers,
            drop_prob=drop_prob,
            activation=tfe_activation
        )
        
        self.norm = nn.LayerNorm(fe_dim if fe_dim == final_dim else final_dim)
        
    def forward(self, src):
        if self.with_mask:
            result = self.feature_encoder(src)
            features = result['masked_x']
            unmasked_features = result['x']
            mask = result['mask']
        else:
            features = self.feature_encoder(src)
            unmasked_features = None
        
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features.transpose(-2, -1)).transpose(-2, -1)
        
        #print(f"features shape: {features.shape}")
        pos_emb = self.positional_embedding(features).transpose(-2, -1)
        x = self.norm(features.transpose(-2, -1) + pos_emb) # (B, T, C)
        quantizer_result = self.quantizer(unmasked_features if unmasked_features is not None else features)
        #y = quantizer
        
        x = self.transformer_encoder(x) # (B, T, C)
        #y = y.transpose(-2, -1)
        mask = mask.transpose(-2, -1)
        
        return {
            "x": x,
            "y": quantizer_result["q"],
            "mask": mask,
            "codebook_logits": quantizer_result["codebook_logits"]
        }