from torch import nn



class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, d_model, num_heads, dff, drop_prob, activation):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=drop_prob)
        self.dffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=dff),
            activation,
            nn.Linear(in_features=dff, out_features=d_model)
        )
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, x):
        x = self.norm1(x + self.attention(x, x, x)[0])
        x = self.norm2(x + self.dropout(self.dffn(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, d_model, num_heads, dff, num_layers, drop_prob, activation='relu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.activation = nn.ReLU()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, d_model, num_heads, dff, drop_prob, self.activation)
            if i == 0 else
            TransformerEncoderLayer(d_model, d_model, num_heads, dff, drop_prob, self.activation)
            for i in range(num_layers)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x