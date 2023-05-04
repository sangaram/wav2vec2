import torch
from wav2vec2.models import Wav2Vec2Model

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
print(optimizer.param_groups)