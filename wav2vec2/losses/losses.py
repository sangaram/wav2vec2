import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class ContrastiveLoss(_Loss):
    def __init__(self, temperature, n_distractors):
        super().__init__()
        self.temperature = temperature
        self.n_distractors = n_distractors
        
    def forward(self, x, y):
        B, T, C = y.shape
        
        with torch.no_grad():
            y = y.view(B*T, -1)

            distractor_ids = torch.randint(low=0, high=T, size=(B,  self.n_distractors*T))
            distractors = y[distractor_ids.view(-1)].view(B, T,  self.n_distractors, C)
            distractors = distractors.permute(2, 0, 1, 3)
            y = y.view(B, T, C)
            targets = torch.cat((y.unsqueeze(0), distractors), dim=0)
        
        sims = F.cosine_similarity(x.unsqueeze(0), targets, dim=-1)/self.temperature # similarities within distractors and the true target
        sim = F.cosine_similarity(x, y, dim=-1)/self.temperature # similarity with the true target
        
        loss = -(sim.exp() / sims.exp().sum(dim=0)).log().mean()
        
        return loss

class DiversityLoss(_Loss):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits):
        # logits should be of shape (B, T, G, V)
        dist = torch.mean(torch.softmax(logits, dim=-1), dim=0)
        loss = (dist*torch.log(dist + 1e-7)).mean()
        return loss

class PretrainingLoss(_Loss):
    def __init__(self, alpha, temperature, n_distractors):
        super().__init__()
        self.alpha = alpha
        self.contrastive_loss = ContrastiveLoss(temperature, n_distractors)
        self.diversity_loss = DiversityLoss()
        
    def forward(self, x, y, logits):
        return self.contrastive_loss(x, y) + self.alpha * self.diversity_loss(logits)