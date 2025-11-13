import torch
import torch.nn as nn
import torch.nn.functional as F


class OCConLossFn(nn.Module):
    def __init__(self, temperature: float = 0.07, device: str = "cpu"):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=self.device)
        mask = mask * logits_mask

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        denom = torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        log_prob = logits - denom

        sum_mask = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / torch.clamp(sum_mask, min=1.0)
        if sum_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return -mean_log_prob_pos[sum_mask > 0].mean()
