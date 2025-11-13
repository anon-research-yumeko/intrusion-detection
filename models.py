import torch
import torch.nn as nn
import torch.nn.functional as F


class OCConNet(nn.Module):
    def __init__(self, input_dim, emb_dim, proj_dim=256):
        super().__init__()

        def _round_to_mult(value: float, mult: int = 8) -> int:
            return int(max(mult, mult * round(value / mult)))

        l1_est = (input_dim ** (2 / 3)) * (emb_dim ** (1 / 3))
        l2_est = (input_dim ** (1 / 3)) * (emb_dim ** (2 / 3))
        enc_l1 = _round_to_mult(l1_est)
        enc_l2 = _round_to_mult(l2_est)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_l1),
            nn.BatchNorm1d(enc_l1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(enc_l1, enc_l2),
            nn.BatchNorm1d(enc_l2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(enc_l2, emb_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, proj_dim),
        )

    def forward(self, x):
        emb = self.encoder(x)
        emb = F.normalize(emb, p=2, dim=1, eps=1e-12)
        proj = self.projection_head(emb)
        proj = F.normalize(proj, p=2, dim=1, eps=1e-12)
        return emb, proj
