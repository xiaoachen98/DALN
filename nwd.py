import torch
import torch.nn as nn
import torch.nn.functional as F
from grl import WarmStartGradientReverseLayer


class NuclearWassersteinDiscrepancy(nn.Module):
    def __init__(self, classifier: nn.Module):
        super(NuclearWassersteinDiscrepancy, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.classifier = classifier

    @staticmethod
    def n_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        pre_s, pre_t = F.softmax(y_s, dim=1), F.softmax(y_t, dim=1)
        loss = (-torch.norm(pre_t, 'nuc') + torch.norm(pre_s, 'nuc')) / y_t.shape[0]
        return loss

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f_grl = self.grl(f)
        y = self.classifier(f_grl)
        y_s, y_t = y.chunk(2, dim=0)

        loss = self.n_discrepancy(y_s, y_t)
        return loss
