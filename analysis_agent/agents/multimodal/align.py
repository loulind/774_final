"""Learned alignment of two frozen feature sets (from Project 3: models/heads.py, the InfoNCE
loss of models/losses.py and the training loop of models/train.py, reduced to one loss).

One projection head per modality maps frozen features into a shared space; the heads
are trained with the symmetric InfoNCE (CLIP) loss and a learned temperature, capped at
100 as in CLIP. Residual heads (input and output of equal size) start as the identity,
so on a pre-aligned dual encoder training starts from the pretrained space, which is
then also a candidate (epoch 0) and is kept unless training beats it on validation
(the idea of CLIP-Adapter). Each epoch is scored by validation recall@10 on a fixed
subsample; the best epoch's weights are kept; training stops on patience, the epoch cap
or the time cap. Test pairs are never seen here.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .metrics import recall_at_k

VAL_CAP = 2000


class MLP(nn.Module):
    """Linear -> LayerNorm -> GELU -> Dropout blocks, then a final Linear (Project 3's head)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, n_layers: int = 2,
                 dropout: float = 0.1, residual: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.residual = residual and in_dim == out_dim
        if self.residual:                      # identity at initialisation
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return x + y if self.residual else y


def info_nce(za: torch.Tensor, zb: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logits = scale * F.normalize(za, dim=-1) @ F.normalize(zb, dim=-1).T
    y = torch.arange(len(za), device=za.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


class ContrastiveAligner:
    def __init__(self, in_a: int, in_b: int, embed_dim: int = 256, hidden: int = 512, n_layers: int = 2,
                 dropout: float = 0.1, residual: bool = False, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 256, max_epochs: int = 50, patience: int = 5, seconds: float = 60.0,
                 temperature_init: float = 0.07, seed: int = 0, device: str = "cpu"):
        torch.manual_seed(seed)
        dim = in_a if residual and in_a == in_b else embed_dim
        self.residual = residual and in_a == in_b
        self.heads = nn.ModuleDict({"a": MLP(in_a, dim, hidden, n_layers, dropout, self.residual),
                                    "b": MLP(in_b, dim, hidden, n_layers, dropout, self.residual)}).to(device)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature_init), device=device))
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.max_epochs, self.patience, self.seconds = max_epochs, patience, seconds
        self.seed, self.device = seed, device
        self.history: dict = {}

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    @torch.no_grad()
    def embed(self, which: str, X: np.ndarray, batch: int = 4096) -> np.ndarray:
        self.heads.eval()
        out = [F.normalize(self.heads[which](torch.as_tensor(X[s:s + batch], dtype=torch.float32,
                                                             device=self.device)), dim=-1).cpu().numpy()
               for s in range(0, len(X), batch)]
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)

    def _val_score(self, Xa, Xb) -> float:
        return recall_at_k(self.embed("a", Xa), self.embed("b", Xb), 10)

    def fit(self, Xa, Xb, Xa_val, Xb_val) -> ContrastiveAligner:
        gen = torch.Generator().manual_seed(self.seed)
        A = torch.as_tensor(Xa, dtype=torch.float32, device=self.device)
        B = torch.as_tensor(Xb, dtype=torch.float32, device=self.device)
        rv = np.random.default_rng(self.seed).permutation(len(Xa_val))[:VAL_CAP]
        va_a, va_b = Xa_val[rv], Xb_val[rv]
        opt = torch.optim.AdamW([{"params": self.heads.parameters(), "weight_decay": self.weight_decay},
                                 {"params": [self.logit_scale], "weight_decay": 0.0}], lr=self.lr)
        n, bs = len(A), int(min(self.batch_size, len(A)))
        init = self._val_score(va_a, va_b)
        best, best_epoch, stale, best_state = -np.inf, 0, 0, None
        if self.residual:                      # the pretrained space is a candidate itself
            best, best_state = init, copy.deepcopy(self.heads.state_dict())
        hist = {"initial_val_recall_at_10": init, "val_recall_at_10": [], "train_loss": []}
        start, why = time.perf_counter(), "epoch cap"
        for epoch in range(1, self.max_epochs + 1):
            self.heads.train()
            perm = torch.randperm(n, generator=gen).to(self.device)
            total, steps = 0.0, 0
            for s in range(0, n - bs + 1 if n >= 2 * bs else n, bs):
                idx = perm[s:s + bs]
                loss = info_nce(self.heads["a"](A[idx]), self.heads["b"](B[idx]), self.scale())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.heads.parameters()) + [self.logit_scale], 1.0)
                opt.step()
                total, steps = total + float(loss.detach()), steps + 1
            sc = self._val_score(va_a, va_b)
            hist["train_loss"].append(round(total / max(1, steps), 5))
            hist["val_recall_at_10"].append(round(sc, 5))
            if sc > best + 1e-6:
                best, best_epoch, stale = sc, epoch, 0
                best_state = copy.deepcopy(self.heads.state_dict())
            else:
                stale += 1
            if stale >= self.patience:
                why = f"no validation improvement for {self.patience} epochs"
                break
            if time.perf_counter() - start > self.seconds:
                why = f"time cap ({self.seconds:.0f}s)"
                break
        if best_state is not None:
            self.heads.load_state_dict(best_state)
        self.history = {**hist, "best_epoch": best_epoch, "best_val_recall_at_10": best,
                        "epochs_run": len(hist["train_loss"]), "stopped_because": why,
                        "seconds": round(time.perf_counter() - start, 1),
                        "temperature": float(1 / self.scale().detach())}
        return self
