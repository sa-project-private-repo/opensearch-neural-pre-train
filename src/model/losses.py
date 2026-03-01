"""
Loss functions for SPLADE V33 neural sparse training.

Includes:
- SPLADELossV33: InfoNCE + FLOPS regularization with quadratic warmup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SPLADELossV33(nn.Module):
    """
    V33 SPLADE loss: clean implementation following SPLADE v2 paper.

    L = L_ranking + lambda_q * L_FLOPS_q + lambda_d * L_FLOPS_d

    Components:
    - InfoNCE with in-batch negatives + explicit hard negatives
    - FLOPS regularization (Eq. 4): sum_j(mean_i(w_j^i))^2
    - Quadratic lambda scheduler (0 -> target over warmup_steps)
    - Optional KD from cross-encoder teacher

    No language filtering, no stopword masking, no positive activation.
    """

    def __init__(
        self,
        lambda_q: float = 1e-2,
        lambda_d: float = 3e-3,
        temperature: float = 1.0,
        flops_warmup_steps: int = 20000,
        lambda_kd: float = 0.0,
        kd_temperature: float = 1.0,
        lambda_initial_ratio: float = 0.1,
    ):
        super().__init__()
        self.lambda_q = lambda_q
        self.lambda_d = lambda_d
        self.temperature = temperature
        self.flops_warmup_steps = flops_warmup_steps
        self.lambda_kd = lambda_kd
        self.kd_temperature = kd_temperature
        self.lambda_initial_ratio = lambda_initial_ratio

        # Monitoring buffers
        self._avg_nonzero_q = 0.0
        self._avg_nonzero_d = 0.0
        self._count = 0

    def _flops_loss(self, sparse_repr: torch.Tensor) -> torch.Tensor:
        """
        FLOPS regularization (SPLADE v2, Eq. 4).

        L_FLOPS = sum_j (1/N * sum_i w_j^i)^2

        Penalizes tokens with high average activation across the batch,
        encouraging sparse, discriminative representations.

        Args:
            sparse_repr: [batch, vocab_size]

        Returns:
            Scalar FLOPS loss
        """
        mean_activation = sparse_repr.mean(dim=0)  # [vocab]
        return (mean_activation ** 2).sum()

    def _lambda_schedule(
        self, step: int, target_lambda: float
    ) -> float:
        """
        Quadratic lambda warmup scheduler with initial floor.

        lambda(t) = target * (r0 + (1 - r0) * min(1, (t / T)^2))

        Starts at r0 * target for immediate sparsity pressure,
        then quadratically warms up to target.
        """
        r0 = self.lambda_initial_ratio
        if step >= self.flops_warmup_steps:
            return target_lambda
        ratio = step / max(self.flops_warmup_steps, 1)
        return target_lambda * (r0 + (1.0 - r0) * ratio * ratio)

    def _infonce_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE with in-batch negatives + explicit hard negative.

        Args:
            anchor: [batch, vocab] query representations
            positive: [batch, vocab] positive doc representations
            negative: [batch, vocab] hard negative representations
        """
        batch_size = anchor.shape[0]

        # Positive scores: [batch]
        pos_scores = (anchor * positive).sum(dim=-1) / self.temperature

        # In-batch negative scores: [batch, batch]
        neg_scores = torch.mm(anchor, positive.t()) / self.temperature

        # Explicit hard negative scores: [batch]
        hard_neg_scores = (
            (anchor * negative).sum(dim=-1) / self.temperature
        )

        # Labels: diagonal is positive
        labels = torch.arange(
            batch_size, device=anchor.device
        )

        # Combine: [batch, batch + 1]
        # Column 0..batch-1: in-batch, column batch: hard negative
        all_scores = torch.cat(
            [neg_scores, hard_neg_scores.unsqueeze(1)], dim=1
        )

        return F.cross_entropy(all_scores, labels)

    def forward(
        self,
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_repr: torch.Tensor,
        global_step: int = 0,
        teacher_scores: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute V33 loss.

        Args:
            anchor_repr: [batch, vocab] query sparse representations
            positive_repr: [batch, vocab] positive doc representations
            negative_repr: [batch, vocab] hard negative representations
            global_step: current training step (for lambda scheduler)
            teacher_scores: [batch, batch] optional KD scores

        Returns:
            (total_loss, loss_dict)
        """
        # InfoNCE
        infonce = self._infonce_loss(
            anchor_repr, positive_repr, negative_repr
        )

        # FLOPS regularization with scheduled lambda
        flops_q = self._flops_loss(anchor_repr)
        flops_d = self._flops_loss(positive_repr)

        cur_lambda_q = self._lambda_schedule(global_step, self.lambda_q)
        cur_lambda_d = self._lambda_schedule(global_step, self.lambda_d)

        loss = infonce + cur_lambda_q * flops_q + cur_lambda_d * flops_d

        # Optional KD
        kd_loss = torch.tensor(0.0, device=loss.device)
        if self.lambda_kd > 0 and teacher_scores is not None:
            student_scores = (
                torch.mm(anchor_repr, positive_repr.t())
                / self.kd_temperature
            )
            teacher_probs = F.softmax(
                teacher_scores / self.kd_temperature, dim=-1
            )
            student_log_probs = F.log_softmax(student_scores, dim=-1)
            kd_loss = F.kl_div(
                student_log_probs, teacher_probs, reduction="batchmean"
            )
            loss = loss + self.lambda_kd * kd_loss

        # Monitoring
        with torch.no_grad():
            nonzero_q = (anchor_repr > 0).float().sum(dim=-1).mean()
            nonzero_d = (positive_repr > 0).float().sum(dim=-1).mean()
            self._avg_nonzero_q = (
                0.9 * self._avg_nonzero_q + 0.1 * nonzero_q.item()
            )
            self._avg_nonzero_d = (
                0.9 * self._avg_nonzero_d + 0.1 * nonzero_d.item()
            )
            self._count += 1

        loss_dict = {
            "infonce": infonce.item(),
            "flops_q": flops_q.item(),
            "flops_d": flops_d.item(),
            "lambda_q": cur_lambda_q,
            "lambda_d": cur_lambda_d,
            "kd": kd_loss.item(),
            "nonzero_q": nonzero_q.item(),
            "nonzero_d": nonzero_d.item(),
        }

        return loss, loss_dict

    def get_avg_nonzero(self) -> Tuple[float, float]:
        """Return EMA of nonzero activations (query, doc)."""
        return self._avg_nonzero_q, self._avg_nonzero_d
