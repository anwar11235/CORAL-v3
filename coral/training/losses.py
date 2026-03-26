"""Loss functions and ACT loss head for CORAL v3.

stablemax_cross_entropy is the default for small-sample experiments (1K).
It computes in float64 to avoid underflow on the stablemax denominator.

ACTLossHead wraps the CoralACT model and computes:
    total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

where:
    lm_loss         = per-sequence mean cross-entropy, summed over batch
    q_halt_loss     = BCE between q_halt_logits and sequence-level correctness
    q_continue_loss = BCE between q_continue_logits and bootstrapped target Q
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID: int = -100


# ---------------------------------------------------------------------------
# Stablemax
# ---------------------------------------------------------------------------


def _stablemax_s(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    """Stablemax transfer function: maps ℝ → (0, ∞).

    For x < 0: s(x) = 1 / (1 - x + ε)
    For x ≥ 0: s(x) = x + 1
    """
    return torch.where(x < 0, 1.0 / (1.0 - x + epsilon), x + 1.0)


def _log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log of stablemax distribution — computed in float64 for numerical stability."""
    s_x = _stablemax_s(x)
    return torch.log(s_x / s_x.sum(dim=dim, keepdim=True))


def stablemax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
) -> torch.Tensor:
    """Per-token stablemax cross-entropy loss (float64 computation).

    Args:
        logits: [B, seq_len, vocab_size] — raw scores.
        labels: [B, seq_len] — int token ids; positions with ignore_index are masked.

    Returns:
        Per-token losses [B, seq_len]; masked positions have loss = 0.
    """
    # Compute in float64 — stablemax denominator can underflow in lower precision
    logprobs = _log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    # Avoid out-of-bounds gather on masked positions by replacing -100 with 0
    safe_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs,
        index=safe_labels.to(torch.long).unsqueeze(-1),
        dim=-1,
    ).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, torch.zeros_like(prediction_logprobs))


def softmax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
) -> torch.Tensor:
    """Per-token standard cross-entropy loss (float32 computation).

    Args:
        logits: [B, seq_len, vocab_size].
        labels: [B, seq_len] int token ids.

    Returns:
        Per-token losses [B, seq_len]; masked positions have loss = 0.
    """
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.shape)


# ---------------------------------------------------------------------------
# ACT loss head
# ---------------------------------------------------------------------------


class ACTLossHead(nn.Module):
    """Wraps the CoralACT model and computes training loss + evaluation metrics.

    The loss combines:
        lm_loss  = per-sequence mean token loss, summed over batch
        q_halt   = BCE(q_halt_logits, sequence_is_correct)
        q_cont   = BCE(q_continue_logits, target_q_continue)   [training only]

        total = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

    Metrics are only accumulated for sequences that have halted
    (new_carry.halted == True) and have at least one valid label token.

    Args:
        model: A CoralACT instance.
        loss_type: One of "stablemax_cross_entropy" or "softmax_cross_entropy".
    """

    def __init__(self, model: nn.Module, loss_type: str) -> None:
        super().__init__()
        self.model = model
        _loss_fns = {
            "stablemax_cross_entropy": stablemax_cross_entropy,
            "softmax_cross_entropy": softmax_cross_entropy,
        }
        if loss_type not in _loss_fns:
            raise ValueError(f"Unknown loss_type: {loss_type!r}. Choose from {list(_loss_fns)}")
        self.loss_fn = _loss_fns[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore[operator]

    @property
    def puzzle_emb(self):
        return self.model.puzzle_emb  # type: ignore[attr-defined]

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """Run one ACT segment and compute loss + metrics.

        Args:
            return_keys: Keys from model outputs to include in detached_outputs.
            **model_kwargs: Forwarded to model.forward() (carry=..., batch=...).

        Returns:
            (new_carry, loss, metrics, detached_outputs, all_halted)
            - new_carry: updated ACTCarry
            - loss: scalar loss tensor (NOT yet divided by global_batch_size)
            - metrics: dict of scalar tensors for logging (not reduced)
            - detached_outputs: subset of model outputs requested by return_keys
            - all_halted: bool tensor scalar — True when every sequence has halted
        """
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID                        # [B, seq_len]
            loss_counts = mask.sum(-1)                              # [B]
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # [B, 1]

            preds_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)  # [B, seq_len]
            seq_is_correct = preds_correct.sum(-1) == loss_counts  # [B] bool

            # Metrics are only valid for halted sequences with at least one label
            valid_metrics = new_carry.halted & (loss_counts > 0)   # [B] bool
            metrics: Dict[str, torch.Tensor] = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (preds_correct.to(torch.float32) / loss_divisor).sum(-1),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(
                    valid_metrics,
                    new_carry.steps.to(torch.float32),
                    torch.zeros_like(new_carry.steps, dtype=torch.float32),
                ).sum(),
            }

        # ---- LM loss ----
        # Per-sequence: sum of per-token losses / number of valid tokens (mean)
        # Then summed across the batch (normalization by global_batch_size is caller's job)
        lm_loss = (
            self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor
        ).sum()

        # ---- Q-halt loss ----
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )

        metrics["lm_loss"] = lm_loss.detach()
        metrics["q_halt_loss"] = q_halt_loss.detach()

        # ---- Q-continue loss (bootstrapped target, only available during training) ----
        q_continue_loss: torch.Tensor = torch.tensor(0.0, device=lm_loss.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
