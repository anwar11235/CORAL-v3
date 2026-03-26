"""Tests for CoralACT (Step 0.8a) and ACTLossHead (Step 0.7).

All tests use CPU + float32 (no flash_attn) where possible.
Tests that require a full forward pass are CUDA_ONLY.
"""

import pytest
import torch
import torch.nn as nn

from coral.models.coral_base import CoralConfig, InnerCarry
from coral.training.act import ACTCarry, CoralACT
from coral.training.losses import (
    IGNORE_LABEL_ID,
    ACTLossHead,
    stablemax_cross_entropy,
    softmax_cross_entropy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN = 64
NUM_HEADS = 4
VOCAB = 32
SEQ_LEN = 8
BATCH = 4

SMALL_CFG = CoralConfig(
    batch_size=BATCH,
    seq_len=SEQ_LEN,
    vocab_size=VOCAB,
    H_cycles=2,
    L_cycles=2,
    H_layers=1,
    L_layers=1,
    hidden_size=HIDDEN,
    num_heads=NUM_HEADS,
    expansion=2.0,
    halt_max_steps=4,
    halt_exploration_prob=0.1,
    forward_dtype="float32",
    puzzle_emb_ndim=0,
)

CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA + flash_attn",
)


def make_act(cfg_overrides=None) -> CoralACT:
    cfg = SMALL_CFG.model_copy(update=cfg_overrides or {})
    return CoralACT(cfg)


def make_batch(B=BATCH, seq=SEQ_LEN, vocab=VOCAB, device="cpu"):
    return {
        "inputs": torch.randint(0, vocab, (B, seq), device=device),
        "labels": torch.randint(0, vocab, (B, seq), device=device),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=device),
    }


# ---------------------------------------------------------------------------
# Loss functions (CPU)
# ---------------------------------------------------------------------------


class TestStablemaxCrossEntropy:
    def test_output_shape(self):
        logits = torch.randn(2, 8, 32)
        labels = torch.randint(0, 32, (2, 8))
        loss = stablemax_cross_entropy(logits, labels)
        assert loss.shape == (2, 8)

    def test_ignored_positions_zero(self):
        logits = torch.randn(2, 8, 32)
        labels = torch.full((2, 8), IGNORE_LABEL_ID)
        loss = stablemax_cross_entropy(logits, labels)
        assert (loss == 0).all()

    def test_valid_positions_positive(self):
        logits = torch.randn(2, 8, 32)
        labels = torch.randint(0, 32, (2, 8))
        loss = stablemax_cross_entropy(logits, labels)
        assert (loss >= 0).all()

    def test_mixed_mask(self):
        B, S, V = 2, 6, 10
        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))
        labels[:, 3:] = IGNORE_LABEL_ID   # mask last 3 tokens
        loss = stablemax_cross_entropy(logits, labels)
        assert (loss[:, 3:] == 0).all()
        assert (loss[:, :3] > 0).all()


class TestSoftmaxCrossEntropy:
    def test_output_shape(self):
        logits = torch.randn(2, 8, 32)
        labels = torch.randint(0, 32, (2, 8))
        loss = softmax_cross_entropy(logits, labels)
        assert loss.shape == (2, 8)

    def test_ignored_positions_zero(self):
        logits = torch.randn(2, 8, 32)
        labels = torch.full((2, 8), IGNORE_LABEL_ID)
        loss = softmax_cross_entropy(logits, labels)
        assert (loss == 0).all()


# ---------------------------------------------------------------------------
# CoralACT structural tests (CPU)
# ---------------------------------------------------------------------------


class TestCoralACTStructure:
    def test_initial_carry_all_halted(self):
        model = make_act()
        batch = make_batch()
        carry = model.initial_carry(batch)
        assert carry.halted.all(), "All sequences should start halted"

    def test_initial_carry_steps_zero(self):
        model = make_act()
        batch = make_batch()
        carry = model.initial_carry(batch)
        assert (carry.steps == 0).all()

    def test_initial_carry_types(self):
        model = make_act()
        batch = make_batch()
        carry = model.initial_carry(batch)
        assert isinstance(carry, ACTCarry)
        assert isinstance(carry.inner_carry, InnerCarry)
        assert carry.steps.dtype == torch.int32
        assert carry.halted.dtype == torch.bool

    def test_initial_carry_data_shape(self):
        model = make_act()
        batch = make_batch()
        carry = model.initial_carry(batch)
        assert carry.current_data["inputs"].shape == (BATCH, SEQ_LEN)

    def test_inner_is_coral_inner(self):
        from coral.models.coral_base import CoralInner
        model = make_act()
        assert isinstance(model.inner, CoralInner)


# ---------------------------------------------------------------------------
# CoralACT forward tests (CUDA required)
# ---------------------------------------------------------------------------


@CUDA_ONLY
class TestCoralACTForward:
    def _model_and_carry(self, training=True):
        cfg = SMALL_CFG.model_copy(update={"forward_dtype": "bfloat16"})
        model = CoralACT(cfg).cuda()
        model.train(training)
        batch = make_batch(device="cuda")
        carry = model.initial_carry(batch)
        carry = ACTCarry(
            inner_carry=model.inner.reset_carry(
                carry.halted,
                InnerCarry(
                    z_H=carry.inner_carry.z_H.cuda(),
                    z_L=carry.inner_carry.z_L.cuda(),
                ),
            ),
            steps=carry.steps.cuda(),
            halted=carry.halted.cuda(),
            current_data={k: v.cuda() for k, v in carry.current_data.items()},
        )
        return model, carry

    def test_carry_reset_for_halted(self):
        """Halted sequences should receive fresh batch data; non-halted keep their carry."""
        model, carry = self._model_and_carry()
        batch = make_batch(device="cuda")

        # Mark only first half as halted
        carry.halted[:] = False
        carry.halted[:BATCH // 2] = True

        # Poison the current_data with a distinguishable value
        carry.current_data["inputs"][:] = VOCAB - 1

        new_carry, _ = model(carry, batch)

        # Halted sequences should have batch["inputs"] in their carry data
        torch.testing.assert_close(
            new_carry.current_data["inputs"][:BATCH // 2],
            batch["inputs"][:BATCH // 2],
        )
        # Non-halted sequences keep old data (99)
        assert (new_carry.current_data["inputs"][BATCH // 2:] == 99).all()

    def test_steps_increment(self):
        """Steps should increment by 1 each call; halted sequences reset to 1."""
        model, carry = self._model_and_carry()
        batch = make_batch(device="cuda")

        # All halted → after forward, steps should be 1 for all
        new_carry, _ = model(carry, batch)
        assert (new_carry.steps == 1).all()

        # Not halted → steps should be 2 after next forward
        new_carry.halted[:] = False
        new_carry2, _ = model(new_carry, batch)
        assert (new_carry2.steps == 2).all()

    def test_halt_at_max_steps(self):
        """All sequences must halt at halt_max_steps regardless of Q-values."""
        model, carry = self._model_and_carry(training=False)  # eval mode
        batch = make_batch(device="cuda")

        carry.halted[:] = False
        carry.steps[:] = SMALL_CFG.halt_max_steps - 1  # one step before max

        new_carry, _ = model(carry, batch)
        assert new_carry.halted.all(), "All sequences should halt at halt_max_steps"

    def test_output_logits_shape(self):
        model, carry = self._model_and_carry()
        batch = make_batch(device="cuda")
        new_carry, outputs = model(carry, batch)
        assert outputs["logits"].shape == (BATCH, SEQ_LEN, VOCAB)

    def test_q_shapes(self):
        model, carry = self._model_and_carry()
        batch = make_batch(device="cuda")
        _, outputs = model(carry, batch)
        assert outputs["q_halt_logits"].shape == (BATCH,)
        assert outputs["q_continue_logits"].shape == (BATCH,)

    def test_target_q_present_in_training(self):
        """target_q_continue should appear in outputs when training."""
        model, carry = self._model_and_carry(training=True)
        batch = make_batch(device="cuda")
        _, outputs = model(carry, batch)
        assert "target_q_continue" in outputs

    def test_target_q_absent_in_eval(self):
        """target_q_continue should NOT appear in outputs during eval."""
        model, carry = self._model_and_carry(training=False)
        batch = make_batch(device="cuda")
        _, outputs = model(carry, batch)
        assert "target_q_continue" not in outputs


# ---------------------------------------------------------------------------
# ACTLossHead (CUDA required for forward)
# ---------------------------------------------------------------------------


@CUDA_ONLY
class TestACTLossHead:
    def _make_model(self):
        cfg = SMALL_CFG.model_copy(update={"forward_dtype": "bfloat16"})
        act = CoralACT(cfg).cuda()
        head = ACTLossHead(act, loss_type="softmax_cross_entropy")
        return head

    def test_loss_is_scalar(self):
        head = self._make_model()
        head.train()
        batch = make_batch(device="cuda")
        with torch.device("cuda"):
            carry = head.initial_carry(batch)
        _, loss, _, _, _ = head(carry=carry, batch=batch, return_keys=[])
        assert loss.shape == ()

    def test_metrics_keys(self):
        head = self._make_model()
        head.train()
        batch = make_batch(device="cuda")
        with torch.device("cuda"):
            carry = head.initial_carry(batch)
        _, _, metrics, _, _ = head(carry=carry, batch=batch, return_keys=[])
        for key in ("count", "accuracy", "exact_accuracy", "q_halt_accuracy", "steps"):
            assert key in metrics, f"Missing metric: {key}"

    def test_all_halted_flag_type(self):
        """all_halted should be a 0-dim bool tensor."""
        head = self._make_model()
        head.eval()
        batch = make_batch(device="cuda")
        with torch.device("cuda"):
            carry = head.initial_carry(batch)
        _, _, _, _, all_halted = head(carry=carry, batch=batch, return_keys=[])
        assert all_halted.dtype == torch.bool

    def test_new_carry_steps_shape(self):
        head = self._make_model()
        head.train()
        batch = make_batch(device="cuda")
        with torch.device("cuda"):
            carry = head.initial_carry(batch)
        new_carry, _, _, _, _ = head(carry=carry, batch=batch, return_keys=[])
        assert new_carry.steps.shape == (BATCH,)

    def test_detached_outputs_returned(self):
        """return_keys should select output tensors, detached."""
        head = self._make_model()
        head.train()
        batch = make_batch(device="cuda")
        with torch.device("cuda"):
            carry = head.initial_carry(batch)
        _, _, _, detached, _ = head(carry=carry, batch=batch, return_keys=["logits"])
        assert "logits" in detached
        assert detached["logits"].grad_fn is None
