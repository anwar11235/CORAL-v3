"""Full pipeline integration test — CPU only, small dimensions, fast.

Tests the complete forward path:
    CoralInner → CoralACT → ACTLossHead → loss.backward()

Uses float32 and the SDPA fallback so no CUDA or flash_attn is needed.
"""

import pytest
import torch

from coral.models.coral_base import CoralConfig, CoralInner
from coral.training.act import ACTCarry, CoralACT
from coral.training.losses import ACTLossHead, IGNORE_LABEL_ID

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

CFG = CoralConfig(
    batch_size=4,
    seq_len=16,
    vocab_size=12,
    hidden_size=64,
    num_heads=2,
    expansion=4.0,
    H_cycles=2,
    L_cycles=2,
    H_layers=1,
    L_layers=1,
    halt_max_steps=2,
    halt_exploration_prob=0.1,
    puzzle_emb_ndim=0,
    forward_dtype="float32",  # float32 — no bfloat16 issues on CPU
)

B = CFG.batch_size
SEQ = CFG.seq_len
VOCAB = CFG.vocab_size


def make_batch():
    return {
        "inputs": torch.randint(0, VOCAB, (B, SEQ)),
        "labels": torch.randint(0, VOCAB, (B, SEQ)),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.int32),
    }


def make_head(loss_type="stablemax_cross_entropy") -> ACTLossHead:
    act = CoralACT(CFG)
    return ACTLossHead(act, loss_type=loss_type)


# ---------------------------------------------------------------------------
# 1. Return-value shapes and types
# ---------------------------------------------------------------------------


class TestReturnValues:
    def test_returns_five_tuple(self):
        head = make_head()
        head.train()
        batch = make_batch()
        carry = head.initial_carry(batch)
        result = head(carry=carry, batch=batch, return_keys=[])
        assert len(result) == 5

    def test_loss_is_scalar(self):
        head = make_head()
        head.train()
        batch = make_batch()
        carry = head.initial_carry(batch)
        _, loss, _, _, _ = head(carry=carry, batch=batch, return_keys=[])
        assert loss.shape == (), f"loss shape should be scalar, got {loss.shape}"

    def test_loss_has_grad(self):
        head = make_head()
        head.train()
        batch = make_batch()
        carry = head.initial_carry(batch)
        _, loss, _, _, _ = head(carry=carry, batch=batch, return_keys=[])
        assert loss.grad_fn is not None, "loss must be part of the computation graph"

    def test_all_halted_is_bool_scalar(self):
        head = make_head()
        head.train()
        batch = make_batch()
        carry = head.initial_carry(batch)
        _, _, _, _, all_halted = head(carry=carry, batch=batch, return_keys=[])
        assert all_halted.dtype == torch.bool
        assert all_halted.shape == ()

    def test_metrics_required_keys(self):
        head = make_head()
        head.train()
        batch = make_batch()
        carry = head.initial_carry(batch)
        _, _, metrics, _, _ = head(carry=carry, batch=batch, return_keys=[])
        for key in ("count", "accuracy", "exact_accuracy", "q_halt_accuracy", "steps", "lm_loss", "q_halt_loss"):
            assert key in metrics, f"Missing metric: {key}"

    def test_new_carry_type(self):
        head = make_head()
        head.train()
        batch = make_batch()
        carry = head.initial_carry(batch)
        new_carry, _, _, _, _ = head(carry=carry, batch=batch, return_keys=[])
        assert isinstance(new_carry, ACTCarry)

    def test_logit_output_via_return_keys(self):
        head = make_head()
        head.train()
        batch = make_batch()
        carry = head.initial_carry(batch)
        _, _, _, detached, _ = head(carry=carry, batch=batch, return_keys=["logits"])
        assert "logits" in detached
        assert detached["logits"].shape == (B, SEQ, VOCAB)
        assert detached["logits"].grad_fn is None  # must be detached


# ---------------------------------------------------------------------------
# 2. Backward pass
# ---------------------------------------------------------------------------


class TestBackward:
    def test_backward_no_error(self):
        head = make_head()
        head.train()
        batch = make_batch()
        carry = head.initial_carry(batch)
        _, loss, _, _, _ = head(carry=carry, batch=batch, return_keys=[])
        loss.backward()  # must not raise

    def test_parameters_have_gradients(self):
        head = make_head()
        head.train()
        batch = make_batch()
        carry = head.initial_carry(batch)
        _, loss, _, _, _ = head(carry=carry, batch=batch, return_keys=[])
        loss.backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert len(grads) > 0, "No parameter received a gradient after backward()"

    def test_carry_detached_after_forward(self):
        """New carry must not be part of the graph (deep supervision)."""
        head = make_head()
        head.train()
        batch = make_batch()
        carry = head.initial_carry(batch)
        new_carry, _, _, _, _ = head(carry=carry, batch=batch, return_keys=[])
        assert new_carry.inner_carry.z_H.grad_fn is None
        assert new_carry.inner_carry.z_L.grad_fn is None


# ---------------------------------------------------------------------------
# 3. Multi-segment carry threading
# ---------------------------------------------------------------------------


class TestMultiSegment:
    def test_second_segment_runs(self):
        """A second call with the carry from the first should not raise."""
        head = make_head()
        head.train()
        batch = make_batch()
        carry = head.initial_carry(batch)
        new_carry, loss1, _, _, _ = head(carry=carry, batch=batch, return_keys=[])
        loss1.backward()

        # Zero grads, then run second segment
        for p in head.parameters():
            if p.grad is not None:
                p.grad.zero_()

        _, loss2, _, _, _ = head(carry=new_carry, batch=batch, return_keys=[])
        loss2.backward()  # must not raise

    def test_steps_increment_across_segments(self):
        """Steps should be 1 after first call, 2 after second (for non-halted)."""
        head = make_head()
        head.train()
        batch = make_batch()
        carry = head.initial_carry(batch)
        new_carry, _, _, _, _ = head(carry=carry, batch=batch, return_keys=[])
        assert (new_carry.steps >= 1).all()

        # Force nothing halted so steps continue incrementing
        new_carry.halted[:] = False
        new_carry2, _, _, _, _ = head(carry=new_carry, batch=batch, return_keys=[])
        assert (new_carry2.steps == 2).all()

    def test_halted_sequences_get_fresh_data(self):
        """On the second call, sequences that halted on step 1 should get the new batch."""
        head = make_head()
        head.train()
        batch = make_batch()
        carry = head.initial_carry(batch)

        # First segment — all start halted, so all get batch data
        new_carry, _, _, _, _ = head(carry=carry, batch=batch, return_keys=[])

        # Manually mark first two as halted, rest not
        new_carry.halted[:] = False
        new_carry.halted[:2] = True

        # Use VOCAB-1 (valid index) as the "old data" sentinel for non-halted sequences
        OLD_VAL = VOCAB - 1  # = 11, valid embedding index
        NEW_VAL = 0          # fresh batch will use 0 — distinguishable from 11
        new_carry.current_data["inputs"][:] = OLD_VAL

        fresh_batch = {
            "inputs": torch.full((B, SEQ), NEW_VAL, dtype=torch.long),
            "labels": torch.randint(0, VOCAB, (B, SEQ)),
            "puzzle_identifiers": torch.zeros(B, dtype=torch.int32),
        }

        new_carry2, _, _, _, _ = head(carry=new_carry, batch=fresh_batch, return_keys=[])

        # Halted sequences (0, 1) should have fresh_batch inputs in their carry data
        assert (new_carry2.current_data["inputs"][:2] == NEW_VAL).all(), \
            "Halted sequences should receive fresh batch data"
        # Non-halted sequences (2, 3) keep their old carry data
        assert (new_carry2.current_data["inputs"][2:] == OLD_VAL).all(), \
            "Non-halted sequences should keep their carry data"

    def test_halt_at_max_steps_eval(self):
        """In eval mode all sequences halt at halt_max_steps."""
        head = make_head()
        head.eval()
        batch = make_batch()

        with torch.no_grad():
            carry = head.initial_carry(batch)
            all_done = False
            steps = 0
            while not all_done:
                carry, _, _, _, all_done = head(carry=carry, batch=batch, return_keys=[])
                steps += 1
                assert steps <= CFG.halt_max_steps + 1, "Did not halt within halt_max_steps"

        assert steps == CFG.halt_max_steps, \
            f"Eval should always run exactly halt_max_steps={CFG.halt_max_steps} segments, ran {steps}"


# ---------------------------------------------------------------------------
# 4. Both loss types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss_type", ["stablemax_cross_entropy", "softmax_cross_entropy"])
def test_loss_types(loss_type):
    head = make_head(loss_type=loss_type)
    head.train()
    batch = make_batch()
    carry = head.initial_carry(batch)
    _, loss, _, _, _ = head(carry=carry, batch=batch, return_keys=[])
    assert loss.item() > 0
    loss.backward()


# ---------------------------------------------------------------------------
# 5. Ignored labels
# ---------------------------------------------------------------------------


def test_ignored_labels_do_not_affect_loss():
    """Setting all labels to IGNORE_LABEL_ID should zero out lm_loss."""
    head = make_head(loss_type="softmax_cross_entropy")
    head.train()
    batch = make_batch()
    batch["labels"][:] = IGNORE_LABEL_ID
    carry = head.initial_carry(batch)
    _, _, metrics, _, _ = head(carry=carry, batch=batch, return_keys=[])
    # lm_loss should be 0 when all labels are ignored
    assert metrics["lm_loss"].item() == 0.0, \
        f"lm_loss should be 0 with all-ignored labels, got {metrics['lm_loss'].item()}"
