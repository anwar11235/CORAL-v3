"""Tests for warm-start checkpoint key remapping under sub-module torch.compile.

Three prefix patterns exercised:
  1. Top-level compiled checkpoint (_orig_mod. at root) → sub-module compiled model
  2. Uncompiled checkpoint (no prefix) → sub-module compiled model
  3. Sub-module compiled checkpoint (_orig_mod. at boundary) → same model (idempotent)

Plus two CPU integration tests against the full load_warmstart_checkpoint path.
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train import (  # noqa: E402
    TrainState,
    _remap_checkpoint_keys_for_submodule_compile,
    load_warmstart_checkpoint,
)


# ---------------------------------------------------------------------------
# Minimal model that mirrors CORAL's LossHead → ACT.inner → [H_level, L_level]
# attribute structure, with non-compiled sibling to verify unchanged keys.
# ---------------------------------------------------------------------------

class _Leaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))


class _Mid(nn.Module):
    def __init__(self):
        super().__init__()
        self.H_level = _Leaf()
        self.L_level = _Leaf()
        self.other = nn.Linear(4, 4, bias=False)


class _Top(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Mid()


def _subcompiled_top() -> _Top:
    top = _Top()
    top.model.H_level = torch.compile(top.model.H_level, backend="eager")
    top.model.L_level = torch.compile(top.model.L_level, backend="eager")
    return top


def _make_state(model: nn.Module) -> TrainState:
    return TrainState(
        model=model,
        optimizers=[],
        optimizer_lrs=[],
        carry=None,
        step=0,
        total_steps=0,
    )


# ---------------------------------------------------------------------------
# Unit tests: _remap_checkpoint_keys_for_submodule_compile
# ---------------------------------------------------------------------------

class TestRemapKeys:
    def test_case1_top_level_compiled_checkpoint(self):
        """After stripping root _orig_mod., insert prefix at H_level / L_level boundary."""
        model = _subcompiled_top()
        # Keys as they appear AFTER stripping the root _orig_mod. prefix
        # (i.e., what a phase1 top-level-compiled checkpoint looks like post-strip)
        ckpt = {
            "model.H_level.weight": torch.zeros(4, 4),
            "model.L_level.weight": torch.zeros(4, 4),
            "model.other.weight": torch.zeros(4, 4),
        }
        remapped, affected = _remap_checkpoint_keys_for_submodule_compile(ckpt, model)

        assert "model.H_level._orig_mod.weight" in remapped
        assert "model.L_level._orig_mod.weight" in remapped
        assert "model.other.weight" in remapped          # non-compiled sibling unchanged
        assert "model.H_level.weight" not in remapped    # old key gone
        assert "model.L_level.weight" not in remapped
        assert len(affected) == 2

    def test_case2_uncompiled_checkpoint(self):
        """Checkpoint has no compile prefix at all; sub-module prefix is still inserted."""
        model = _subcompiled_top()
        ckpt = {
            "model.H_level.weight": torch.zeros(4, 4),
            "model.L_level.weight": torch.zeros(4, 4),
            "model.other.weight": torch.zeros(4, 4),
        }
        remapped, affected = _remap_checkpoint_keys_for_submodule_compile(ckpt, model)

        assert "model.H_level._orig_mod.weight" in remapped
        assert "model.L_level._orig_mod.weight" in remapped
        assert "model.other.weight" in remapped
        assert len(affected) == 2

    def test_case3_submodule_compiled_checkpoint_idempotent(self):
        """Checkpoint already has _orig_mod. at sub-module boundary — no change."""
        model = _subcompiled_top()
        ckpt = {
            "model.H_level._orig_mod.weight": torch.zeros(4, 4),
            "model.L_level._orig_mod.weight": torch.zeros(4, 4),
            "model.other.weight": torch.zeros(4, 4),
        }
        remapped, affected = _remap_checkpoint_keys_for_submodule_compile(ckpt, model)

        assert remapped == ckpt, "Idempotency violated: keys changed when already correct"
        assert affected == [], f"Expected no affected prefixes, got {affected}"

    def test_no_compiled_submodules_passthrough(self):
        """Uncompiled model: dict is returned unchanged."""
        model = _Top()  # no torch.compile
        ckpt = {
            "model.H_level.weight": torch.zeros(4, 4),
            "model.other.weight": torch.zeros(4, 4),
        }
        remapped, affected = _remap_checkpoint_keys_for_submodule_compile(ckpt, model)

        assert remapped == ckpt
        assert affected == []

    def test_non_compiled_keys_never_modified(self):
        """Keys that don't belong to compiled sub-modules are always left alone."""
        model = _subcompiled_top()
        ckpt = {
            "model.H_level.weight": torch.zeros(4, 4),
            "model.other.weight": torch.zeros(4, 4),
            "some.random.key": torch.zeros(1),
        }
        remapped, _ = _remap_checkpoint_keys_for_submodule_compile(ckpt, model)

        assert "model.other.weight" in remapped
        assert "some.random.key" in remapped


# ---------------------------------------------------------------------------
# Integration test 1: sub-module compiled → sub-module compiled (same structure)
# ---------------------------------------------------------------------------

def test_integration_subcompiled_roundtrip():
    """Save from sub-module compiled model; reload into a fresh sub-module compiled model.

    Verifies that backbone weights land correctly and no unexpected keys remain
    for the H_level / L_level parameters after the fix.
    """
    source = _subcompiled_top()
    with torch.no_grad():
        source.model.H_level._orig_mod.weight.fill_(1.0)
        source.model.L_level._orig_mod.weight.fill_(2.0)
        source.model.other.weight.fill_(3.0)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmppath = f.name
    try:
        torch.save(source.state_dict(), tmppath)

        target = _subcompiled_top()
        state = _make_state(target)
        load_warmstart_checkpoint(state, tmppath, rank=1)  # rank=1 suppresses prints

        assert target.model.H_level._orig_mod.weight.mean().item() == pytest.approx(1.0, abs=1e-5)
        assert target.model.L_level._orig_mod.weight.mean().item() == pytest.approx(2.0, abs=1e-5)
        assert target.model.other.weight.mean().item() == pytest.approx(3.0, abs=1e-5)
    finally:
        os.unlink(tmppath)


# ---------------------------------------------------------------------------
# Integration test 2: Phase 1 checkpoint format → sub-module compiled model
# ---------------------------------------------------------------------------

def test_integration_phase1_format_to_subcompiled():
    """Simulate loading a Phase 1 top-level-compiled checkpoint into a sub-module compiled model.

    Constructs a checkpoint whose keys match the format produced by
    torch.compile(whole_model) (root _orig_mod. prefix, no sub-module prefix),
    then verifies load_warmstart_checkpoint maps them onto the sub-module compiled
    model without any unexpected keys for backbone params.
    """
    # Build reference weights in an uncompiled model
    reference = _Top()
    with torch.no_grad():
        reference.model.H_level.weight.fill_(7.0)
        reference.model.L_level.weight.fill_(8.0)
        reference.model.other.weight.fill_(9.0)

    # Construct Phase 1-format checkpoint: prefix every key with "_orig_mod."
    # (top-level compile wraps the root module, so all keys gain this prefix)
    phase1_ckpt = {
        "_orig_mod." + k: v.clone()
        for k, v in reference.state_dict().items()
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmppath = f.name
    try:
        torch.save(phase1_ckpt, tmppath)

        target = _subcompiled_top()
        state = _make_state(target)
        load_warmstart_checkpoint(state, tmppath, rank=1)

        assert target.model.H_level._orig_mod.weight.mean().item() == pytest.approx(7.0, abs=1e-5)
        assert target.model.L_level._orig_mod.weight.mean().item() == pytest.approx(8.0, abs=1e-5)
        assert target.model.other.weight.mean().item() == pytest.approx(9.0, abs=1e-5)
    finally:
        os.unlink(tmppath)
