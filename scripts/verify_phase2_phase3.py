"""Dry-run verification for Phase 2, Phase 3, and combined configs.

CPU only — no GPU, no subprocess, no actual training.
Imports components directly, runs one forward pass per config,
and confirms the expected metric keys appear in the output dict.

Also does a Hydra config dry-parse to confirm the +flag=value
override syntax is accepted without errors.

Usage:
    python scripts/verify_phase2_phase3.py
"""

import os
import sys
import traceback

import torch

# Ensure the project root is importable regardless of cwd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PASS_COUNT = 0
FAIL_COUNT = 0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_coral_cfg(**phase_flags):
    """Small CPU-compatible CoralConfig for one-forward smoke tests."""
    from coral.models.coral_base import CoralConfig
    return CoralConfig(
        batch_size=2,
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
        halt_exploration_prob=0.0,
        puzzle_emb_ndim=0,
        forward_dtype="float32",   # bfloat16 attention can fail on CPU
        # Phase 2: keep S small so the test stays fast
        num_columns=4,
        active_columns=2,
        # Phase 3: tiny codebook / projection
        codebook_size=16,
        crystal_proj_dim=32,
        crystal_buffer_capacity=200,
        **phase_flags,
    )


def _make_batch(cfg):
    B, SEQ = cfg.batch_size, cfg.seq_len
    return {
        "inputs": torch.randint(0, cfg.vocab_size, (B, SEQ)),
        "labels": torch.randint(0, cfg.vocab_size, (B, SEQ)),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.int32),
    }


def _forward_metrics(cfg) -> set:
    """Build model, run one training-mode forward pass, return metric key set."""
    from coral.training.act import CoralV3ACT
    from coral.training.losses import CoralV3LossHead

    head = CoralV3LossHead(CoralV3ACT(cfg), loss_type="softmax_cross_entropy")
    head.train()
    batch = _make_batch(cfg)
    carry = head.initial_carry(batch)
    _, _loss, metrics, _, _ = head(carry=carry, batch=batch, return_keys=[])
    return set(metrics.keys())


def _record(ok: bool):
    global PASS_COUNT, FAIL_COUNT
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1


# ---------------------------------------------------------------------------
# Forward-pass tests
# ---------------------------------------------------------------------------

_FORWARD_CASES = [
    (
        "Phase 2 — columnar routing only",
        dict(use_columnar_routing=True),
        ["router_entropy", "load_balance_loss"],
    ),
    (
        "Phase 3 — crystallization only",
        dict(use_crystallization=True),
        ["crystal_confidence_mean"],
    ),
    (
        "Phase 1 + 2 — predictive coding + routing",
        dict(use_predictive_coding=True, use_columnar_routing=True),
        ["prediction_error", "precision_mean", "precision_std",
         "router_entropy", "load_balance_loss"],
    ),
    (
        "Phase 1 + 2 + 3 — all mechanisms",
        dict(use_predictive_coding=True, use_columnar_routing=True,
             use_crystallization=True),
        ["prediction_error", "precision_mean", "precision_std",
         "router_entropy", "load_balance_loss",
         "crystal_confidence_mean"],
    ),
]


def run_forward_tests():
    print("\n[1/2]  Model forward-pass tests  (CPU, float32, train mode)")
    print("=" * 60)
    for label, flags, required in _FORWARD_CASES:
        print(f"\n  Config : {label}")
        print(f"  Flags  : {flags}")
        try:
            cfg = _make_coral_cfg(**flags)
            found = _forward_metrics(cfg)
            missing = [k for k in required if k not in found]
            if missing:
                print(f"  FAIL — missing keys: {missing}")
                print(f"         present: {sorted(found)}")
                _record(False)
            else:
                print(f"  PASS — keys confirmed: {required}")
                _record(True)
        except Exception as exc:
            print(f"  FAIL — exception: {exc}")
            traceback.print_exc()
            _record(False)


# ---------------------------------------------------------------------------
# Hydra CLI override dry-parse
# ---------------------------------------------------------------------------

# Each entry: (description, Hydra-style overrides using + for new keys)
_HYDRA_CASES = [
    (
        "Phase 2 only",
        ["+use_columnar_routing=True"],
        dict(use_columnar_routing=True),
    ),
    (
        "Phase 3 only",
        ["+use_crystallization=True"],
        dict(use_crystallization=True),
    ),
    (
        "Phase 1 + 2",
        ["+use_predictive_coding=True", "+use_columnar_routing=True"],
        dict(use_predictive_coding=True, use_columnar_routing=True),
    ),
    (
        "Phase 1 + 2 + 3",
        ["+use_predictive_coding=True", "+use_columnar_routing=True",
         "+use_crystallization=True"],
        dict(use_predictive_coding=True, use_columnar_routing=True,
             use_crystallization=True),
    ),
]


def run_hydra_dry_parse():
    """Use Hydra compose API to verify that +flag=value overrides are parsed
    without errors, then validate the resulting dict through TrainConfig."""
    print("\n[2/2]  Hydra CLI override dry-parse")
    print("       (simulates: python scripts/train.py data_path=X +use_columnar_routing=True ...)")
    print("=" * 60)

    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
    except ImportError as exc:
        print(f"  SKIP — hydra-core not importable: {exc}")
        return

    try:
        from scripts.train import TrainConfig
    except Exception as exc:
        print(f"  SKIP — could not import TrainConfig: {exc}")
        return

    config_dir = os.path.join(_ROOT, "configs")

    for label, hydra_overrides, expected_flags in _HYDRA_CASES:
        print(f"\n  Config  : {label}")
        print(f"  Overrides: {hydra_overrides}")
        GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                raw = compose(
                    config_name="base",
                    overrides=["data_path=dummy_path"] + hydra_overrides,
                )
            cfg_dict = OmegaConf.to_container(raw, resolve=True)

            # Feed into TrainConfig to confirm pydantic accepts every field
            tc = TrainConfig(**cfg_dict)

            # Verify the phase flags took effect
            for field, val in expected_flags.items():
                actual = getattr(tc, field)
                assert actual == val, (
                    f"field '{field}' expected {val!r}, got {actual!r}"
                )
            print(f"  PASS — TrainConfig accepted all overrides")
            _record(True)
        except Exception as exc:
            print(f"  FAIL — {exc}")
            traceback.print_exc()
            _record(False)
        finally:
            GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CORAL v3 — Phase 2/3 dry-run verification  (CPU only)")
    print("=" * 60)

    run_forward_tests()
    run_hydra_dry_parse()

    total = PASS_COUNT + FAIL_COUNT
    print(f"\n{'=' * 60}")
    print(f"Results: {PASS_COUNT}/{total} passed")
    if FAIL_COUNT:
        print("SOME CHECKS FAILED")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)
