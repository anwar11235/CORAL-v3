"""CORAL v3 training entry point.

Usage (single GPU):
    python scripts/train.py data_path=data/sudoku-extreme-1k-aug-1000 lr=7e-5

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=8 scripts/train.py data_path=... lr=1e-4

All hyperparameters are defined in configs/base.yaml and can be overridden
as Hydra command-line overrides (key=value).
"""

import math
import os
import shutil
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import hydra
import pydantic
import torch
import torch.distributed as dist

# Enable TF32 tensor cores for fp32 matmuls (residual fp32 ops benefit on A100; ~1-2% speedup).
torch.set_float32_matmul_precision("high")

# Suppress recompile-limit warnings from variable sub-batch sizes in columnar routing
# torch 2.6 uses cache_size_limit; torch 2.7+ renamed it to recompile_limit.
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 64      # torch 2.7+
else:
    torch._dynamo.config.cache_size_limit = 64     # torch 2.6 (legacy name)
import tqdm
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

try:
    from adam_atan2_pytorch import AdamAtan2 as AdamATan2
    FUSED_ADAM_ATAN2 = True
except ImportError:
    from coral.training.adam_atan2 import AdamATan2
    FUSED_ADAM_ATAN2 = False

from coral.data.puzzle_dataset import PuzzleDatasetMetadata, create_dataloader
from coral.models.columnar import ColumnarTransformerBlock
from coral.models.coral_base import CoralConfig
from coral.models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from coral.training.act import CoralACT, CoralV3ACT
from coral.training.losses import ACTLossHead, CoralV3LossHead
from coral.training.scheduler import cosine_schedule_with_warmup_lr_lambda


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TrainConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    # Data
    data_path: str
    global_batch_size: int = 384
    epochs: int = 20000
    eval_interval: Optional[int] = 2000

    # Model
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0
    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 4
    L_layers: int = 4
    puzzle_emb_ndim: int = 512
    pos_encodings: str = "rope"
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    forward_dtype: str = "bfloat16"

    # Phase 1: predictive coding
    use_predictive_coding: bool = False
    lambda_pred: float = 0.1
    lambda_pi: float = 0.01

    # Phase 2: sparse columnar routing
    use_columnar_routing: bool = False
    num_columns: int = 8
    active_columns: int = 2
    lambda_balance: float = 0.1
    column_warmup_steps: int = 10000  # steps to anneal from start_k → active_columns; 0 = skip
    column_warmup_start_k: int = 8    # k at step 0 (defaults to S = num_columns)

    # Phase 3b: Soft MoE Crystallization
    use_crystallization: bool = False
    codebook_size: int = 256
    crystal_proj_dim: int = 128
    crystal_buffer_capacity: int = 10000
    crystal_consolidation_interval: int = 5000  # steps between spatial k-means runs; 0 = never
    crystal_bootstrap_steps: int = 5000    # steps before first k-means consolidation
    moe_num_modes: int = 32               # K_modes — spatial codebook experts
    lambda_moe_recon: float = 0.1         # weight for unweighted reconstruction loss
    lambda_moe_balance: float = 0.01      # weight for codebook load-balancing KL loss

    # Warm-start
    resume_from_checkpoint: Optional[str] = None  # path to a .pt state_dict to warm-start from

    # Loss
    loss_type: str = "stablemax_cross_entropy"

    # Optimizer
    lr: float = 7e-5
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 1000
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # Puzzle embedding optimizer
    puzzle_emb_lr: float = 1e-3
    puzzle_emb_weight_decay: float = 0.1

    # Run management
    seed: int = 0
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    eval_save_outputs: List[str] = []


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int


# ---------------------------------------------------------------------------
# Model and optimizer construction
# ---------------------------------------------------------------------------


def build_model(config: TrainConfig, metadata: PuzzleDatasetMetadata, world_size: int) -> nn.Module:
    coral_cfg = CoralConfig(
        batch_size=config.global_batch_size // world_size,
        seq_len=metadata.seq_len,
        vocab_size=metadata.vocab_size,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        puzzle_emb_ndim=config.puzzle_emb_ndim,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        H_layers=config.H_layers,
        L_layers=config.L_layers,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        expansion=config.expansion,
        pos_encodings=config.pos_encodings,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        halt_max_steps=config.halt_max_steps,
        halt_exploration_prob=config.halt_exploration_prob,
        forward_dtype=config.forward_dtype,
        # Phase 1: predictive coding
        use_predictive_coding=config.use_predictive_coding,
        lambda_pred=config.lambda_pred,
        lambda_pi=config.lambda_pi,
        # Phase 2: sparse columnar routing
        use_columnar_routing=config.use_columnar_routing,
        num_columns=config.num_columns,
        active_columns=config.active_columns,
        lambda_balance=config.lambda_balance,
        # Phase 3b: Soft MoE crystallization
        use_crystallization=config.use_crystallization,
        codebook_size=config.codebook_size,
        crystal_proj_dim=config.crystal_proj_dim,
        crystal_buffer_capacity=config.crystal_buffer_capacity,
        crystal_consolidation_interval=config.crystal_consolidation_interval,
        crystal_bootstrap_steps=config.crystal_bootstrap_steps,
        moe_num_modes=config.moe_num_modes,
        lambda_moe_recon=config.lambda_moe_recon,
        lambda_moe_balance=config.lambda_moe_balance,
    )

    _any_v3 = config.use_predictive_coding or config.use_columnar_routing or config.use_crystallization

    with torch.device("cuda"):
        if _any_v3:
            inner_model = CoralV3ACT(coral_cfg)
            model: nn.Module = CoralV3LossHead(inner_model, loss_type=config.loss_type)
        else:
            inner_model = CoralACT(coral_cfg)
            model = ACTLossHead(inner_model, loss_type=config.loss_type)

        if "DISABLE_COMPILE" not in os.environ:
            if _any_v3:
                # Compile only the hot transformer kernels (H_level, L_level) to avoid the
                # graph-break recompile storm:  the disabled moe_losses() returns new scalar
                # tensors each call; when the entire model is torch.compiled, dynamo resumes
                # at act.py:277 and guards on PredMetrics.moe_lb_loss object identity —
                # failing every call and hitting cache_size_limit=64.
                # Compiling H_level / L_level as standalone sub-modules gives ~same speedup
                # (they contain all the attention + FFN compute) while the outer ACT / loss
                # code runs eagerly with no graph breaks.
                _inner = inner_model.inner  # CoralV3Inner (inherits H_level, L_level from CoralInner)
                _inner.H_level = torch.compile(_inner.H_level)  # type: ignore[assignment]
                _inner.L_level = torch.compile(_inner.L_level)  # type: ignore[assignment]
            else:
                model = torch.compile(model, dynamic=config.use_columnar_routing)  # type: ignore[assignment]

        if world_size > 1:
            with torch.no_grad():
                for p in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(p, src=0)

    return model


def build_optimizers(model: nn.Module, config: TrainConfig, world_size: int):
    optimizers = []
    optimizer_lrs = []

    # Sparse embedding optimizer (skip if no puzzle embedding)
    if config.puzzle_emb_ndim > 0:
        optimizers.append(
            CastedSparseEmbeddingSignSGD_Distributed(
                model.puzzle_emb.buffers(),  # type: ignore[operator]
                lr=1e-30,  # Near-zero init; actual LR set by scheduler.
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            )
        )
        optimizer_lrs.append(config.puzzle_emb_lr)

    # Main optimizer for all trainable parameters
    optimizers.append(
        AdamATan2(
            model.parameters(),
            lr=1e-30,  # Near-zero init; actual LR set by scheduler.
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
    )
    optimizer_lrs.append(config.lr)

    return optimizers, optimizer_lrs


def compute_active_columns(config: TrainConfig, step: int) -> int:
    """Linearly anneal active columns from column_warmup_start_k to active_columns."""
    if not config.use_columnar_routing or config.column_warmup_steps == 0:
        return config.active_columns
    if step >= config.column_warmup_steps:
        return config.active_columns
    frac = step / config.column_warmup_steps
    k = config.column_warmup_start_k + frac * (config.active_columns - config.column_warmup_start_k)
    return max(config.active_columns, round(k))


def set_active_columns(model: nn.Module, k: int) -> None:
    """Update k on every ColumnarTransformerBlock in the model."""
    for module in model.modules():
        if isinstance(module, ColumnarTransformerBlock):
            module.k = k


def init_train_state(
    config: TrainConfig,
    metadata: PuzzleDatasetMetadata,
    world_size: int,
) -> TrainState:
    total_steps = int(
        config.epochs * metadata.total_groups * metadata.mean_puzzle_examples
        / config.global_batch_size
    )
    model = build_model(config, metadata, world_size)
    optimizers, optimizer_lrs = build_optimizers(model, config, world_size)
    return TrainState(
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        step=0,
        total_steps=total_steps,
    )


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def compute_lr(base_lr: float, config: TrainConfig, state: TrainState) -> float:
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=state.total_steps,
        min_ratio=config.lr_min_ratio,
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_batch(
    config: TrainConfig,
    state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
) -> Optional[dict]:
    state.step += 1
    if state.step > state.total_steps:
        return None

    # Column warm-up: anneal active k from start_k → target k over warmup_steps
    current_k = config.active_columns
    if config.use_columnar_routing:
        current_k = compute_active_columns(config, state.step)
        set_active_columns(state.model, current_k)

    batch = {k: v.cuda() for k, v in batch.items()}

    if state.carry is None:
        with torch.device("cuda"):
            state.carry = state.model.initial_carry(batch)  # type: ignore[operator]

    state.carry, loss, metrics, _, _ = state.model(  # type: ignore[operator]
        carry=state.carry, batch=batch, return_keys=[]
    )

    ((1.0 / global_batch_size) * loss).backward()

    if world_size > 1:
        for p in state.model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)

    lr_this_step = None
    for optim, base_lr in zip(state.optimizers, state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, state)
        for pg in optim.param_groups:
            pg["lr"] = lr_this_step
        optim.step()
        optim.zero_grad()

    if rank == 0 and len(metrics):
        metric_keys = sorted(metrics.keys())
        metric_values = torch.stack([metrics[k] for k in metric_keys])

        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        vals = metric_values.cpu().numpy()
        reduced = {k: vals[i] for i, k in enumerate(metric_keys)}
        count = max(reduced["count"], 1)
        reduced = {
            f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
            for k, v in reduced.items()
        }
        reduced["train/lr"] = lr_this_step
        if config.use_columnar_routing:
            reduced["train/active_columns"] = current_k
        return reduced

    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    config: TrainConfig,
    state: TrainState,
    eval_loader: DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
) -> Optional[dict]:
    with torch.inference_mode():
        set_ids = {k: i for i, k in enumerate(eval_metadata.sets)}
        metric_keys: List[str] = []
        metric_values = None
        metric_gbs = [0] * len(set_ids)

        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = state.model.initial_carry(batch)  # type: ignore[operator]

            # Run all halt_max_steps segments
            while True:
                carry, _, metrics, _, all_done = state.model(  # type: ignore[operator]
                    carry=carry, batch=batch, return_keys=[]
                )
                if all_done:
                    break

            sid = set_ids[set_name]
            if metric_values is None:
                metric_keys = sorted(metrics.keys())
                metric_values = torch.zeros(
                    (len(set_ids), len(metric_keys)), dtype=torch.float32, device="cuda"
                )
            metric_values[sid] += torch.stack([metrics[k] for k in metric_keys])
            metric_gbs[sid] += global_batch_size

        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                mv = metric_values.cpu().numpy()
                result = {}
                for sid, sname in enumerate(eval_metadata.sets):
                    m = {metric_keys[i]: mv[sid, i] for i in range(len(metric_keys))}
                    count = max(m.pop("count"), 1)
                    m = {k: v / count for k, v in m.items()}
                    # Convert moe_passthrough_weight to codebook_weight for logging clarity
                    if "crystal/mean_passthrough_weight" in m:
                        m["crystal/mean_codebook_weight"] = 1.0 - m["crystal/mean_passthrough_weight"]
                    result[sname] = m
                return result

    return None


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    config: TrainConfig,
    state: TrainState,
    run_name: str,
    eval_exact_acc: float,
    best_path: Optional[str],
    latest_path: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Save a step-stamped checkpoint; keep only best and latest.

    Returns:
        (new_best_path, new_latest_path) — callers should persist these.
    """
    if config.checkpoint_path is None:
        return best_path, latest_path

    os.makedirs(config.checkpoint_path, exist_ok=True)
    new_path = os.path.join(
        config.checkpoint_path, f"{run_name}_step{state.step}.pt"
    )
    torch.save(state.model.state_dict(), new_path)
    print(
        f"[CORAL-v3] Checkpoint saved: {new_path}"
        f" (eval_exact_accuracy={eval_exact_acc:.4f})"
    )

    # Determine whether this is the new best
    best_acc = getattr(save_checkpoint, "_best_acc", -1.0)
    is_new_best = eval_exact_acc > best_acc
    if is_new_best:
        save_checkpoint._best_acc = eval_exact_acc  # type: ignore[attr-defined]
        print(f"[CORAL-v3] New best checkpoint: {new_path}")

    new_best_path = new_path if is_new_best else best_path
    new_latest_path = new_path

    # Delete stale checkpoints (anything that is neither best nor latest)
    paths_to_keep = {p for p in (new_best_path, new_latest_path) if p is not None}
    for old_path in (best_path, latest_path):
        if old_path is not None and old_path not in paths_to_keep:
            try:
                os.remove(old_path)
            except FileNotFoundError:
                pass

    return new_best_path, new_latest_path


# ---------------------------------------------------------------------------
# Warm-start checkpoint loading
# ---------------------------------------------------------------------------


def load_warmstart_checkpoint(
    state: TrainState,
    checkpoint_path: str,
    rank: int,
) -> TrainState:
    """Warm-start model weights from a checkpoint, ignoring missing keys.

    Designed for cross-phase warm-starts (e.g., Phase 1 → Phase 3) where the
    source checkpoint does not contain crystallization module weights.  Uses
    strict=False so only the keys present in the checkpoint are loaded; new
    modules (SpatialMoECodebook, CrystallizationBuffer parameters) keep their
    random initialisation.

    Handles torch.compile prefix mismatches on both sides: if either the
    checkpoint or the current model was wrapped by torch.compile, keys are
    normalised before load_state_dict so backbone weights match regardless
    of compile state at save/load time.

    Optimizer state is NOT restored — optimizers always start fresh.  This is
    correct for cross-phase warm-starts where the parameter set has changed.

    Args:
        state:           Current TrainState (model already built).
        checkpoint_path: Path to a .pt file saved via torch.save(model.state_dict()).
        rank:            Current process rank (logging only on rank 0).

    Returns:
        The same TrainState with model weights partially loaded.
    """
    if rank == 0:
        print(f"[CORAL-v3] Warm-starting from checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Handle the case where the checkpoint is wrapped in a dict (e.g., from
    # a trainer that saved {"model_state_dict": ..., "optimizer_state_dict": ...}).
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    # Strip torch.compile "_orig_mod." prefix from checkpoint keys if present.
    # Checkpoints saved from compiled models have this prefix; uncompiled
    # models expect keys without it.
    if any(k.startswith("_orig_mod.") for k in ckpt.keys()):
        if rank == 0:
            print("[CORAL-v3] Stripping '_orig_mod.' prefix from checkpoint keys (source was compiled)")
        ckpt = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
                for k, v in ckpt.items()}

    # Unwrap torch.compile on the target model side as well (defensive:
    # works whether current model is compiled or not).
    target_model = state.model
    if hasattr(target_model, "_orig_mod"):
        target_model = target_model._orig_mod  # type: ignore[attr-defined]

    result = target_model.load_state_dict(ckpt, strict=False)

    if rank == 0:
        if result.missing_keys:
            print(
                f"[CORAL-v3] Warm-start: {len(result.missing_keys)} keys not in checkpoint "
                f"(expected for new modules):"
            )
            for k in sorted(result.missing_keys):
                print(f"  MISSING  {k}")
        if result.unexpected_keys:
            print(
                f"[CORAL-v3] Warm-start: {len(result.unexpected_keys)} unexpected keys "
                f"(checkpoint has keys not in current model — verify this is intentional):"
            )
            for k in sorted(result.unexpected_keys):
                print(f"  UNEXPECTED  {k}")
        print("[CORAL-v3] Warm-start complete.")

    return state


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(hydra_config: DictConfig) -> None:
    RANK = 0
    WORLD_SIZE = 1

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Parse config
    config_dict = dict(hydra_config)
    config = TrainConfig(**config_dict)

    # Auto-generate names
    if config.project_name is None:
        config.project_name = f"{os.path.basename(config.data_path).capitalize()} CORAL-v3"
    if config.run_name is None:
        try:
            import coolname
            config.run_name = coolname.generate_slug(2)
        except ImportError:
            import uuid
            config.run_name = str(uuid.uuid4())[:8]
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join(
            "checkpoints", config.project_name, config.run_name
        )

    if RANK == 0:
        backend = "fused CUDA" if FUSED_ADAM_ATAN2 else "pure PyTorch"
        print(f"[CORAL-v3] AdamATan2 backend: {backend}")

    torch.manual_seed(config.seed + RANK)

    # Data
    eval_interval = config.eval_interval or config.epochs
    total_iters = config.epochs // eval_interval
    assert config.epochs % eval_interval == 0, "eval_interval must divide epochs evenly."

    train_loader, train_meta = create_dataloader(
        dataset_path=config.data_path,
        split="train",
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
        seed=config.seed,
        test_set_mode=False,
        epochs_per_iter=eval_interval,
    )
    eval_loader, eval_meta = create_dataloader(
        dataset_path=config.data_path,
        split="test",
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
        seed=config.seed,
        test_set_mode=True,
        epochs_per_iter=1,
    )

    state = init_train_state(config, train_meta, world_size=WORLD_SIZE)

    # Warm-start: load backbone weights from a prior-phase checkpoint.
    # Missing keys (new modules like SpatialMoECodebook) are expected and logged.
    if config.resume_from_checkpoint:
        state = load_warmstart_checkpoint(state, config.resume_from_checkpoint, RANK)

    if RANK == 0:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True),
        )
        wandb.log(
            {"num_params": sum(p.numel() for p in state.model.parameters())}, step=0
        )
        pbar = tqdm.tqdm(total=state.total_steps)

    best_checkpoint_path: Optional[str] = None
    latest_checkpoint_path: Optional[str] = None

    # Bootstrap-phase state — persists across eval checkpoints.
    # first_consolidation_done tracks whether the first spatial k-means consolidation
    # has succeeded; until then the codebook mask is active (w_pt forced to 1.0).
    first_consolidation_done: bool = False
    # When crystal_bootstrap_steps == 0 consolidation is skipped entirely (immediate live).
    if config.use_crystallization and config.crystal_bootstrap_steps == 0:
        first_consolidation_done = True

    for _iter in range(total_iters):
        if RANK == 0:
            print(f"[CORAL-v3] Epoch {_iter * eval_interval}")

        # ---- Train ----
        state.model.train()
        for set_name, batch, gbs in train_loader:
            metrics = train_batch(config, state, batch, gbs, rank=RANK, world_size=WORLD_SIZE)
            if RANK == 0 and metrics:
                wandb.log(metrics, step=state.step)
                pbar.update(state.step - pbar.n)  # type: ignore[operator]

            if config.use_crystallization:
                # --- Crystal bootstrap-phase monitoring ---
                inner = state.model.model.inner  # type: ignore[attr-defined]

                # Log buffer fill and phase indicator every step (cheap scalars)
                if RANK == 0:
                    buf_fill = len(inner.crystal_buffer) / max(inner.crystal_buffer.capacity, 1)
                    wandb.log(
                        {
                            "train/crystal/buffer_fill": buf_fill,
                            "train/crystal/first_consolidation_done": float(first_consolidation_done),
                        },
                        step=state.step,
                    )

                # --- Consolidation trigger ---
                # First consolidation fires at crystal_bootstrap_steps (with full replace).
                # Subsequent consolidations fire every crystal_consolidation_interval steps.
                should_consolidate = (
                    config.crystal_consolidation_interval > 0
                    and state.step >= config.crystal_bootstrap_steps
                    and (state.step - config.crystal_bootstrap_steps)
                        % config.crystal_consolidation_interval == 0
                )
                if should_consolidate:
                    is_first = not first_consolidation_done
                    usage = inner.consolidate_codebook(is_first_consolidation=is_first)
                    if RANK == 0:
                        if usage is None:
                            print(
                                f"[CORAL-v3] Consolidation skipped at step {state.step}: "
                                f"buffer disabled post-bootstrap (backprop-only update path active)"
                            )
                        else:
                            print(
                                f"[CORAL-v3] Codebook consolidation at step {state.step}"
                                f" (first={is_first}, usage={usage})"
                            )
                    if usage is not None and not first_consolidation_done:
                        first_consolidation_done = True
                        # consolidate_codebook() already deactivates bootstrap mask internally
                        if RANK == 0:
                            print("[CORAL-v3] Spatial k-means consolidation done — MoE codebook live.")
                    if usage is not None and RANK == 0:
                        wandb.log(
                            {"train/crystal/codebook_utilisation_frac": usage},
                            step=state.step,
                        )

        # ---- Eval ----
        state.model.eval()
        eval_metrics = evaluate(config, state, eval_loader, eval_meta, rank=RANK, world_size=WORLD_SIZE)
        if RANK == 0 and eval_metrics:
            flat = {f"eval/{sname}/{k}": v for sname, m in eval_metrics.items() for k, v in m.items()}
            # Also log flat aliases (eval/<metric>) for the primary eval set ("test").
            # This matches the naming used in Phase 1 runs (poetic-giraffe) so W&B
            # comparison charts can overlay Phase 1 and Phase 3a'+ runs on the same axis.
            primary = eval_metrics.get("test") or (next(iter(eval_metrics.values())) if eval_metrics else None)
            if primary:
                for k, v in primary.items():
                    flat[f"eval/{k}"] = v
            wandb.log(flat, step=state.step)

        # ---- Checkpoint ----
        if RANK == 0 and eval_metrics:
            # Extract best exact_accuracy across all eval sets
            eval_exact_acc = max(
                m.get("exact_accuracy", 0.0)
                for m in eval_metrics.values()
            )
            best_checkpoint_path, latest_checkpoint_path = save_checkpoint(
                config,
                state,
                run_name=config.run_name,
                eval_exact_acc=eval_exact_acc,
                best_path=best_checkpoint_path,
                latest_path=latest_checkpoint_path,
            )

    if dist.is_initialized():
        dist.destroy_process_group()
    if RANK == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
