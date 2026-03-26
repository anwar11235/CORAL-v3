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
from typing import Any, List, Optional, Sequence

import hydra
import pydantic
import torch
import torch.distributed as dist
import tqdm
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

try:
    from adam_atan2 import AdamATan2
except (ImportError, ModuleNotFoundError):
    from torch.optim import AdamW as AdamATan2
    print("WARNING: adam_atan2 not available, falling back to AdamW")

from coral.data.puzzle_dataset import PuzzleDatasetMetadata, create_dataloader
from coral.models.coral_base import CoralConfig
from coral.models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from coral.training.act import CoralACT
from coral.training.losses import ACTLossHead
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
    checkpoint_every_eval: bool = False
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
    )

    with torch.device("cuda"):
        inner_model = CoralACT(coral_cfg)
        model: nn.Module = ACTLossHead(inner_model, loss_type=config.loss_type)
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore[assignment]

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
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            )
        )
        optimizer_lrs.append(config.puzzle_emb_lr)

    # Main optimizer for all trainable parameters
    optimizers.append(
        AdamATan2(
            model.parameters(),
            lr=0,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
    )
    optimizer_lrs.append(config.lr)

    return optimizers, optimizer_lrs


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
                    result[sname] = {k: v / count for k, v in m.items()}
                return result

    return None


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(config: TrainConfig, state: TrainState) -> None:
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(
        state.model.state_dict(),
        os.path.join(config.checkpoint_path, f"step_{state.step}.pt"),
    )


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

        # ---- Eval ----
        state.model.eval()
        eval_metrics = evaluate(config, state, eval_loader, eval_meta, rank=RANK, world_size=WORLD_SIZE)
        if RANK == 0 and eval_metrics:
            flat = {f"eval/{sname}/{k}": v for sname, m in eval_metrics.items() for k, v in m.items()}
            wandb.log(flat, step=state.step)

        # ---- Checkpoint ----
        if RANK == 0 and (config.checkpoint_every_eval or _iter == total_iters - 1):
            save_checkpoint(config, state)

    if dist.is_initialized():
        dist.destroy_process_group()
    if RANK == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
