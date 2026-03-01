"""
V33 Multi-GPU Training with DDP.

Clean SPLADE-max with ModernBERT (skt/A.X-Encoder-base):
- Pure SPLADE v2 architecture (MLM -> log(1+ReLU) -> max pool)
- FLOPS regularization with quadratic lambda scheduler
- 50K vocab (48.4% Korean), no language filtering needed
- InfoNCE + explicit hard negatives

Usage:
    torchrun --nproc_per_node=8 -m src.train.cli.train_v33_ddp
    torchrun --nproc_per_node=8 -m src.train.cli.train_v33_ddp --config configs/train_v33.yaml
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.model.losses import SPLADELossV33
from src.model.splade_modern import SPLADEModernBERT
from src.train.config.v33 import (
    V33Config,
    V33DataConfig,
    V33LossConfig,
    V33ModelConfig,
    V33TrainingConfig,
)
from src.train.data import load_training_data
from src.train.data.collator import create_tokenizer
from src.train.data.dataloader import TripletCollator
try:
    from src.train.eval import MidTrainingEvaluator
except ImportError:
    MidTrainingEvaluator = None
from src.train.utils import TensorBoardLogger, setup_logging


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="V33 DDP Training: SPLADE-max with ModernBERT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_v33.yaml",
        help="YAML config file path",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override num_epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override per-GPU batch"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override output dir"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to resume"
    )
    parser.add_argument(
        "--lambda-q", type=float, default=None, help="Override lambda_q"
    )
    parser.add_argument(
        "--lambda-d", type=float, default=None, help="Override lambda_d"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (100 steps)"
    )
    return parser.parse_args()


def setup_distributed() -> int:
    """Initialize DDP and return local_rank."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed() -> None:
    """Destroy process group."""
    dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is rank 0."""
    return dist.get_rank() == 0


def load_config(args: argparse.Namespace) -> V33Config:
    """Load config from YAML, apply CLI overrides."""
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        config = V33Config(
            model=V33ModelConfig(**raw.get("model", {})),
            loss=V33LossConfig(**raw.get("loss", {})),
            data=V33DataConfig(**raw.get("data", {})),
            training=V33TrainingConfig(**raw.get("training", {})),
        )
    else:
        config = V33Config()

    # CLI overrides
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    if args.lambda_q is not None:
        config.loss.lambda_q = args.lambda_q
    if args.lambda_d is not None:
        config.loss.lambda_d = args.lambda_d
    if args.grad_accum is not None:
        config.training.gradient_accumulation_steps = args.grad_accum
    if args.seed is not None:
        config.training.seed = args.seed

    return config


def create_dataloader_ddp(
    dataset,
    tokenizer,
    config: V33Config,
    is_train: bool = True,
) -> DataLoader:
    """Create distributed dataloader with asymmetric collator."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=is_train,
    )

    collator = TripletCollator(
        tokenizer=tokenizer,
        max_length=config.data.doc_max_length,
        query_max_length=config.data.query_max_length,
        doc_max_length=config.data.doc_max_length,
        use_in_batch_negatives=True,
    )

    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        sampler=sampler,
        num_workers=config.data.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=is_train,
    )


def save_checkpoint(
    model: DDP,
    optimizer: AdamW,
    scheduler,
    epoch: int,
    global_step: int,
    output_dir: str,
    config: V33Config,
    best_metric: Optional[float] = None,
) -> str:
    """Save checkpoint (rank 0 only)."""
    if not is_main_process():
        return ""

    ckpt_dir = Path(output_dir) / f"checkpoint_epoch{epoch}_step{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights (unwrap DDP)
    model_to_save = model.module
    torch.save(model_to_save.state_dict(), ckpt_dir / "model.pt")

    # Save optimizer and scheduler
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
        },
        ckpt_dir / "training_state.pt",
    )

    # Save config
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(
            {
                "model": config.model.__dict__,
                "loss": config.loss.__dict__,
                "data": config.data.__dict__,
                "training": config.training.__dict__,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved checkpoint: {ckpt_dir}")
    return str(ckpt_dir)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[AdamW],
    scheduler,
    checkpoint_path: str,
) -> Dict:
    """Load checkpoint."""
    ckpt_dir = Path(checkpoint_path)

    # Load model
    state_dict = torch.load(
        ckpt_dir / "model.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    logger.info(f"Loaded model from {ckpt_dir / 'model.pt'}")

    # Load training state
    training_state = torch.load(
        ckpt_dir / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    if optimizer is not None:
        optimizer.load_state_dict(training_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(training_state["scheduler"])

    return training_state


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = sorted(
        output_path.glob("checkpoint_epoch*_step*"),
        key=lambda p: int(p.name.split("_step")[1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def train_epoch(
    model: DDP,
    dataloader: DataLoader,
    loss_fn: SPLADELossV33,
    optimizer: AdamW,
    scheduler,
    config: V33Config,
    epoch: int,
    global_step: int,
    device: torch.device,
    tb_logger: Optional[TensorBoardLogger] = None,
    debug: bool = False,
) -> tuple[float, int]:
    """Train one epoch."""
    model.train()
    dataloader.sampler.set_epoch(epoch)

    total_loss = 0.0
    num_batches = 0

    if is_main_process():
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        progress_bar = dataloader

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        if debug and batch_idx >= 100:
            break

        # Move to device
        q_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        p_ids = batch["positive_input_ids"].to(device)
        p_mask = batch["positive_attention_mask"].to(device)
        n_ids = batch["negative_input_ids"].to(device)
        n_mask = batch["negative_attention_mask"].to(device)
        num_negatives = batch.get("num_negatives", 1)

        # Teacher scores for MarginMSE KD (pre-computed)
        t_pos = batch.get("teacher_pos_scores")
        t_neg = batch.get("teacher_neg_scores")
        if t_pos is not None:
            t_pos = t_pos.to(device)
        if t_neg is not None:
            t_neg = t_neg.to(device)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            # Encode query, positive
            anchor_repr, _ = model(q_ids, q_mask)
            positive_repr, _ = model(p_ids, p_mask)

            # Encode negatives: [batch*k, vocab] for multi-neg
            negative_repr, _ = model(n_ids, n_mask)

            # Reshape to [batch, k, vocab] if multi-neg
            if num_negatives > 1:
                neg_batch = anchor_repr.shape[0]
                negative_repr = negative_repr.view(
                    neg_batch, num_negatives, -1
                )

            # Compute loss
            loss, loss_dict = loss_fn(
                anchor_repr=anchor_repr,
                positive_repr=positive_repr,
                negative_repr=negative_repr,
                global_step=global_step,
                teacher_pos_scores=t_pos,
                teacher_neg_scores=t_neg,
            )

        # Backward
        scaled_loss = loss / config.training.gradient_accumulation_steps
        scaled_loss.backward()

        # Optimizer step
        if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging
            if (
                is_main_process()
                and global_step % config.training.log_every_n_steps == 0
            ):
                lr = scheduler.get_last_lr()[0]
                nz_q, nz_d = loss_fn.get_avg_nonzero()

                kd_str = ""
                if loss_dict.get("kd", 0) > 0:
                    kd_str += f" | kd={loss_dict['kd']:.4f}"
                if loss_dict.get("margin_mse", 0) > 0:
                    kd_str += f" | mmse={loss_dict['margin_mse']:.4f}"

                logger.info(
                    f"Step {global_step} | "
                    f"loss={loss.item():.4f} | "
                    f"infonce={loss_dict['infonce']:.4f} | "
                    f"flops_q={loss_dict['flops_q']:.2f} | "
                    f"flops_d={loss_dict['flops_d']:.2f} | "
                    f"lam_q={loss_dict['lambda_q']:.6f} | "
                    f"lam_d={loss_dict['lambda_d']:.6f} | "
                    f"nz_q={nz_q:.0f} | nz_d={nz_d:.0f} | "
                    f"lr={lr:.2e}{kd_str}"
                )

                if tb_logger:
                    tb_logger.log_scalar("train/loss", loss.item(), global_step)
                    tb_logger.log_scalar("train/lr", lr, global_step)
                    tb_logger.log_scalar(
                        "train/infonce", loss_dict["infonce"], global_step
                    )
                    tb_logger.log_scalar(
                        "train/flops_q", loss_dict["flops_q"], global_step
                    )
                    tb_logger.log_scalar(
                        "train/flops_d", loss_dict["flops_d"], global_step
                    )
                    tb_logger.log_scalar(
                        "train/lambda_q", loss_dict["lambda_q"], global_step
                    )
                    tb_logger.log_scalar(
                        "train/lambda_d", loss_dict["lambda_d"], global_step
                    )
                    tb_logger.log_scalar(
                        "train/nonzero_q", nz_q, global_step
                    )
                    tb_logger.log_scalar(
                        "train/nonzero_d", nz_d, global_step
                    )
                    if loss_dict.get("kd", 0) > 0:
                        tb_logger.log_scalar(
                            "train/kd", loss_dict["kd"], global_step
                        )
                    if loss_dict.get("margin_mse", 0) > 0:
                        tb_logger.log_scalar(
                            "train/margin_mse",
                            loss_dict["margin_mse"],
                            global_step,
                        )

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


def main():
    """V33 training entry point."""
    args = parse_args()

    # DDP setup
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    # Load config
    config = load_config(args)

    # Output dir
    output_dir = Path(config.training.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # Logging
    if is_main_process():
        setup_logging(
            output_dir=str(output_dir),
            log_file="training.log",
        )

    if is_main_process():
        effective_batch = (
            config.data.batch_size
            * config.training.gradient_accumulation_steps
            * world_size
        )
        logger.info("=" * 60)
        logger.info("V33 Training: SPLADE-max with ModernBERT")
        logger.info("=" * 60)
        logger.info(f"Model: {config.model.name}")
        logger.info(f"GPUs: {world_size}")
        logger.info(f"Per-GPU batch: {config.data.batch_size}")
        logger.info(f"Grad accum: {config.training.gradient_accumulation_steps}")
        logger.info(f"Effective batch: {effective_batch}")
        logger.info(f"Learning rate: {config.training.learning_rate}")
        logger.info(f"Epochs: {config.training.num_epochs}")
        logger.info(f"lambda_q: {config.loss.lambda_q}")
        logger.info(f"lambda_d: {config.loss.lambda_d}")
        logger.info(f"FLOPS warmup: {config.loss.flops_warmup_steps} steps")
        if config.loss.lambda_kd > 0:
            logger.info(f"KD (KL): lambda={config.loss.lambda_kd}")
        if config.loss.lambda_margin_mse > 0:
            logger.info(f"KD (MarginMSE): lambda={config.loss.lambda_margin_mse}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 60)

    # Seed
    torch.manual_seed(config.training.seed + dist.get_rank())

    # Tokenizer
    tokenizer = create_tokenizer(config.model.name)

    # Data
    if is_main_process():
        logger.info("Loading training data...")
    train_dataset = load_training_data(config.data.train_files)
    if is_main_process():
        logger.info(f"Train samples: {len(train_dataset):,}")

    val_dataset = None
    if config.data.val_files:
        val_dataset = load_training_data(config.data.val_files)
        if is_main_process():
            logger.info(f"Val samples: {len(val_dataset):,}")

    train_dl = create_dataloader_ddp(
        train_dataset, tokenizer, config, is_train=True
    )

    # Model
    if is_main_process():
        logger.info(f"Loading model: {config.model.name}")
    model = SPLADEModernBERT(
        model_name=config.model.name,
        dropout=config.model.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    if is_main_process():
        logger.info(f"Model parameters: {param_count:,}")
        logger.info(f"Vocab size: {model.vocab_size}")

    # DDP wrap
    model = DDP(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    # Loss
    loss_fn = SPLADELossV33(
        lambda_q=config.loss.lambda_q,
        lambda_d=config.loss.lambda_d,
        temperature=config.loss.temperature,
        flops_warmup_steps=config.loss.flops_warmup_steps,
        lambda_kd=config.loss.lambda_kd,
        kd_temperature=config.loss.kd_temperature,
        lambda_initial_ratio=config.loss.lambda_initial_ratio,
        lambda_margin_mse=config.loss.lambda_margin_mse,
    ).to(device)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.training.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_groups, lr=config.training.learning_rate
    )

    # Scheduler
    steps_per_epoch = len(train_dl) // config.training.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.training.num_epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if is_main_process():
        logger.info(f"Steps/epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

    # Resume
    start_epoch = 1
    global_step = 0
    best_metric = None

    if args.resume or args.checkpoint:
        ckpt_path = args.checkpoint or find_latest_checkpoint(
            config.training.output_dir
        )
        if ckpt_path:
            state = load_checkpoint(
                model.module, optimizer, scheduler, ckpt_path
            )
            start_epoch = state["epoch"] + 1
            global_step = state["global_step"]
            best_metric = state.get("best_metric")
            if is_main_process():
                logger.info(
                    f"Resumed from epoch {state['epoch']}, "
                    f"step {global_step}"
                )

    # TensorBoard
    tb_logger = None
    if is_main_process():
        tb_logger = TensorBoardLogger(
            log_dir=str(output_dir / "tensorboard"),
            experiment_name="v33_modernbert",
        )

    # Mid-training evaluator
    evaluator = None
    if is_main_process() and val_dataset is not None and MidTrainingEvaluator:
        try:
            evaluator = MidTrainingEvaluator(
                tokenizer=tokenizer,
                val_file=config.data.val_files[0],
                max_queries=200,
                max_docs=1000,
                device=str(device),
                query_max_length=config.data.query_max_length,
                doc_max_length=config.data.doc_max_length,
            )
            logger.info("Mid-training evaluator initialized")
        except Exception as e:
            logger.warning(f"Could not init evaluator: {e}")

    # Training loop
    if is_main_process():
        logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(start_epoch, config.training.num_epochs + 1):
        epoch_start = time.time()

        avg_loss, global_step = train_epoch(
            model=model,
            dataloader=train_dl,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            global_step=global_step,
            device=device,
            tb_logger=tb_logger,
            debug=args.debug,
        )

        epoch_time = time.time() - epoch_start

        if is_main_process():
            nz_q, nz_d = loss_fn.get_avg_nonzero()
            logger.info(
                f"Epoch {epoch}/{config.training.num_epochs} | "
                f"avg_loss={avg_loss:.4f} | "
                f"nz_q={nz_q:.0f} | nz_d={nz_d:.0f} | "
                f"time={epoch_time/60:.1f}min"
            )

        # Mid-training eval
        if is_main_process() and evaluator and epoch % 5 == 0:
            try:
                eval_model = model.module
                eval_model.eval()
                metrics = evaluator.evaluate(eval_model)
                logger.info(
                    f"Eval epoch {epoch}: "
                    f"R@1={metrics.get('recall@1', 0):.4f}, "
                    f"R@5={metrics.get('recall@5', 0):.4f}"
                )
                if tb_logger:
                    for k, v in metrics.items():
                        tb_logger.log_scalar(
                            f"eval/{k}", v, global_step
                        )
            except Exception as e:
                logger.warning(f"Eval failed: {e}")

        # Save checkpoint
        if (
            epoch % config.training.save_every_n_epochs == 0
            or epoch == config.training.num_epochs
        ):
            dist.barrier()
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                output_dir=str(output_dir),
                config=config,
                best_metric=best_metric,
            )

        if args.debug:
            if is_main_process():
                logger.info("Debug mode: stopping after 1 epoch")
            break

    # Final save
    total_time = time.time() - start_time
    if is_main_process():
        logger.info(f"Training complete in {total_time/3600:.1f}h")

        # Save final model
        final_dir = output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.module.state_dict(), final_dir / "model.pt")
        tokenizer.save_pretrained(str(final_dir))
        logger.info(f"Final model saved to {final_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
