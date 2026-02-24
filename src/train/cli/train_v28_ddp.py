"""
DDP CLI for V28 training on multiple GPUs.

Supports 8x GPU DDP for efficient large-scale training.

Usage:
    # Single node, multiple GPUs
    torchrun --nproc_per_node=8 -m src.train.cli.train_v28_ddp

    # With custom config
    torchrun --nproc_per_node=8 -m src.train.cli.train_v28_ddp \
        --config configs/train_v28_b200.yaml

    # Resume training
    torchrun --nproc_per_node=8 -m src.train.cli.train_v28_ddp --resume
"""


import logging
import os
import sys
from pathlib import Path
import argparse
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.model.splade_xlmr import SPLADEDocContextGated, SPLADEDocXLMR
from src.train.config.v28 import V28Config, V28LossConfig, V28ModelConfig
from src.train.config.base import DataConfig, TrainingConfig
from src.train.core.checkpoint import CheckpointManager
from src.train.data import load_training_data
from src.train.data.collator import create_tokenizer
from src.train.data.dataloader import TripletCollator
from src.train.idf import load_or_compute_idf, create_stopword_mask_v26, get_special_token_ids_only
from src.train.idf.korean_tokens import load_or_compute_korean_tokens
from src.train.utils import TensorBoardLogger, setup_logging


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="V28 Multi-GPU Training with DDP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--config", type=str, default=None, help="Config YAML file path")
    parser.add_argument("--output-dir", type=str, default="outputs/train_v28_ddp", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint to resume from")

    # V28 specific - FIXED parameters
    parser.add_argument("--non-korean-penalty", type=float, default=5.0,
                        help="Penalty for non-Korean tokens (reduced from 100.0 to prevent collapse)")
    parser.add_argument("--lambda-language", type=float, default=0.3,
                        help="Weight for language filtering loss")
    parser.add_argument("--no-context-gate", action="store_true", help="Disable context gate")
    parser.add_argument("--no-language-filtering", action="store_true", help="Disable language filtering")

    # Gradient accumulation for effective batch size
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return dist.get_rank() == 0


def create_v28_config(args: argparse.Namespace) -> V28Config:
    """Create V28 config from YAML file or defaults."""
    if args.config:
        from src.train.config.loader import load_config
        config = load_config(config_path=args.config, config_type=V28Config)
        # CLI overrides
        if args.output_dir != "outputs/train_v28_ddp":
            config.training.output_dir = args.output_dir
        if args.epochs != 30:
            config.training.num_epochs = args.epochs
        if args.lr != 3e-5:
            config.training.learning_rate = args.lr
        if args.batch_size != 32:
            config.data.batch_size = args.batch_size
        if args.grad_accum != 2:
            config.training.gradient_accumulation_steps = args.grad_accum
        if args.no_language_filtering:
            config.loss.enable_language_filtering = False
        if args.no_context_gate:
            config.model.use_context_gate = False
            config.loss.use_context_gate = False
        return config

    # Fallback: build config from CLI args (legacy behavior)
    config = V28Config(
        model=V28ModelConfig(
            name="xlm-roberta-base",
            dropout=0.1,
            use_expansion=True,
            expansion_mode="mlm",
            use_context_gate=not args.no_context_gate,
            context_gate_hidden=256,
            context_attention_heads=4,
        ),
        data=DataConfig(
            train_files=["data/v24.0/train_*.jsonl"],
            val_files=["data/v24.0/val.jsonl"],
            batch_size=args.batch_size,
            max_length=192,
            num_workers=4,
        ),
        loss=V28LossConfig(
            enable_language_filtering=not args.no_language_filtering,
            non_korean_penalty=args.non_korean_penalty,
            lambda_language=args.lambda_language,
            korean_token_penalty=0.0,
            use_context_gate=not args.no_context_gate,
            context_gate_hidden=256,
            context_attention_heads=4,
            lambda_infonce=2.5,
            lambda_self=1.0,
            lambda_positive=0.5,
            lambda_flops=1e-5,
            lambda_min_act=0.5,
            temperature=0.07,
            top_k=256,
            min_activation=0.1,
            use_stopword_mask=True,
            stopword_penalty=50.0,
        ),
        training=TrainingConfig(
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            gradient_clip=1.0,
            gradient_accumulation_steps=args.grad_accum,
            mixed_precision="bf16",
            output_dir=args.output_dir,
            experiment_name="splade_v28_ddp_fixed",
            log_every_n_steps=50,
            save_every_n_epochs=5,
        ),
        seed=args.seed,
        enable_curriculum=True,
    )
    return config


def create_model(config: V28Config, device: torch.device) -> nn.Module:
    """Create V28 model."""
    if config.model.use_context_gate:
        model = SPLADEDocContextGated(
            model_name=config.model.name,
            dropout=config.model.dropout,
            use_mlm_head=config.model.use_expansion,
            gate_hidden=config.model.context_gate_hidden,
            gate_heads=config.model.context_attention_heads,
            top_k_sparse=getattr(config.model, 'top_k_sparse', 0),
        )
        logger.info("Created SPLADEDocContextGated model")
    else:
        model = SPLADEDocXLMR(
            model_name=config.model.name,
            dropout=config.model.dropout,
            use_mlm_head=config.model.use_expansion,
        )
        logger.info("Created SPLADEDocXLMR model (no context gate)")

    return model.to(device)


def create_loss_fn(config: V28Config, tokenizer, device: torch.device) -> nn.Module:
    """Create loss function (V29 with separate query/doc FLOPS, or V28 fallback)."""
    # Get special token IDs
    special_token_ids = get_special_token_ids_only(tokenizer)

    # Load Korean token IDs if language filtering enabled
    korean_token_ids = None
    if config.loss.enable_language_filtering:
        korean_tokens_path = f"{config.training.output_dir}/korean_token_ids.json"
        korean_token_ids = load_or_compute_korean_tokens(
            cache_path=korean_tokens_path,
            tokenizer=tokenizer,
            recompute=False,
        )
        if is_main_process():
            logger.info(f"Loaded Korean token IDs: {len(korean_token_ids):,}")

    # Load IDF weights
    idf_cache_path = config.get_idf_cache_path()
    idf_weights = load_or_compute_idf(
        cache_path=idf_cache_path,
        corpus_files=config.data.train_files,
        tokenizer=tokenizer,
        recompute=False,
    )

    # Create stopword mask
    stopword_mask = None
    if config.loss.use_stopword_mask:
        stopword_mask = create_stopword_mask_v26(tokenizer)

    # Use V29 (SPLADE v2 separate FLOPS) if lambda_flops_q/d are set
    lambda_flops_q = getattr(config.loss, 'lambda_flops_q', 0.0)
    lambda_flops_d = getattr(config.loss, 'lambda_flops_d', 0.0)

    if lambda_flops_q > 0 or lambda_flops_d > 0:
        from src.model.losses import SPLADELossV29

        loss_fn = SPLADELossV29(
            idf_weights=idf_weights,
            special_token_ids=special_token_ids,
            lambda_flops_q=lambda_flops_q,
            lambda_flops_d=lambda_flops_d,
            korean_token_ids=korean_token_ids,
            lambda_language=config.loss.lambda_language,
            non_korean_penalty=config.loss.non_korean_penalty,
            korean_penalty=config.loss.korean_token_penalty,
            enable_language_filtering=config.loss.enable_language_filtering,
            lambda_infonce=config.loss.lambda_infonce,
            lambda_self=config.loss.lambda_self,
            lambda_positive=config.loss.lambda_positive,
            lambda_flops=config.loss.lambda_flops,  # 0.0 (unified disabled)
            lambda_min_act=config.loss.lambda_min_act,
            lambda_kd=getattr(config.loss, 'lambda_kd', 2.0),
            temperature=config.loss.temperature,
            top_k=config.loss.top_k,
            min_activation=config.loss.min_activation,
            stopword_mask=stopword_mask,
            stopword_penalty=config.loss.stopword_penalty,
        )
        if is_main_process():
            logger.info(
                f"Using SPLADELossV29 (SPLADE v2): "
                f"lambda_flops_q={lambda_flops_q}, lambda_flops_d={lambda_flops_d}"
            )
    else:
        from src.model.losses import SPLADELossV28

        loss_fn = SPLADELossV28(
            idf_weights=idf_weights,
            special_token_ids=special_token_ids,
            korean_token_ids=korean_token_ids,
            lambda_language=config.loss.lambda_language,
            non_korean_penalty=config.loss.non_korean_penalty,
            korean_penalty=config.loss.korean_token_penalty,
            enable_language_filtering=config.loss.enable_language_filtering,
            lambda_infonce=config.loss.lambda_infonce,
            lambda_self=config.loss.lambda_self,
            lambda_positive=config.loss.lambda_positive,
            lambda_flops=config.loss.lambda_flops,
            lambda_min_act=config.loss.lambda_min_act,
            temperature=config.loss.temperature,
            top_k=config.loss.top_k,
            min_activation=config.loss.min_activation,
            stopword_mask=stopword_mask,
            stopword_penalty=config.loss.stopword_penalty,
            flops_warmup_steps=getattr(config.loss, 'flops_warmup_steps', 0),
        )
        if is_main_process():
            logger.info(f"Using SPLADELossV28: lambda_flops={config.loss.lambda_flops}")

    return loss_fn.to(device)


def create_dataloader_ddp(
    dataset,
    tokenizer,
    batch_size: int,
    max_length: int,
    num_workers: int,
    is_train: bool = True,
    query_max_length: Optional[int] = None,
    doc_max_length: Optional[int] = None,
) -> DataLoader:
    """Create distributed dataloader."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=is_train,
    )

    collator = TripletCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        query_max_length=query_max_length,
        doc_max_length=doc_max_length,
        use_in_batch_negatives=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=is_train,
    )

    return dataloader


def train_epoch(
    model: DDP,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: AdamW,
    scheduler,
    scaler: Optional[GradScaler],
    config: V28Config,
    epoch: int,
    global_step: int,
    device: torch.device,
    tb_logger: Optional[TensorBoardLogger] = None,
) -> tuple[float, int]:
    """Train one epoch with DDP."""
    model.train()
    dataloader.sampler.set_epoch(epoch)  # Important for shuffling

    total_loss = 0.0
    num_batches = 0

    # Only show progress bar on main process
    if is_main_process():
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        progress_bar = dataloader

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        query_input_ids = batch["query_input_ids"].to(device)
        query_attention_mask = batch["query_attention_mask"].to(device)
        positive_input_ids = batch["positive_input_ids"].to(device)
        positive_attention_mask = batch["positive_attention_mask"].to(device)

        # Forward with mixed precision
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            # Encode anchor - pass explicit token_type_ids to avoid HuggingFace internal buffer issue
            # (XLM-RoBERTa's internal token_type_ids buffer causes inplace op errors)
            query_token_type_ids = torch.zeros_like(query_input_ids)
            anchor_repr, _ = model(query_input_ids, query_attention_mask, query_token_type_ids)

            # Encode positive
            positive_token_type_ids = torch.zeros_like(positive_input_ids)
            positive_repr, _ = model(positive_input_ids, positive_attention_mask, positive_token_type_ids)

            # In-batch negatives
            negative_repr = torch.roll(positive_repr, shifts=1, dims=0)

            # Compute loss - use contiguous copies of input_ids to avoid inplace op issues
            loss, loss_dict = loss_fn(
                anchor_repr=anchor_repr,
                positive_repr=positive_repr,
                negative_repr=negative_repr,
                anchor_input_ids=query_input_ids.detach().clone(),
                anchor_attention_mask=query_attention_mask.detach(),
                positive_input_ids=positive_input_ids.detach().clone(),
                positive_attention_mask=positive_attention_mask.detach(),
            )

        # Backward
        loss = loss / config.training.gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step
        if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging
            if is_main_process() and global_step % config.training.log_every_n_steps == 0:
                lr = scheduler.get_last_lr()[0]

                # Get metrics from loss_fn
                semantic_ratio = loss_fn.get_average_semantic_ratio() if hasattr(loss_fn, 'get_average_semantic_ratio') else 0
                korean_ratio = loss_fn.get_average_korean_ratio() if hasattr(loss_fn, 'get_average_korean_ratio') else 0
                active_tokens = loss_dict.get("active_tokens", 0)
                flops_q = loss_dict.get("flops_q", 0)
                flops_d = loss_dict.get("flops_d", 0)

                logger.info(
                    f"Step {global_step} - loss: {loss.item() * config.training.gradient_accumulation_steps:.4f}, "
                    f"lr: {lr:.2e}, semantic_ratio: {semantic_ratio:.4f}, korean_ratio: {korean_ratio:.4f}, "
                    f"active_tokens: {active_tokens:.0f}, flops_q: {flops_q:.4f}, flops_d: {flops_d:.4f}"
                )

                if tb_logger:
                    tb_logger.log_scalar("train/loss", loss.item() * config.training.gradient_accumulation_steps, global_step)
                    tb_logger.log_scalar("train/lr", lr, global_step)
                    tb_logger.log_scalar("train/semantic_ratio", semantic_ratio, global_step)
                    tb_logger.log_scalar("train/korean_ratio", korean_ratio, global_step)
                    tb_logger.log_scalar("train/active_tokens", active_tokens, global_step)
                    if flops_q > 0:
                        tb_logger.log_scalar("train/flops_q", flops_q, global_step)
                        tb_logger.log_scalar("train/flops_d", flops_d, global_step)

        total_loss += loss.item() * config.training.gradient_accumulation_steps
        num_batches += 1

        if is_main_process() and isinstance(progress_bar, tqdm):
            progress_bar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "step": global_step,
            })

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


def save_checkpoint(
    model: DDP,
    optimizer: AdamW,
    scheduler,
    epoch: int,
    step: int,
    output_dir: str,
    is_best: bool = False,
):
    """Save checkpoint (only on main process)."""
    if not is_main_process():
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights (unwrap DDP)
    model_state = model.module.state_dict()

    checkpoint_dir = output_dir / f"checkpoint_epoch{epoch}_step{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model_state, checkpoint_dir / "model.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

    # Save checkpoint info
    info = {"epoch": epoch, "step": step}
    torch.save(info, checkpoint_dir / "checkpoint_info.pt")

    logger.info(f"Saved checkpoint to {checkpoint_dir}")


def main():
    """Main training function."""
    args = parse_args()

    # Setup distributed
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Enable anomaly detection for debugging (only on rank 0)
    if local_rank == 0 and args.debug:
        torch.autograd.set_detect_anomaly(True)

    # Setup logging (only on main process)
    if is_main_process():
        log_level = logging.DEBUG if args.debug else logging.INFO
        setup_logging(output_dir=args.output_dir, level=log_level)

        logger.info("=" * 70)
        logger.info("V28 Multi-GPU Training with DDP (FIXED)")
        logger.info("=" * 70)
        logger.info(f"World size: {dist.get_world_size()}")
        logger.info(f"Local rank: {local_rank}")
        logger.info("")
        logger.info("=== FIXES APPLIED ===")
        logger.info(f"  non_korean_penalty: {args.non_korean_penalty} (was 100.0)")
        logger.info(f"  lambda_language: {args.lambda_language} (was 0.5)")
        logger.info("=" * 70)

    # Set seed
    torch.manual_seed(args.seed + dist.get_rank())
    torch.cuda.manual_seed_all(args.seed + dist.get_rank())

    # Create config
    config = create_v28_config(args)

    if is_main_process():
        config.validate()
        logger.info(f"Batch size per GPU: {config.data.batch_size}")
        logger.info(f"Effective batch size: {config.data.batch_size * dist.get_world_size() * config.training.gradient_accumulation_steps}")
        logger.info(f"Epochs: {config.training.num_epochs}")

    # Create tokenizer
    tokenizer = create_tokenizer(config.model.name)

    # Create model
    model = create_model(config, device)

    # Log model info
    if is_main_process():
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create loss function
    loss_fn = create_loss_fn(config, tokenizer, device)

    # Load data
    if is_main_process():
        logger.info(f"Loading training data from: {config.data.train_files}")

    train_dataset = load_training_data(config.data.train_files)

    if is_main_process():
        logger.info(f"Training samples: {len(train_dataset):,}")

    # Create dataloader
    train_dataloader = create_dataloader_ddp(
        train_dataset,
        tokenizer,
        batch_size=config.data.batch_size,
        max_length=config.data.max_length,
        num_workers=config.data.num_workers,
        is_train=True,
        query_max_length=config.data.query_max_length,
        doc_max_length=config.data.doc_max_length,
    )

    # Create optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.training.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.training.learning_rate)

    # Create scheduler
    total_steps = len(train_dataloader) * config.training.num_epochs // config.training.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main_process():
        logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    # GradScaler for mixed precision (not needed for bf16)
    scaler = None

    # TensorBoard logger
    tb_logger = None
    if is_main_process():
        tb_logger = TensorBoardLogger(
            log_dir=str(Path(args.output_dir) / "tensorboard"),
            experiment_name=config.training.experiment_name,
        )

    # Resume from checkpoint
    start_epoch = 1
    global_step = 0

    if args.resume or args.checkpoint:
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            # Find latest checkpoint
            output_dir = Path(args.output_dir)
            checkpoints = sorted(output_dir.glob("checkpoint_epoch*"))
            if checkpoints:
                checkpoint_path = str(checkpoints[-1])

        if checkpoint_path:
            if is_main_process():
                logger.info(f"Resuming from {checkpoint_path}")

            model_state = torch.load(Path(checkpoint_path) / "model.pt", map_location=device)
            model.module.load_state_dict(model_state)

            optimizer.load_state_dict(torch.load(Path(checkpoint_path) / "optimizer.pt", map_location=device))
            scheduler.load_state_dict(torch.load(Path(checkpoint_path) / "scheduler.pt", map_location=device))

            info = torch.load(Path(checkpoint_path) / "checkpoint_info.pt")
            start_epoch = info["epoch"] + 1
            global_step = info["step"]

            if is_main_process():
                logger.info(f"Resumed from epoch {start_epoch - 1}, step {global_step}")

    # Training loop
    if is_main_process():
        logger.info("Starting training...")

    try:
        for epoch in range(start_epoch, config.training.num_epochs + 1):
            if is_main_process():
                logger.info(f"\n{'=' * 50}")
                logger.info(f"Epoch {epoch}/{config.training.num_epochs}")
                logger.info(f"{'=' * 50}")

            avg_loss, global_step = train_epoch(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                epoch=epoch,
                global_step=global_step,
                device=device,
                tb_logger=tb_logger,
            )

            if is_main_process():
                logger.info(f"Epoch {epoch} completed. Avg loss: {avg_loss:.4f}")

                # Check semantic ratio
                semantic_ratio = loss_fn.get_average_semantic_ratio() if hasattr(loss_fn, 'get_average_semantic_ratio') else 0
                korean_ratio = loss_fn.get_average_korean_ratio() if hasattr(loss_fn, 'get_average_korean_ratio') else 0

                logger.info(f"  Semantic ratio: {semantic_ratio:.4f}")
                logger.info(f"  Korean ratio: {korean_ratio:.4f}")

                # Warning if collapsing
                if semantic_ratio < 1.0:
                    logger.warning("WARNING: semantic_ratio < 1.0 - model may be collapsing!")

            # Save checkpoint
            if epoch % config.training.save_every_n_epochs == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, args.output_dir)

            # Sync across processes
            dist.barrier()

        # Final checkpoint
        save_checkpoint(model, optimizer, scheduler, config.training.num_epochs, global_step, args.output_dir)

        if is_main_process():
            logger.info("\nTraining completed!")

    except KeyboardInterrupt:
        if is_main_process():
            logger.info("Training interrupted. Saving checkpoint...")
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, args.output_dir)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
