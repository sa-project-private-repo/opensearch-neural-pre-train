"""
DDP CLI for V28 training on multiple GPUs.

Usage:
    torchrun --nproc_per_node=8 -m src.train.cli.train_v28_ddp
    torchrun --nproc_per_node=8 -m src.train.cli.train_v28_ddp \
        --config configs/train_v28_b200.yaml
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.train.cli.train_v28 import (
    build_config,
    create_v28_model,
    parse_args,
)
from src.train.config.v28 import V28Config
from src.train.core.checkpoint import CheckpointManager
from src.train.core.collapse_detector import CollapseDetectionHook
from src.train.core.ddp_trainer import DDPSPLADETrainer
from src.train.core.hooks import (
    CheckpointHook,
    CurriculumHook,
    GradientMonitorHook,
    LoggingHook,
)
from src.train.data import (
    TripletCollator,
    create_dataloader,
    load_training_data,
)
from src.train.data.collator import create_tokenizer
from src.train.idf import (
    create_stopword_mask_v26,
    get_special_token_ids_only,
    load_or_compute_idf,
)
from src.train.idf.korean_tokens import load_or_compute_korean_tokens
from src.train.utils import TensorBoardLogger, setup_logging

logger = logging.getLogger(__name__)


def create_ddp_dataloader(
    dataset: torch.utils.data.Dataset,
    tokenizer: "PreTrainedTokenizer",
    batch_size: int,
    max_length: int,
    num_workers: int,
    world_size: int,
    rank: int,
    shuffle: bool = True,
    use_in_batch_negatives: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    """
    Create a DataLoader with DistributedSampler for DDP.

    Args:
        dataset: Dataset to load from
        tokenizer: HuggingFace tokenizer
        batch_size: Per-GPU batch size
        max_length: Maximum sequence length
        num_workers: Number of worker processes
        world_size: Total number of GPUs
        rank: Current GPU rank
        shuffle: Whether to shuffle data
        use_in_batch_negatives: Use in-batch negatives
        pin_memory: Pin memory for GPU transfer
        drop_last: Drop last incomplete batch

    Returns:
        Tuple of (DataLoader, DistributedSampler)
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )

    collator = TripletCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        use_in_batch_negatives=use_in_batch_negatives,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # Sampler handles shuffling
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )

    return dataloader, sampler


def setup_ddp_training(
    config: V28Config,
    local_rank: int,
    world_size: int,
    resume_from: Optional[str] = None,
    korean_tokens_path: Optional[str] = None,
    recompute_korean_tokens: bool = False,
) -> DDPSPLADETrainer:
    """
    Setup DDP training components for V28.

    Args:
        config: V28 configuration
        local_rank: Local GPU rank
        world_size: Total number of GPUs
        resume_from: Optional checkpoint path
        korean_tokens_path: Path to cached Korean token IDs
        recompute_korean_tokens: Force recomputation

    Returns:
        Configured DDPSPLADETrainer
    """
    from src.model.losses import SPLADELossV28

    is_main = local_rank == 0

    # Set random seed (offset by rank for diversity)
    torch.manual_seed(config.seed + local_rank)
    torch.cuda.manual_seed_all(config.seed + local_rank)

    # Create tokenizer
    if is_main:
        logger.info(f"Loading tokenizer: {config.model.name}")
    tokenizer = create_tokenizer(config.model.name)

    # Create model
    if is_main:
        logger.info(f"Creating V28 model: {config.model.name}")
    model = create_v28_model(config)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        trainable = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
        logger.info(f"Model parameters: {num_params:,}")
        logger.info(f"Trainable parameters: {trainable:,}")

    # === Korean token IDs for language filtering ===
    korean_token_ids = None
    if config.loss.enable_language_filtering:
        if korean_tokens_path is None:
            korean_tokens_path = (
                f"{config.training.output_dir}/korean_token_ids.json"
            )
        korean_token_ids = load_or_compute_korean_tokens(
            cache_path=korean_tokens_path,
            tokenizer=tokenizer,
            recompute=recompute_korean_tokens,
        )
        if is_main:
            logger.info(
                f"Korean token IDs: {len(korean_token_ids):,}"
            )

    # Special token IDs
    special_token_ids = get_special_token_ids_only(tokenizer)

    # IDF weights
    idf_cache_path = config.get_idf_cache_path()
    idf_weights = load_or_compute_idf(
        cache_path=idf_cache_path,
        corpus_files=config.data.train_files,
        tokenizer=tokenizer,
        recompute=config.loss.recompute_idf,
        smoothing=config.loss.idf_smoothing,
    )

    # Stopword mask
    stopword_mask = None
    if config.loss.use_stopword_mask:
        stopword_mask = create_stopword_mask_v26(tokenizer)

    # Create loss function (with warmup + collapse detection)
    loss_fn = SPLADELossV28(
        idf_weights=idf_weights,
        special_token_ids=special_token_ids,
        korean_token_ids=korean_token_ids,
        lambda_language=config.loss.lambda_language,
        non_korean_penalty=config.loss.non_korean_penalty,
        korean_penalty=config.loss.korean_token_penalty,
        enable_language_filtering=(
            config.loss.enable_language_filtering
        ),
        language_warmup_steps=(
            config.loss.language_warmup_steps
        ),
        language_penalty_max=(
            config.loss.language_penalty_max
        ),
        collapse_flops_threshold=(
            config.loss.collapse_flops_threshold
        ),
        collapse_check_window=(
            config.loss.collapse_check_window
        ),
        lambda_infonce=config.loss.lambda_infonce,
        lambda_self=config.loss.lambda_self,
        lambda_positive=config.loss.lambda_positive,
        lambda_flops=config.loss.lambda_flops,
        lambda_min_act=config.loss.lambda_min_act,
        lambda_kd=config.loss.lambda_kd,
        temperature=config.loss.temperature,
        kd_temperature=config.loss.kd_temperature,
        top_k=config.loss.top_k,
        min_activation=config.loss.min_activation,
        idf_alpha=config.loss.idf_alpha,
        special_penalty=config.loss.special_token_penalty,
        stopword_mask=stopword_mask,
        stopword_penalty=config.loss.stopword_penalty,
    )

    # Load training data
    if is_main:
        logger.info(
            f"Loading training data: {config.data.train_files}"
        )
    train_dataset = load_training_data(config.data.train_files)
    if is_main:
        logger.info(f"Training samples: {len(train_dataset):,}")
        pair_counts = train_dataset.get_pair_type_counts()
        for pair_type, count in sorted(pair_counts.items()):
            pct = 100 * count / len(train_dataset)
            logger.info(f"  {pair_type}: {count:,} ({pct:.1f}%)")

    # Load validation data
    val_dataset = None
    if config.data.val_files:
        val_dataset = load_training_data(config.data.val_files)
        if is_main:
            logger.info(
                f"Validation samples: {len(val_dataset):,}"
            )

    # Create DDP dataloaders
    train_dataloader, _ = create_ddp_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        batch_size=config.data.batch_size,
        max_length=config.data.max_length,
        num_workers=config.data.num_workers,
        world_size=world_size,
        rank=local_rank,
        shuffle=True,
        use_in_batch_negatives=True,
    )

    val_dataloader = None
    if val_dataset:
        val_dataloader, _ = create_ddp_dataloader(
            dataset=val_dataset,
            tokenizer=tokenizer,
            batch_size=config.data.batch_size,
            max_length=config.data.max_length,
            num_workers=config.data.num_workers,
            world_size=world_size,
            rank=local_rank,
            shuffle=False,
            use_in_batch_negatives=True,
        )

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir=config.training.output_dir,
        keep_last_n=config.training.keep_last_n_checkpoints,
        save_best=True,
        metric_for_best=(
            "val_loss" if val_dataloader else "train_loss"
        ),
    )

    # Hooks (only rank 0 gets TensorBoard)
    hooks = []
    tb_logger = None
    if is_main:
        tb_logger = TensorBoardLogger(
            log_dir=str(
                Path(config.training.output_dir) / "tensorboard"
            ),
            experiment_name=config.training.experiment_name,
        )

    hooks.append(
        LoggingHook(
            log_every_n_steps=config.training.log_every_n_steps,
            tensorboard_writer=(
                tb_logger.writer if tb_logger else None
            ),
        )
    )
    hooks.append(
        CheckpointHook(
            checkpoint_manager=checkpoint_manager,
            save_every_n_epochs=config.training.save_every_n_epochs,
            save_every_n_steps=config.training.save_every_n_steps,
        )
    )
    hooks.append(
        GradientMonitorHook(
            log_every_n_steps=(
                config.training.log_every_n_steps * 2
            ),
            tensorboard_writer=(
                tb_logger.writer if tb_logger else None
            ),
        )
    )

    # Add collapse detection hook
    if config.loss.enable_language_filtering:
        hooks.append(CollapseDetectionHook(
            flops_threshold=(
                config.loss.collapse_flops_threshold
            ),
            check_window=config.loss.collapse_check_window,
            check_every_n_steps=(
                config.training.log_every_n_steps
            ),
        ))
        if is_main:
            logger.info("Collapse detection enabled")

    if config.enable_curriculum:
        hooks.append(CurriculumHook(config))
        if is_main:
            logger.info("Curriculum learning enabled")

    # Create DDP trainer
    trainer = DDPSPLADETrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        config=config,
        checkpoint_manager=checkpoint_manager,
        hooks=hooks,
        local_rank=local_rank,
        world_size=world_size,
    )

    # Resume from checkpoint if requested
    if resume_from:
        trainer.resume_from_checkpoint(resume_from)

    return trainer


def main() -> int:
    """Main entry point for DDP V28 training."""
    # Get DDP environment variables (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    # Parse args
    args = parse_args()

    # Setup logging (WARNING for non-main ranks)
    log_level = logging.DEBUG if args.debug else logging.INFO
    if not is_main:
        log_level = logging.WARNING

    output_dir = args.output_dir or "outputs/train_v28_ddp"
    setup_logging(output_dir=output_dir, level=log_level)

    if is_main:
        logger.info("=" * 70)
        logger.info("SPLADE V28 DDP Training (Multi-GPU)")
        logger.info(
            "Language Filtering + Context-Gated Sparse Expansion"
        )
        logger.info("=" * 70)
        logger.info(f"World size: {world_size}")
        logger.info(f"Local rank: {local_rank}")
        logger.info("=" * 70)

    try:
        # Initialize distributed
        DDPSPLADETrainer.setup_distributed(
            local_rank, world_size
        )

        # Build config
        config = build_config(args)

        # Override output dir for DDP
        if args.output_dir:
            config.training.output_dir = args.output_dir

        config.validate()

        if is_main:
            logger.info("Configuration:")
            logger.info(f"  Model: {config.model.name}")
            logger.info(
                f"  Use context gate: "
                f"{config.model.use_context_gate}"
            )
            logger.info(
                f"  Per-GPU batch size: "
                f"{config.data.batch_size}"
            )
            logger.info(
                f"  Effective batch size: "
                f"{config.data.batch_size * world_size}"
            )
            logger.info(
                f"  Learning rate: "
                f"{config.training.learning_rate}"
            )
            logger.info(
                f"  Epochs: {config.training.num_epochs}"
            )
            logger.info(
                f"  Mixed precision: "
                f"{config.training.mixed_precision}"
            )
            logger.info(
                f"  Output dir: {config.training.output_dir}"
            )

        # Determine resume checkpoint
        resume_from = None
        if args.checkpoint:
            resume_from = args.checkpoint
        elif args.resume:
            ckpt_mgr = CheckpointManager(
                output_dir=config.training.output_dir
            )
            if ckpt_mgr.has_checkpoint():
                resume_from = str(
                    ckpt_mgr.get_latest_checkpoint_path()
                )
                if is_main:
                    logger.info(f"Resuming from: {resume_from}")

        # Setup and run training
        trainer = setup_ddp_training(
            config=config,
            local_rank=local_rank,
            world_size=world_size,
            resume_from=resume_from,
            korean_tokens_path=args.korean_tokens_path,
            recompute_korean_tokens=args.recompute_korean_tokens,
        )

        results = trainer.train()

        if is_main:
            logger.info("=" * 70)
            logger.info("Training Complete!")
            logger.info(
                f"  Epochs: {results['epochs_completed']}"
            )
            logger.info(
                f"  Final train loss: "
                f"{results['final_train_loss']:.4f}"
            )
            if results["best_val_loss"] < float("inf"):
                logger.info(
                    f"  Best val loss: "
                    f"{results['best_val_loss']:.4f}"
                )

            # Completion flag
            flag = (
                Path(config.training.output_dir)
                / "training_complete.flag"
            )
            flag.write_text(
                f"completed_epochs="
                f"{results['epochs_completed']}\n"
                f"world_size={world_size}\n"
            )

        # Cleanup
        trainer.cleanup()
        return 0

    except FileNotFoundError as e:
        if is_main:
            logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        if is_main:
            logger.error(f"Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        if is_main:
            logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        if is_main:
            logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
