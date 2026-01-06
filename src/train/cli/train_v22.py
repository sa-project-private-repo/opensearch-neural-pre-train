"""
CLI for V22 curriculum training.

Usage:
    python -m train v22
    python -m train v22 --config configs/train_v22.yaml
    python -m train v22 --resume
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from src.train.config import V22Config, load_config
from src.train.config.v22 import create_default_v22_config
from src.train.core import SPLADETrainer, CheckpointManager
from src.train.core.hooks import (
    LoggingHook,
    CheckpointHook,
    CurriculumHook,
    GradientMonitorHook,
)
from src.train.data import load_training_data, create_dataloader
from src.train.data.collator import create_tokenizer
from src.train.utils import setup_logging, TensorBoardLogger


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="V22 Curriculum Training for SPLADE Neural Sparse",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint to resume from",
    )

    # Override options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )

    parser.add_argument(
        "--train-files",
        type=str,
        nargs="+",
        default=None,
        help="Training data files (glob patterns supported)",
    )

    parser.add_argument(
        "--val-files",
        type=str,
        nargs="+",
        default=None,
        help="Validation data files",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Pretrained model name",
    )

    parser.add_argument(
        "--mixed-precision",
        type=str,
        choices=["fp16", "bf16", "none"],
        default=None,
        help="Mixed precision mode",
    )

    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum learning",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> V22Config:
    """
    Build configuration from args and config file.

    Args:
        args: Parsed command line arguments

    Returns:
        V22Config instance
    """
    # Build overrides from args
    overrides = {}

    if args.output_dir:
        overrides.setdefault("training", {})["output_dir"] = args.output_dir
    if args.epochs:
        overrides.setdefault("training", {})["num_epochs"] = args.epochs
    if args.batch_size:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.learning_rate:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.train_files:
        overrides.setdefault("data", {})["train_files"] = args.train_files
    if args.val_files:
        overrides.setdefault("data", {})["val_files"] = args.val_files
    if args.model_name:
        overrides.setdefault("model", {})["name"] = args.model_name
    if args.mixed_precision:
        mp = args.mixed_precision if args.mixed_precision != "none" else None
        overrides.setdefault("training", {})["mixed_precision"] = mp
    if args.no_curriculum:
        overrides["enable_curriculum"] = False
    if args.seed:
        overrides["seed"] = args.seed

    # Load config
    if args.config:
        config = load_config(args.config, V22Config, overrides)
    else:
        # Create default config if no config file
        train_files = args.train_files or ["data/v22.0/*.jsonl"]
        val_files = args.val_files or []

        config = create_default_v22_config(
            train_files=train_files,
            val_files=val_files,
            output_dir=args.output_dir or "outputs/train_v22",
            model_name=args.model_name or "skt/kobert-base-v1",
            batch_size=args.batch_size or 64,
            num_epochs=args.epochs or 30,
        )

        # Apply remaining overrides
        if args.learning_rate:
            config.training.learning_rate = args.learning_rate
        if args.mixed_precision:
            mp = args.mixed_precision if args.mixed_precision != "none" else None
            config.training.mixed_precision = mp
        if args.no_curriculum:
            config.enable_curriculum = False
        if args.seed:
            config.seed = args.seed

    return config


def setup_training(
    config: V22Config,
    resume_from: Optional[str] = None,
) -> SPLADETrainer:
    """
    Setup training components.

    Args:
        config: Training configuration
        resume_from: Optional checkpoint path to resume from

    Returns:
        Configured SPLADETrainer
    """
    # Import model and loss
    from src.model.splade_model import create_splade_model
    from src.model.losses import SPLADELossV22

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Create tokenizer
    logger.info(f"Loading tokenizer: {config.model.name}")
    tokenizer = create_tokenizer(config.model.name)

    # Create model
    logger.info(f"Creating model: {config.model.name}")
    model = create_splade_model(
        model_name=config.model.name,
        dropout=config.model.dropout,
        use_expansion=config.model.use_expansion,
        expansion_mode=config.model.expansion_mode,
    )

    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Create loss function
    loss_fn = SPLADELossV22(
        lambda_infonce=config.loss.lambda_infonce,
        lambda_self=config.loss.lambda_self,
        lambda_positive=config.loss.lambda_positive,
        lambda_margin=config.loss.lambda_margin,
        lambda_flops=config.loss.lambda_flops,
        lambda_min_act=config.loss.lambda_min_act,
        temperature=config.loss.temperature,
        margin=config.loss.margin,
        top_k=config.loss.top_k,
        min_activation=config.loss.min_activation,
    )

    # Load training data
    logger.info(f"Loading training data from: {config.data.train_files}")
    train_dataset = load_training_data(config.data.train_files)
    logger.info(f"Training samples: {len(train_dataset):,}")

    # Log data distribution
    pair_counts = train_dataset.get_pair_type_counts()
    logger.info("Data distribution:")
    for pair_type, count in sorted(pair_counts.items()):
        pct = 100 * count / len(train_dataset)
        logger.info(f"  {pair_type}: {count:,} ({pct:.1f}%)")

    # Load validation data
    val_dataset = None
    if config.data.val_files:
        logger.info(f"Loading validation data from: {config.data.val_files}")
        val_dataset = load_training_data(config.data.val_files)
        logger.info(f"Validation samples: {len(val_dataset):,}")

    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        tokenizer,
        batch_size=config.data.batch_size,
        max_length=config.data.max_length,
        num_workers=config.data.num_workers,
        shuffle=True,
        use_in_batch_negatives=True,
    )

    val_dataloader = None
    if val_dataset:
        val_dataloader = create_dataloader(
            val_dataset,
            tokenizer,
            batch_size=config.data.batch_size,
            max_length=config.data.max_length,
            num_workers=config.data.num_workers,
            shuffle=False,
            use_in_batch_negatives=True,
        )

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir=config.training.output_dir,
        keep_last_n=config.training.keep_last_n_checkpoints,
        save_best=True,
        metric_for_best="val_loss" if val_dataloader else "train_loss",
    )

    # Create TensorBoard logger
    tb_logger = TensorBoardLogger(
        log_dir=str(Path(config.training.output_dir) / "tensorboard"),
        experiment_name=config.training.experiment_name,
    )

    # Create hooks
    hooks = [
        LoggingHook(
            log_every_n_steps=config.training.log_every_n_steps,
            tensorboard_writer=tb_logger.writer,
        ),
        CheckpointHook(
            checkpoint_manager=checkpoint_manager,
            save_every_n_epochs=config.training.save_every_n_epochs,
            save_every_n_steps=config.training.save_every_n_steps,
        ),
        GradientMonitorHook(
            log_every_n_steps=config.training.log_every_n_steps * 2,
            tensorboard_writer=tb_logger.writer,
        ),
    ]

    # Add curriculum hook if enabled
    if config.enable_curriculum:
        hooks.append(CurriculumHook(config))
        logger.info("Curriculum learning enabled")
        for i, phase in enumerate(config.curriculum_phases, 1):
            logger.info(f"  Phase {i}: epochs {phase.start_epoch}-{phase.end_epoch}")
            logger.info(f"    Temperature: {phase.temperature}")
            logger.info(f"    Lambda InfoNCE: {phase.lambda_infonce}")

    # Create trainer
    trainer = SPLADETrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        config=config,
        checkpoint_manager=checkpoint_manager,
        hooks=hooks,
    )

    # Resume from checkpoint if requested
    if resume_from:
        trainer.resume_from_checkpoint(resume_from)

    return trainer


def main() -> int:
    """Main entry point for V22 training."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    output_dir = args.output_dir or "outputs/train_v22"
    setup_logging(output_dir=output_dir, level=log_level)

    logger.info("SPLADE V22 Curriculum Training")
    logger.info("=" * 60)

    try:
        # Build config
        config = build_config(args)
        config.validate()

        # Log configuration
        logger.info("Configuration:")
        logger.info(f"  Model: {config.model.name}")
        logger.info(f"  Batch size: {config.data.batch_size}")
        logger.info(f"  Learning rate: {config.training.learning_rate}")
        logger.info(f"  Epochs: {config.training.num_epochs}")
        logger.info(f"  Mixed precision: {config.training.mixed_precision}")
        logger.info(f"  Output dir: {config.training.output_dir}")

        # Determine resume checkpoint
        resume_from = None
        if args.checkpoint:
            resume_from = args.checkpoint
        elif args.resume:
            # Find latest checkpoint
            ckpt_manager = CheckpointManager(output_dir=config.training.output_dir)
            if ckpt_manager.has_checkpoint():
                resume_from = str(ckpt_manager.get_latest_checkpoint_path())
                logger.info(f"Resuming from: {resume_from}")
            else:
                logger.warning("No checkpoint found to resume from")

        # Setup and run training
        trainer = setup_training(config, resume_from)
        results = trainer.train()

        # Log results
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"  Epochs completed: {results['epochs_completed']}")
        logger.info(f"  Final train loss: {results['final_train_loss']:.4f}")
        if results['best_val_loss'] < float('inf'):
            logger.info(f"  Best val loss: {results['best_val_loss']:.4f}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
