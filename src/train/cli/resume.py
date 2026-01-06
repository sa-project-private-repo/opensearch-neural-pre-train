"""
CLI for resuming training from checkpoint.

Usage:
    python -m train resume
    python -m train resume --checkpoint outputs/train_v22/checkpoint_epoch5_step1000
"""

import argparse
import logging
import sys
from pathlib import Path

from src.train.cli.train_v22 import build_config, setup_training
from src.train.config import V22Config, load_config
from src.train.core import CheckpointManager
from src.train.utils import setup_logging


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Resume SPLADE training from checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory. If not specified, uses latest.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/train_v22",
        help="Output directory to search for checkpoints",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (uses checkpoint config if not specified)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def find_checkpoint(args: argparse.Namespace) -> str:
    """
    Find checkpoint to resume from.

    Args:
        args: Command line arguments

    Returns:
        Path to checkpoint directory

    Raises:
        FileNotFoundError: If no checkpoint found
    """
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        return str(checkpoint_path)

    # Search in output directory
    ckpt_manager = CheckpointManager(output_dir=args.output_dir)
    if not ckpt_manager.has_checkpoint():
        raise FileNotFoundError(f"No checkpoints found in: {args.output_dir}")

    latest = ckpt_manager.get_latest_checkpoint_path()
    return str(latest)


def load_checkpoint_config(checkpoint_path: str) -> V22Config:
    """
    Load configuration from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        V22Config instance
    """
    import json

    info_path = Path(checkpoint_path) / "checkpoint_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Checkpoint info not found: {info_path}")

    with open(info_path, "r") as f:
        info = json.load(f)

    if "config" in info:
        return V22Config(**info["config"])
    else:
        raise ValueError("Checkpoint does not contain configuration")


def main() -> int:
    """Main entry point for resume command."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(output_dir=args.output_dir, level=log_level)

    logger.info("Resuming SPLADE Training")
    logger.info("=" * 60)

    try:
        # Find checkpoint
        checkpoint_path = find_checkpoint(args)
        logger.info(f"Checkpoint: {checkpoint_path}")

        # Load config
        if args.config:
            config = load_config(args.config, V22Config)
            logger.info(f"Using config file: {args.config}")
        else:
            try:
                config = load_checkpoint_config(checkpoint_path)
                logger.info("Using config from checkpoint")
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Could not load config from checkpoint: {e}")
                logger.info("Using default configuration")

                # Create default config
                from src.train.config.v22 import create_default_v22_config
                config = create_default_v22_config(
                    train_files=["data/v22.0/*.jsonl"],
                    output_dir=args.output_dir,
                )

        # Setup and run training
        trainer = setup_training(config, resume_from=checkpoint_path)
        results = trainer.train()

        # Log results
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"  Epochs completed: {results['epochs_completed']}")
        logger.info(f"  Final train loss: {results['final_train_loss']:.4f}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
