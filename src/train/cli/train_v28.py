"""
CLI for V28 XLM-RoBERTa training with Language Filtering and Context Gate.

Key features:
- V28a: Korean language filtering (suppress non-Korean tokens)
- V28b: Context-gated sparse expansion (context-dependent activation)

Usage:
    python -m src.train v28
    python -m src.train v28 --config configs/train_v28.yaml
    python -m src.train v28 --resume
    python -m src.train v28 --no-context-gate  # V28a only (no context gate)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from src.train.config import load_config
from src.train.config.v28 import V28Config, create_default_v28_config
from src.train.core import SPLADETrainer, CheckpointManager
from src.train.core.collapse_detector import CollapseDetectionHook
from src.train.core.hooks import (
    CheckpointHook,
    CurriculumHook,
    GradientMonitorHook,
    LoggingHook,
)
from src.train.data import create_dataloader, load_training_data
from src.train.data.collator import create_tokenizer
from src.train.idf import (
    load_or_compute_idf,
    create_stopword_mask_v26,
    get_special_token_ids_only,
)
from src.train.idf.korean_tokens import (
    build_korean_token_ids,
    load_or_compute_korean_tokens,
    analyze_token_language_distribution,
)
from src.train.utils import TensorBoardLogger, setup_logging


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="V28 XLM-RoBERTa Training with Language Filtering and Context Gate",
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
        help="XLM-RoBERTa model name",
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
        "--no-kd",
        action="store_true",
        help="Disable knowledge distillation",
    )

    parser.add_argument(
        "--no-stopword-mask",
        action="store_true",
        help="Disable Korean stopword masking",
    )

    parser.add_argument(
        "--recompute-idf",
        action="store_true",
        help="Force recomputation of IDF weights",
    )

    parser.add_argument(
        "--idf-weights-path",
        type=str,
        default=None,
        help="Path to pre-computed IDF weights",
    )

    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Teacher model for knowledge distillation",
    )

    # V28 specific options
    parser.add_argument(
        "--no-language-filtering",
        action="store_true",
        help="Disable language filtering (V28a feature)",
    )

    parser.add_argument(
        "--no-context-gate",
        action="store_true",
        help="Disable context gate (V28b feature, run V28a only)",
    )

    parser.add_argument(
        "--lambda-language",
        type=float,
        default=None,
        help="Weight for language filtering loss (default: 0.5)",
    )

    parser.add_argument(
        "--non-korean-penalty",
        type=float,
        default=None,
        help="Penalty for non-Korean tokens (default: 100.0)",
    )

    parser.add_argument(
        "--context-gate-hidden",
        type=int,
        default=None,
        help="Hidden dimension for context gate (default: 256)",
    )

    parser.add_argument(
        "--context-attention-heads",
        type=int,
        default=None,
        help="Attention heads in context gate (default: 4)",
    )

    parser.add_argument(
        "--korean-tokens-path",
        type=str,
        default=None,
        help="Path to cached Korean token IDs JSON",
    )

    parser.add_argument(
        "--recompute-korean-tokens",
        action="store_true",
        help="Force recomputation of Korean token IDs",
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

    parser.add_argument(
        "--analyze-vocab",
        action="store_true",
        help="Analyze vocabulary language distribution and exit",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> V28Config:
    """
    Build configuration from args and config file.

    Args:
        args: Parsed command line arguments

    Returns:
        V28Config instance
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
    if args.no_kd:
        overrides.setdefault("knowledge_distillation", {})["enabled"] = False
    if args.no_stopword_mask:
        overrides.setdefault("loss", {})["use_stopword_mask"] = False
    if args.recompute_idf:
        overrides.setdefault("loss", {})["recompute_idf"] = True
    if args.idf_weights_path:
        overrides.setdefault("loss", {})["idf_weights_path"] = args.idf_weights_path
    if args.teacher_model:
        overrides.setdefault("knowledge_distillation", {})["teacher_model"] = args.teacher_model
    if args.seed:
        overrides["seed"] = args.seed

    # V28 specific overrides
    if args.no_language_filtering:
        overrides.setdefault("loss", {})["enable_language_filtering"] = False
    if args.no_context_gate:
        overrides.setdefault("loss", {})["use_context_gate"] = False
        overrides.setdefault("model", {})["use_context_gate"] = False
    if args.lambda_language:
        overrides.setdefault("loss", {})["lambda_language"] = args.lambda_language
    if args.non_korean_penalty:
        overrides.setdefault("loss", {})["non_korean_penalty"] = args.non_korean_penalty
    if args.context_gate_hidden:
        overrides.setdefault("loss", {})["context_gate_hidden"] = args.context_gate_hidden
        overrides.setdefault("model", {})["context_gate_hidden"] = args.context_gate_hidden
    if args.context_attention_heads:
        overrides.setdefault("loss", {})["context_attention_heads"] = args.context_attention_heads
        overrides.setdefault("model", {})["context_attention_heads"] = args.context_attention_heads

    # Load config
    if args.config:
        config = load_config(args.config, V28Config, overrides)
    else:
        # Create default config
        use_context_gate = not args.no_context_gate
        enable_language_filtering = not args.no_language_filtering

        config = create_default_v28_config(
            train_files=args.train_files,
            val_files=args.val_files,
            output_dir=args.output_dir or "outputs/train_v28",
            model_name=args.model_name or "xlm-roberta-base",
            batch_size=args.batch_size or 24,
            num_epochs=args.epochs or 25,
            idf_weights_path=args.idf_weights_path,
            use_context_gate=use_context_gate,
            enable_language_filtering=enable_language_filtering,
        )

        # Apply remaining overrides
        if args.learning_rate:
            config.training.learning_rate = args.learning_rate
        if args.mixed_precision:
            mp = args.mixed_precision if args.mixed_precision != "none" else None
            config.training.mixed_precision = mp
        if args.no_curriculum:
            config.enable_curriculum = False
        if args.no_kd:
            config.knowledge_distillation.enabled = False
        if args.no_stopword_mask:
            config.loss.use_stopword_mask = False
        if args.recompute_idf:
            config.loss.recompute_idf = True
        if args.teacher_model:
            config.knowledge_distillation.teacher_model = args.teacher_model
        if args.lambda_language:
            config.loss.lambda_language = args.lambda_language
        if args.non_korean_penalty:
            config.loss.non_korean_penalty = args.non_korean_penalty
        if args.context_gate_hidden:
            config.loss.context_gate_hidden = args.context_gate_hidden
            config.model.context_gate_hidden = args.context_gate_hidden
        if args.context_attention_heads:
            config.loss.context_attention_heads = args.context_attention_heads
            config.model.context_attention_heads = args.context_attention_heads
        if args.seed:
            config.seed = args.seed

    return config


def create_v28_model(config: V28Config) -> torch.nn.Module:
    """
    Create V28 SPLADE model (context-gated or standard).

    Args:
        config: V28 configuration

    Returns:
        SPLADEDocContextGated or SPLADEDocXLMR instance
    """
    if config.model.use_context_gate:
        from src.model.splade_xlmr import SPLADEDocContextGated

        model = SPLADEDocContextGated(
            model_name=config.model.name,
            dropout=config.model.dropout,
            use_mlm_head=config.model.use_expansion,
            gate_hidden=config.model.context_gate_hidden,
            gate_heads=config.model.context_attention_heads,
        )
        logger.info("Created SPLADEDocContextGated (V28b full)")
    else:
        from src.model.splade_xlmr import SPLADEDocXLMR

        model = SPLADEDocXLMR(
            model_name=config.model.name,
            dropout=config.model.dropout,
            use_mlm_head=config.model.use_expansion,
        )
        logger.info("Created SPLADEDocXLMR (V28a, no context gate)")

    return model


def setup_training(
    config: V28Config,
    resume_from: Optional[str] = None,
    korean_tokens_path: Optional[str] = None,
    recompute_korean_tokens: bool = False,
) -> SPLADETrainer:
    """
    Setup training components for V28.

    Args:
        config: V28 configuration
        resume_from: Optional checkpoint path to resume from
        korean_tokens_path: Path to cached Korean token IDs
        recompute_korean_tokens: Force recomputation of Korean tokens

    Returns:
        Configured SPLADETrainer
    """
    from src.model.losses import SPLADELossV28

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Create tokenizer
    logger.info(f"Loading tokenizer: {config.model.name}")
    tokenizer = create_tokenizer(config.model.name)

    # Create V28 model
    logger.info(f"Creating V28 model: {config.model.name}")
    model = create_v28_model(config)

    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Vocabulary size: {model.vocab_size:,}")

    # === V28: Get Korean token IDs for language filtering ===
    korean_token_ids = None
    if config.loss.enable_language_filtering:
        if korean_tokens_path is None:
            korean_tokens_path = f"{config.training.output_dir}/korean_token_ids.json"

        korean_token_ids = load_or_compute_korean_tokens(
            cache_path=korean_tokens_path,
            tokenizer=tokenizer,
            recompute=recompute_korean_tokens,
        )
        logger.info(f"Korean token IDs: {len(korean_token_ids):,}")

    # Get special token IDs
    special_token_ids = get_special_token_ids_only(tokenizer)
    logger.info(f"Special token IDs: {len(special_token_ids)}")

    # Compute or load IDF weights
    idf_cache_path = config.get_idf_cache_path()
    logger.info(f"IDF cache path: {idf_cache_path}")

    idf_weights = load_or_compute_idf(
        cache_path=idf_cache_path,
        corpus_files=config.data.train_files,
        tokenizer=tokenizer,
        recompute=config.loss.recompute_idf,
        smoothing=config.loss.idf_smoothing,
    )
    logger.info(f"Loaded IDF weights: shape={idf_weights.shape}")

    # Create stopword mask
    stopword_mask = None
    if config.loss.use_stopword_mask:
        logger.info("Creating V26 extended Korean stopword mask...")
        stopword_mask = create_stopword_mask_v26(tokenizer)
        masked_count = (stopword_mask == 0).sum().item()
        logger.info(f"Stopword mask: {masked_count} tokens masked")

    # Create V28 loss function
    logger.info("Creating SPLADELossV28 with language filtering...")
    logger.info(f"  enable_language_filtering: {config.loss.enable_language_filtering}")
    logger.info(f"  lambda_language: {config.loss.lambda_language}")
    logger.info(f"  non_korean_penalty: {config.loss.non_korean_penalty}")
    logger.info(f"  use_context_gate: {config.loss.use_context_gate}")

    loss_fn = SPLADELossV28(
        idf_weights=idf_weights,
        special_token_ids=special_token_ids,
        korean_token_ids=korean_token_ids,
        lambda_language=config.loss.lambda_language,
        non_korean_penalty=config.loss.non_korean_penalty,
        korean_penalty=config.loss.korean_token_penalty,
        enable_language_filtering=config.loss.enable_language_filtering,
        language_warmup_steps=config.loss.language_warmup_steps,
        language_penalty_max=config.loss.language_penalty_max,
        collapse_flops_threshold=config.loss.collapse_flops_threshold,
        collapse_check_window=config.loss.collapse_check_window,
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

    # Add collapse detection hook
    if config.loss.enable_language_filtering:
        hooks.append(CollapseDetectionHook(
            flops_threshold=config.loss.collapse_flops_threshold,
            check_window=config.loss.collapse_check_window,
            check_every_n_steps=config.training.log_every_n_steps,
        ))
        logger.info("Collapse detection enabled")
        logger.info(
            f"  threshold={config.loss.collapse_flops_threshold}, "
            f"window={config.loss.collapse_check_window}"
        )

    # Add curriculum hook if enabled
    if config.enable_curriculum:
        hooks.append(CurriculumHook(config))
        logger.info("Curriculum learning enabled")
        for i, phase in enumerate(config.curriculum_phases, 1):
            logger.info(f"  Phase {i}: epochs {phase.start_epoch}-{phase.end_epoch}")
            logger.info(f"    Temperature: {phase.temperature}")
            logger.info(f"    Lambda InfoNCE: {phase.lambda_infonce}")

    # Log KD configuration
    if config.knowledge_distillation.enabled:
        logger.info("Knowledge distillation enabled")
        logger.info(f"  Teacher: {config.knowledge_distillation.teacher_model}")
        logger.info(f"  Warmup epochs: {config.knowledge_distillation.warmup_epochs}")

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
    """Main entry point for V28 training."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    output_dir = args.output_dir or "outputs/train_v28"
    setup_logging(output_dir=output_dir, level=log_level)

    logger.info("=" * 70)
    logger.info("SPLADE V28 XLM-RoBERTa Training")
    logger.info("Language Filtering + Context-Gated Sparse Expansion")
    logger.info("=" * 70)
    logger.info("V28 Features:")
    logger.info("  V28a: Korean language filtering (suppress non-Korean tokens)")
    logger.info("  V28b: Context-gated expansion (context-dependent activation)")
    logger.info("=" * 70)

    try:
        # Handle vocab analysis mode
        if args.analyze_vocab:
            from src.train.data.collator import create_tokenizer
            tokenizer = create_tokenizer(args.model_name or "xlm-roberta-base")
            analyze_token_language_distribution(tokenizer)
            return 0

        # Build config
        config = build_config(args)
        config.validate()

        # Log configuration
        logger.info("Configuration:")
        logger.info(f"  Model: {config.model.name}")
        logger.info(f"  Model class: {config.model.model_class}")
        logger.info(f"  Use context gate: {config.model.use_context_gate}")
        logger.info(f"  Vocabulary: 250K (XLM-RoBERTa)")
        logger.info(f"  Batch size: {config.data.batch_size}")
        logger.info(f"  Max length: {config.data.max_length}")
        logger.info(f"  Learning rate: {config.training.learning_rate}")
        logger.info(f"  Epochs: {config.training.num_epochs}")
        logger.info(f"  Mixed precision: {config.training.mixed_precision}")
        logger.info(f"  Output dir: {config.training.output_dir}")
        logger.info("V28 Configuration:")
        logger.info(f"  enable_language_filtering: {config.loss.enable_language_filtering}")
        logger.info(f"  lambda_language: {config.loss.lambda_language}")
        logger.info(f"  non_korean_penalty: {config.loss.non_korean_penalty}")
        logger.info(f"  use_context_gate: {config.loss.use_context_gate}")
        logger.info(f"  context_gate_hidden: {config.loss.context_gate_hidden}")
        logger.info(f"  context_attention_heads: {config.loss.context_attention_heads}")

        # Determine resume checkpoint
        resume_from = None
        if args.checkpoint:
            resume_from = args.checkpoint
        elif args.resume:
            ckpt_manager = CheckpointManager(output_dir=config.training.output_dir)
            if ckpt_manager.has_checkpoint():
                resume_from = str(ckpt_manager.get_latest_checkpoint_path())
                logger.info(f"Resuming from: {resume_from}")
            else:
                logger.warning("No checkpoint found to resume from")

        # Setup and run training
        trainer = setup_training(
            config,
            resume_from,
            korean_tokens_path=args.korean_tokens_path,
            recompute_korean_tokens=args.recompute_korean_tokens,
        )
        results = trainer.train()

        # Log results
        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info(f"  Epochs completed: {results['epochs_completed']}")
        logger.info(f"  Final train loss: {results['final_train_loss']:.4f}")
        if results["best_val_loss"] < float("inf"):
            logger.info(f"  Best val loss: {results['best_val_loss']:.4f}")

        # Log V28 metrics
        if hasattr(trainer.loss_fn, "get_average_semantic_ratio"):
            semantic_ratio = trainer.loss_fn.get_average_semantic_ratio()
            logger.info(f"  Average semantic ratio: {semantic_ratio:.2f}")

        if hasattr(trainer.loss_fn, "get_average_korean_ratio"):
            korean_ratio = trainer.loss_fn.get_average_korean_ratio()
            logger.info(f"  Average Korean ratio: {korean_ratio:.2f}")
            if korean_ratio > 1.0:
                logger.info("  SUCCESS: Korean tokens dominating (ratio > 1.0)")
            else:
                logger.warning("  WARNING: Non-Korean tokens still present (ratio < 1.0)")

        # Create completion flag for automation
        flag_path = Path(config.training.output_dir) / "training_complete.flag"
        flag_path.write_text(f"completed_epochs={results['epochs_completed']}\n")
        logger.info(f"Created completion flag: {flag_path}")

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
