"""
SPLADE Trainer for neural sparse retrieval.

Production-ready trainer with:
- Mixed precision training (BF16/FP16)
- Curriculum learning support
- TensorBoard logging
- Checkpoint management
- Graceful interruption handling
"""

import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

from src.train.config.base import BaseConfig
from src.train.config.v22 import V22Config
from src.train.core.checkpoint import CheckpointManager
from src.train.core.hooks import TrainingHook


logger = logging.getLogger(__name__)


class SPLADETrainer:
    """
    Production trainer for SPLADE neural sparse models.

    Features:
    - Mixed precision training (BF16/FP16)
    - Curriculum learning with phase-based hyperparameters
    - Checkpoint management with automatic cleanup
    - TensorBoard logging
    - Graceful interruption handling
    - Hook system for extensibility
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        loss_fn: nn.Module,
        config: BaseConfig,
        checkpoint_manager: Optional[CheckpointManager] = None,
        hooks: Optional[List[TrainingHook]] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: SPLADE model to train
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            loss_fn: Loss function (e.g., SPLADELossV22)
            config: Training configuration
            checkpoint_manager: Optional checkpoint manager
            hooks: Optional list of training hooks
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.config = config

        # Device setup
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

        # Mixed precision setup
        self.mixed_precision = config.training.mixed_precision
        self.scaler: Optional[GradScaler] = None
        self._setup_mixed_precision()

        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Checkpoint manager
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            output_dir=config.training.output_dir,
            keep_last_n=config.training.keep_last_n_checkpoints,
        )

        # Training hooks
        self.hooks = hooks or []

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        # Interruption handling
        self._interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _setup_mixed_precision(self) -> None:
        """Setup mixed precision training."""
        if self.mixed_precision == "fp16":
            self.scaler = GradScaler()
            logger.info("Using FP16 mixed precision with GradScaler")
        elif self.mixed_precision == "bf16":
            # BF16 doesn't need GradScaler
            logger.info("Using BF16 mixed precision")
        else:
            logger.info("Using FP32 (no mixed precision)")

    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.training.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        return optimizer

    def _create_scheduler(self) -> LRScheduler:
        """Create learning rate scheduler."""
        total_steps = (
            len(self.train_dataloader)
            * self.config.training.num_epochs
            // self.config.training.gradient_accumulation_steps
        )

        warmup_steps = int(total_steps * self.config.training.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(f"Scheduler: cosine with warmup")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")

        return scheduler

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle keyboard interrupt gracefully."""
        if self._interrupted:
            logger.warning("Force quit. Progress may not be saved.")
            sys.exit(1)

        logger.warning("\nInterrupt received. Finishing current batch and saving...")
        self._interrupted = True

    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.

        Returns:
            Dict with training results and final metrics
        """
        # Notify hooks
        for hook in self.hooks:
            hook.on_train_begin(self)

        results = {
            "epochs_completed": 0,
            "final_train_loss": float("inf"),
            "best_val_loss": float("inf"),
        }

        try:
            # Start from resumed epoch + 1 (or 1 if fresh start)
            start_epoch = getattr(self, "_resume_epoch", 0) + 1
            for epoch in range(start_epoch, self.config.training.num_epochs + 1):
                if self._interrupted:
                    break

                self.current_epoch = epoch

                # Notify hooks
                for hook in self.hooks:
                    hook.on_epoch_begin(self, epoch)

                # Train epoch
                train_metrics = self._train_epoch(epoch)

                # Validate
                val_metrics = {}
                if self.val_dataloader is not None:
                    val_metrics = self._validate()

                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}

                # Notify hooks
                for hook in self.hooks:
                    hook.on_epoch_end(self, epoch, epoch_metrics)

                # Update results
                results["epochs_completed"] = epoch
                results["final_train_loss"] = train_metrics.get("train_loss", float("inf"))
                if "val_loss" in val_metrics:
                    if val_metrics["val_loss"] < results["best_val_loss"]:
                        results["best_val_loss"] = val_metrics["val_loss"]

                # Check early stopping
                for hook in self.hooks:
                    if hasattr(hook, "should_stop") and hook.should_stop:
                        logger.info("Early stopping triggered")
                        break

        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        finally:
            # Always try to save on exit
            if self.global_step > 0:
                self._save_final_checkpoint()

            # Notify hooks
            for hook in self.hooks:
                hook.on_train_end(self)

        return results

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dict of training metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        loss_components: Dict[str, float] = {}

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not sys.stdout.isatty(),
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            if self._interrupted:
                break

            # Move batch to device (skip non-tensor metadata)
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Notify hooks
            for hook in self.hooks:
                batch = hook.on_batch_begin(self, batch)

            # Forward pass
            loss, step_loss_dict = self._train_step(batch)

            # Backward pass
            loss = loss / self.config.training.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate loss components
            for k, v in step_loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v

            # Optimizer step
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                self._optimizer_step()

                # Notify hooks
                step_metrics = {
                    "loss": loss.item() * self.config.training.gradient_accumulation_steps,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                step_metrics.update({
                    k: v / self.config.training.gradient_accumulation_steps
                    for k, v in loss_components.items()
                })

                for hook in self.hooks:
                    hook.on_step_end(self, self.global_step, step_metrics)

                loss_components = {}

            total_loss += loss.item() * self.config.training.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        return {"train_loss": total_loss / max(num_batches, 1)}

    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute a single training step.

        Args:
            batch: Tokenized batch

        Returns:
            loss: Scalar loss tensor
            loss_dict: Dictionary of loss components
        """
        # Clone all tensors to prevent DDP inplace modification errors
        batch = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        # Determine autocast dtype
        autocast_dtype = None
        if self.mixed_precision == "bf16":
            autocast_dtype = torch.bfloat16
        elif self.mixed_precision == "fp16":
            autocast_dtype = torch.float16

        with autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=(autocast_dtype is not None),
        ):
            # Encode query/anchor
            anchor_repr, _ = self.model(
                batch["query_input_ids"],
                batch["query_attention_mask"],
            )

            # Encode positive
            positive_repr, _ = self.model(
                batch["positive_input_ids"],
                batch["positive_attention_mask"],
            )

            # Encode negative (if available)
            negative_repr = None
            if "negative_input_ids" in batch:
                negative_repr, _ = self.model(
                    batch["negative_input_ids"],
                    batch["negative_attention_mask"],
                )
            elif "hard_negative_input_ids" in batch:
                negative_repr, _ = self.model(
                    batch["hard_negative_input_ids"],
                    batch["hard_negative_attention_mask"],
                )

            # For in-batch negatives, use positives from other samples
            if negative_repr is None:
                # Roll positives to create negatives
                negative_repr = torch.roll(positive_repr, shifts=1, dims=0)

            # Compute loss
            loss, loss_dict = self.loss_fn(
                anchor_repr=anchor_repr,
                positive_repr=positive_repr,
                negative_repr=negative_repr,
                anchor_input_ids=batch["query_input_ids"],
                anchor_attention_mask=batch["query_attention_mask"],
                positive_input_ids=batch["positive_input_ids"],
                positive_attention_mask=batch["positive_attention_mask"],
                anchor_texts=batch.get("query_texts"),
                positive_texts=batch.get("positive_texts"),
            )

        return loss, loss_dict

    def _optimizer_step(self) -> None:
        """Execute optimizer step with gradient clipping."""
        if self.scaler is not None:
            # FP16: unscale before clipping
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.gradient_clip,
        )

        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

        # Sync global step to loss_fn for warmup scheduling
        if hasattr(self.loss_fn, "set_global_step"):
            self.loss_fn.set_global_step(self.global_step)

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dict of validation metrics
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Validation", disable=not sys.stdout.isatty()):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            loss, _ = self._train_step(batch)
            total_loss += loss.item()
            num_batches += 1

        self.model.train()

        val_loss = total_loss / max(num_batches, 1)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        return {"val_loss": val_loss}

    def _save_final_checkpoint(self) -> None:
        """Save final checkpoint on training end."""
        logger.info("Saving final checkpoint...")
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.global_step,
            metrics={"best_val_loss": self.best_val_loss},
        )

    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Optional path to specific checkpoint.
                           If None, resumes from latest.
        """
        if checkpoint_path:
            info = self.checkpoint_manager.load(
                Path(checkpoint_path),
                self.model,
                self.optimizer,
                self.scheduler,
                str(self.device),
            )
        else:
            info = self.checkpoint_manager.load_latest(
                self.model,
                self.optimizer,
                self.scheduler,
                str(self.device),
            )

        self.current_epoch = info["epoch"]
        self.global_step = info["step"]
        self.best_val_loss = info.get("metrics", {}).get("best_val_loss", float("inf"))

        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")


def create_trainer_from_config(
    config: V22Config,
    model: Optional[nn.Module] = None,
    loss_fn: Optional[nn.Module] = None,
) -> SPLADETrainer:
    """
    Create a trainer from configuration.

    Args:
        config: V22 configuration
        model: Optional pre-created model
        loss_fn: Optional pre-created loss function

    Returns:
        Configured SPLADETrainer instance
    """
    from src.model.splade_model import create_splade_model
    from src.model.losses import SPLADELossV22
    from src.train.data import load_training_data, create_dataloader
    from src.train.data.collator import create_tokenizer

    # Create tokenizer
    tokenizer = create_tokenizer(config.model.name)

    # Create model if not provided
    if model is None:
        model = create_splade_model(
            model_name=config.model.name,
            dropout=config.model.dropout,
            use_expansion=config.model.use_expansion,
            expansion_mode=config.model.expansion_mode,
        )

    # Create loss function if not provided
    if loss_fn is None:
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

    # Load data
    train_dataset = load_training_data(config.data.train_files)
    val_dataset = None
    if config.data.val_files:
        val_dataset = load_training_data(config.data.val_files)

    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        tokenizer,
        batch_size=config.data.batch_size,
        max_length=config.data.max_length,
        num_workers=config.data.num_workers,
        shuffle=True,
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
        )

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir=config.training.output_dir,
        keep_last_n=config.training.keep_last_n_checkpoints,
    )

    # Create trainer
    trainer = SPLADETrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        config=config,
        checkpoint_manager=checkpoint_manager,
    )

    return trainer
