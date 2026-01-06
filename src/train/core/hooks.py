"""
Training hooks for SPLADE training.

Hooks allow custom actions at various points during training:
- on_train_begin / on_train_end
- on_epoch_begin / on_epoch_end
- on_step_begin / on_step_end
- on_batch_begin / on_batch_end
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.train.core.trainer import SPLADETrainer
    from src.train.config.v22 import V22Config


logger = logging.getLogger(__name__)


class TrainingHook(ABC):
    """
    Base class for training hooks.

    Override the methods you need to customize training behavior.
    """

    def on_train_begin(self, trainer: "SPLADETrainer") -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: "SPLADETrainer") -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: "SPLADETrainer", epoch: int) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self,
        trainer: "SPLADETrainer",
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_begin(self, trainer: "SPLADETrainer", step: int) -> None:
        """Called at the start of each optimization step."""
        pass

    def on_step_end(
        self,
        trainer: "SPLADETrainer",
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Called at the end of each optimization step."""
        pass

    def on_batch_begin(
        self,
        trainer: "SPLADETrainer",
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Called before processing each batch.

        Can modify the batch before processing.

        Args:
            trainer: The trainer instance
            batch: Input batch

        Returns:
            Modified batch (or original)
        """
        return batch

    def on_batch_end(
        self,
        trainer: "SPLADETrainer",
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, Any],
    ) -> None:
        """Called after processing each batch."""
        pass


class LoggingHook(TrainingHook):
    """
    Hook for logging training progress.

    Logs to both console and TensorBoard.
    """

    def __init__(
        self,
        log_every_n_steps: int = 50,
        tensorboard_writer: Optional[Any] = None,
    ):
        """
        Initialize logging hook.

        Args:
            log_every_n_steps: Log every N steps
            tensorboard_writer: Optional TensorBoard SummaryWriter
        """
        self.log_every_n_steps = log_every_n_steps
        self.writer = tensorboard_writer
        self.step_metrics: Dict[str, float] = {}

    def on_train_begin(self, trainer: "SPLADETrainer") -> None:
        """Log training start."""
        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info("=" * 80)
        logger.info(f"  Epochs: {trainer.config.training.num_epochs}")
        logger.info(f"  Batch size: {trainer.config.data.batch_size}")
        logger.info(f"  Learning rate: {trainer.config.training.learning_rate}")
        logger.info(f"  Device: {trainer.device}")
        logger.info("=" * 80)

    def on_train_end(self, trainer: "SPLADETrainer") -> None:
        """Log training end."""
        logger.info("=" * 80)
        logger.info("Training complete!")
        logger.info("=" * 80)

        if self.writer:
            self.writer.close()

    def on_epoch_begin(self, trainer: "SPLADETrainer", epoch: int) -> None:
        """Log epoch start."""
        logger.info(f"\n{'='*40}")
        logger.info(f"Epoch {epoch}/{trainer.config.training.num_epochs}")
        logger.info(f"{'='*40}")

    def on_epoch_end(
        self,
        trainer: "SPLADETrainer",
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log epoch metrics."""
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"Epoch {epoch} complete - {metrics_str}")

        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(f"epoch/{name}", value, epoch)

    def on_step_end(
        self,
        trainer: "SPLADETrainer",
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log step metrics."""
        # Accumulate metrics
        for name, value in metrics.items():
            self.step_metrics[name] = value

        if step % self.log_every_n_steps == 0:
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in self.step_metrics.items())
            logger.info(f"Step {step} - {metrics_str}")

            if self.writer:
                for name, value in self.step_metrics.items():
                    self.writer.add_scalar(f"train/{name}", value, step)

            self.step_metrics = {}


class CheckpointHook(TrainingHook):
    """
    Hook for saving checkpoints during training.
    """

    def __init__(
        self,
        checkpoint_manager: Any,
        save_every_n_epochs: int = 5,
        save_every_n_steps: Optional[int] = None,
    ):
        """
        Initialize checkpoint hook.

        Args:
            checkpoint_manager: CheckpointManager instance
            save_every_n_epochs: Save every N epochs
            save_every_n_steps: Save every N steps (overrides epochs if set)
        """
        self.checkpoint_manager = checkpoint_manager
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_steps = save_every_n_steps
        self.last_epoch_metrics: Dict[str, float] = {}

    def on_epoch_end(
        self,
        trainer: "SPLADETrainer",
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save checkpoint at end of epoch if appropriate."""
        self.last_epoch_metrics = metrics

        if self.save_every_n_steps is not None:
            # Step-based saving takes precedence
            return

        if epoch % self.save_every_n_epochs == 0:
            self.checkpoint_manager.save(
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=epoch,
                step=trainer.global_step,
                metrics=metrics,
            )

    def on_step_end(
        self,
        trainer: "SPLADETrainer",
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save checkpoint at step if using step-based saving."""
        if self.save_every_n_steps is None:
            return

        if step % self.save_every_n_steps == 0:
            self.checkpoint_manager.save(
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=trainer.current_epoch,
                step=step,
                metrics=metrics,
            )


class CurriculumHook(TrainingHook):
    """
    Hook for curriculum learning.

    Adjusts training parameters (temperature, loss weights, learning rate)
    based on the current curriculum phase.
    """

    def __init__(self, config: "V22Config"):
        """
        Initialize curriculum hook.

        Args:
            config: V22 configuration with curriculum phases
        """
        self.config = config
        self.current_phase = None

    def on_epoch_begin(self, trainer: "SPLADETrainer", epoch: int) -> None:
        """Adjust parameters for current curriculum phase."""
        if not self.config.enable_curriculum:
            return

        phase = self.config.get_phase_for_epoch(epoch)

        if phase is None:
            return

        # Log phase change
        if phase != self.current_phase:
            self.current_phase = phase
            logger.info(f"Curriculum phase change: {phase.description}")
            logger.info(f"  Temperature: {phase.temperature}")
            logger.info(f"  Lambda InfoNCE: {phase.lambda_infonce}")
            logger.info(f"  LR multiplier: {phase.lr_multiplier}")

        # Update loss function temperature
        if hasattr(trainer.loss_fn, "update_temperature"):
            trainer.loss_fn.update_temperature(phase.temperature)

        # Update loss weights
        if hasattr(trainer.loss_fn, "update_weights"):
            trainer.loss_fn.update_weights(lambda_infonce=phase.lambda_infonce)

        # Update learning rate
        if phase.lr_multiplier != 1.0:
            for param_group in trainer.optimizer.param_groups:
                base_lr = self.config.training.learning_rate
                param_group["lr"] = base_lr * phase.lr_multiplier


class GradientMonitorHook(TrainingHook):
    """
    Hook for monitoring gradient statistics.

    Useful for debugging training issues.
    """

    def __init__(
        self,
        log_every_n_steps: int = 100,
        tensorboard_writer: Optional[Any] = None,
    ):
        """
        Initialize gradient monitor.

        Args:
            log_every_n_steps: Log every N steps
            tensorboard_writer: Optional TensorBoard writer
        """
        self.log_every_n_steps = log_every_n_steps
        self.writer = tensorboard_writer

    def on_step_end(
        self,
        trainer: "SPLADETrainer",
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log gradient statistics."""
        if step % self.log_every_n_steps != 0:
            return

        total_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        logger.debug(f"Step {step} - Gradient norm: {total_norm:.4f}")

        if self.writer:
            self.writer.add_scalar("gradients/total_norm", total_norm, step)


class EarlyStoppingHook(TrainingHook):
    """
    Hook for early stopping based on validation metrics.
    """

    def __init__(
        self,
        patience: int = 5,
        metric: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            metric: Metric to monitor
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta

        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(
        self,
        trainer: "SPLADETrainer",
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Check if training should stop."""
        if self.metric not in metrics:
            return

        current = metrics[self.metric]

        if self.best_value is None:
            self.best_value = current
            return

        if self.mode == "min":
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered after {epoch} epochs")
            logger.info(f"  Best {self.metric}: {self.best_value:.4f}")
