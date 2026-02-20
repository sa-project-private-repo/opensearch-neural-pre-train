"""
Collapse detection hook for SPLADE training.

Auto-detects training collapse when FLOPS drops to near-zero
and halves the language penalty to recover.
"""

import logging
from typing import Dict, TYPE_CHECKING

from src.train.core.hooks import TrainingHook

if TYPE_CHECKING:
    from src.train.core.trainer import SPLADETrainer

logger = logging.getLogger(__name__)


class CollapseDetectionHook(TrainingHook):
    """
    Detects training collapse and auto-adjusts penalties.

    Monitors FLOPS metric and triggers recovery when the model
    stops producing activations (collapse). Recovery halves the
    language penalty weight.
    """

    def __init__(
        self,
        flops_threshold: float = 0.01,
        check_window: int = 3,
        check_every_n_steps: int = 50,
        max_halvings: int = 5,
    ):
        """
        Initialize collapse detector.

        Args:
            flops_threshold: FLOPS below this = potential collapse
            check_window: Consecutive low-FLOPS before triggering
            check_every_n_steps: How often to check (in steps)
            max_halvings: Max times to halve before giving up
        """
        self.flops_threshold = flops_threshold
        self.check_window = check_window
        self.check_every_n_steps = check_every_n_steps
        self.max_halvings = max_halvings

        self._low_flops_count = 0
        self._halvings = 0
        self._last_flops = 1.0

    def on_step_end(
        self,
        trainer: "SPLADETrainer",
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Check for collapse at each step."""
        if step % self.check_every_n_steps != 0:
            return

        flops = metrics.get("flops", 1.0)
        self._last_flops = flops

        if flops < self.flops_threshold:
            self._low_flops_count += 1
            logger.warning(
                f"Step {step}: Low FLOPS detected "
                f"({flops:.6f} < {self.flops_threshold}), "
                f"count={self._low_flops_count}/{self.check_window}"
            )
        else:
            self._low_flops_count = 0

        if self._low_flops_count >= self.check_window:
            self._trigger_recovery(trainer, step)

    def _trigger_recovery(
        self,
        trainer: "SPLADETrainer",
        step: int,
    ) -> None:
        """Halve language penalty to recover from collapse."""
        if self._halvings >= self.max_halvings:
            logger.error(
                f"Step {step}: Max collapse halvings reached "
                f"({self.max_halvings}). Training may be stuck."
            )
            return

        self._halvings += 1
        self._low_flops_count = 0

        loss_fn = trainer.loss_fn
        if hasattr(loss_fn, "language_penalty_max"):
            old_val = loss_fn.language_penalty_max
            loss_fn.language_penalty_max *= 0.5
            logger.warning(
                f"COLLAPSE RECOVERY #{self._halvings} at step {step}: "
                f"language_penalty_max {old_val:.6f} -> "
                f"{loss_fn.language_penalty_max:.6f}"
            )
        if hasattr(loss_fn, "lambda_language"):
            old_val = loss_fn.lambda_language
            loss_fn.lambda_language *= 0.5
            logger.warning(
                f"  lambda_language {old_val:.6f} -> "
                f"{loss_fn.lambda_language:.6f}"
            )

    def on_epoch_end(
        self,
        trainer: "SPLADETrainer",
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log collapse detection status at epoch end."""
        logger.info(
            f"CollapseDetector: halvings={self._halvings}, "
            f"last_flops={self._last_flops:.6f}"
        )
