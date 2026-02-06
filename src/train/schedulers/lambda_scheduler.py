"""
Lambda schedulers for SPLADE v2 style FLOPS regularization warmup.

Based on SPLADE v2 paper: λ increases from 0 to target over warmup steps
using quadratic scheduling for smooth transition.
"""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseLambdaScheduler(ABC):
    """Base class for lambda schedulers."""

    def __init__(
        self,
        initial_lambda_q: float = 0.0,
        initial_lambda_d: float = 0.0,
        target_lambda_q: float = 1e-4,
        target_lambda_d: float = 1e-3,
        warmup_steps: int = 50000,
    ):
        """
        Initialize lambda scheduler.

        Args:
            initial_lambda_q: Initial query FLOPS weight
            initial_lambda_d: Initial document FLOPS weight
            target_lambda_q: Target query FLOPS weight
            target_lambda_d: Target document FLOPS weight
            warmup_steps: Number of steps to reach target
        """
        self.initial_lambda_q = initial_lambda_q
        self.initial_lambda_d = initial_lambda_d
        self.target_lambda_q = target_lambda_q
        self.target_lambda_d = target_lambda_d
        self.warmup_steps = warmup_steps
        self._current_step = 0

    @abstractmethod
    def _compute_progress(self, step: int) -> float:
        """Compute progress value [0, 1] based on step."""
        pass

    def get_lambda(self, step: int) -> Tuple[float, float]:
        """
        Get lambda values for current step.

        Args:
            step: Current training step

        Returns:
            Tuple of (lambda_q, lambda_d)
        """
        self._current_step = step

        if step >= self.warmup_steps:
            return self.target_lambda_q, self.target_lambda_d

        progress = self._compute_progress(step)

        lambda_q = self.initial_lambda_q + (self.target_lambda_q - self.initial_lambda_q) * progress
        lambda_d = self.initial_lambda_d + (self.target_lambda_d - self.initial_lambda_d) * progress

        return lambda_q, lambda_d

    def step(self) -> Tuple[float, float]:
        """Increment step and return current lambda values."""
        self._current_step += 1
        return self.get_lambda(self._current_step)

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            "initial_lambda_q": self.initial_lambda_q,
            "initial_lambda_d": self.initial_lambda_d,
            "target_lambda_q": self.target_lambda_q,
            "target_lambda_d": self.target_lambda_d,
            "warmup_steps": self.warmup_steps,
            "current_step": self._current_step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint."""
        self.initial_lambda_q = state_dict["initial_lambda_q"]
        self.initial_lambda_d = state_dict["initial_lambda_d"]
        self.target_lambda_q = state_dict["target_lambda_q"]
        self.target_lambda_d = state_dict["target_lambda_d"]
        self.warmup_steps = state_dict["warmup_steps"]
        self._current_step = state_dict["current_step"]


class QuadraticLambdaScheduler(BaseLambdaScheduler):
    """
    Quadratic lambda warmup scheduler (SPLADE v2 style).

    Lambda increases quadratically from initial to target:
    λ(t) = λ_init + (λ_target - λ_init) * (t / warmup_steps)²

    This provides slow initial increase followed by faster growth.
    """

    def _compute_progress(self, step: int) -> float:
        """Quadratic progress: slow start, fast finish."""
        linear_progress = step / self.warmup_steps
        return linear_progress ** 2


class LinearLambdaScheduler(BaseLambdaScheduler):
    """
    Linear lambda warmup scheduler.

    Lambda increases linearly from initial to target:
    λ(t) = λ_init + (λ_target - λ_init) * (t / warmup_steps)
    """

    def _compute_progress(self, step: int) -> float:
        """Linear progress."""
        return step / self.warmup_steps


class ExponentialLambdaScheduler(BaseLambdaScheduler):
    """
    Exponential lambda warmup scheduler.

    Lambda increases exponentially (slow then fast):
    λ(t) = λ_init + (λ_target - λ_init) * (exp(t/warmup_steps) - 1) / (e - 1)
    """

    def __init__(self, *args, **kwargs):
        """Initialize with exponential base."""
        super().__init__(*args, **kwargs)
        import math
        self._e_minus_1 = math.e - 1

    def _compute_progress(self, step: int) -> float:
        """Exponential progress."""
        import math
        linear_progress = step / self.warmup_steps
        return (math.exp(linear_progress) - 1) / self._e_minus_1
