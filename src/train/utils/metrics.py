"""
Training metrics utilities.

Provides:
- Metrics computation and tracking
- Moving averages
- Summary statistics
"""

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


@dataclass
class TrainingMetrics:
    """Container for training metrics at a single point."""

    step: int
    epoch: int
    loss: float
    learning_rate: float
    loss_components: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "loss_components": self.loss_components,
            "timestamp": self.timestamp,
        }


class MovingAverage:
    """
    Exponential moving average tracker.

    Useful for smoothing noisy training metrics.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize moving average.

        Args:
            alpha: Smoothing factor (0 < alpha <= 1).
                   Higher = more weight to recent values.
        """
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, value: float) -> float:
        """
        Update with new value and return smoothed result.

        Args:
            value: New value to incorporate

        Returns:
            Smoothed value
        """
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * value + (1 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> Optional[float]:
        """Get current smoothed value."""
        return self._value

    def reset(self) -> None:
        """Reset the moving average."""
        self._value = None


class WindowedAverage:
    """
    Windowed average tracker.

    Maintains average over a fixed window of recent values.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize windowed average.

        Args:
            window_size: Number of recent values to average
        """
        self.window_size = window_size
        self._values: Deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        """
        Update with new value and return windowed average.

        Args:
            value: New value to incorporate

        Returns:
            Windowed average
        """
        self._values.append(value)
        return sum(self._values) / len(self._values)

    @property
    def value(self) -> Optional[float]:
        """Get current average."""
        if not self._values:
            return None
        return sum(self._values) / len(self._values)

    def reset(self) -> None:
        """Reset the tracker."""
        self._values.clear()


class MetricsTracker:
    """
    Comprehensive metrics tracker for training.

    Features:
    - Tracks multiple metrics
    - Maintains moving and windowed averages
    - Logs to JSONL file
    - Computes summary statistics
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        log_file: str = "metrics.jsonl",
        window_size: int = 100,
        ema_alpha: float = 0.1,
    ):
        """
        Initialize metrics tracker.

        Args:
            output_dir: Directory for metrics log file
            log_file: Name of JSONL log file
            window_size: Window size for windowed averages
            ema_alpha: Alpha for exponential moving average
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.log_file = log_file
        self.window_size = window_size
        self.ema_alpha = ema_alpha

        # Per-metric trackers
        self._ema: Dict[str, MovingAverage] = {}
        self._windowed: Dict[str, WindowedAverage] = {}

        # History
        self._history: List[TrainingMetrics] = []
        self._best_metrics: Dict[str, float] = {}

        # Setup log file
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = self.output_dir / log_file

    def _get_ema(self, metric: str) -> MovingAverage:
        """Get or create EMA tracker for metric."""
        if metric not in self._ema:
            self._ema[metric] = MovingAverage(self.ema_alpha)
        return self._ema[metric]

    def _get_windowed(self, metric: str) -> WindowedAverage:
        """Get or create windowed average tracker for metric."""
        if metric not in self._windowed:
            self._windowed[metric] = WindowedAverage(self.window_size)
        return self._windowed[metric]

    def update(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        loss_components: Optional[Dict[str, float]] = None,
    ) -> TrainingMetrics:
        """
        Update metrics with new training step.

        Args:
            step: Global step
            epoch: Current epoch
            loss: Total loss
            learning_rate: Current learning rate
            loss_components: Optional dict of individual losses

        Returns:
            TrainingMetrics object
        """
        loss_components = loss_components or {}

        # Create metrics object
        metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            loss_components=loss_components,
        )

        # Update trackers
        self._get_ema("loss").update(loss)
        self._get_windowed("loss").update(loss)

        for name, value in loss_components.items():
            self._get_ema(name).update(value)
            self._get_windowed(name).update(value)

        # Track best
        if "loss" not in self._best_metrics or loss < self._best_metrics["loss"]:
            self._best_metrics["loss"] = loss

        # Add to history
        self._history.append(metrics)

        # Log to file
        self._log_metrics(metrics)

        return metrics

    def _log_metrics(self, metrics: TrainingMetrics) -> None:
        """Append metrics to log file."""
        if not self.output_dir:
            return

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")

    def get_smoothed(self, metric: str = "loss") -> Optional[float]:
        """Get EMA-smoothed value for metric."""
        tracker = self._ema.get(metric)
        return tracker.value if tracker else None

    def get_windowed_average(self, metric: str = "loss") -> Optional[float]:
        """Get windowed average for metric."""
        tracker = self._windowed.get(metric)
        return tracker.value if tracker else None

    def get_best(self, metric: str = "loss") -> Optional[float]:
        """Get best value seen for metric."""
        return self._best_metrics.get(metric)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all tracked metrics.

        Returns:
            Dict with summary statistics
        """
        if not self._history:
            return {}

        summary: Dict[str, Any] = {
            "total_steps": len(self._history),
            "epochs": self._history[-1].epoch if self._history else 0,
        }

        # Loss statistics
        losses = [m.loss for m in self._history]
        summary["loss"] = {
            "final": losses[-1] if losses else None,
            "best": min(losses) if losses else None,
            "mean": sum(losses) / len(losses) if losses else None,
        }

        # Per-component statistics
        if self._history and self._history[-1].loss_components:
            for component in self._history[-1].loss_components.keys():
                values = [
                    m.loss_components.get(component, 0)
                    for m in self._history
                    if component in m.loss_components
                ]
                if values:
                    summary[f"loss_{component}"] = {
                        "final": values[-1],
                        "mean": sum(values) / len(values),
                    }

        return summary

    def save_summary(self, path: Optional[str] = None) -> None:
        """
        Save summary statistics to JSON file.

        Args:
            path: Output path. If None, uses output_dir/summary.json
        """
        if path is None and self.output_dir:
            path = str(self.output_dir / "summary.json")

        if path is None:
            return

        summary = self.get_summary()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def reset(self) -> None:
        """Reset all trackers and history."""
        self._ema.clear()
        self._windowed.clear()
        self._history.clear()
        self._best_metrics.clear()


def compute_throughput(
    num_samples: int,
    elapsed_seconds: float,
) -> Dict[str, float]:
    """
    Compute training throughput metrics.

    Args:
        num_samples: Number of samples processed
        elapsed_seconds: Elapsed time in seconds

    Returns:
        Dict with throughput metrics
    """
    samples_per_second = num_samples / elapsed_seconds if elapsed_seconds > 0 else 0
    seconds_per_sample = elapsed_seconds / num_samples if num_samples > 0 else 0

    return {
        "samples_per_second": samples_per_second,
        "seconds_per_sample": seconds_per_sample,
        "samples_per_hour": samples_per_second * 3600,
    }
