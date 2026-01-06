"""
Logging utilities for SPLADE training.

Provides:
- Console logging with colored output
- File logging
- TensorBoard integration
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.RED,
    }

    def __init__(self, fmt: str, use_colors: bool = True):
        """
        Initialize formatter.

        Args:
            fmt: Log format string
            use_colors: Whether to use ANSI colors
        """
        super().__init__(fmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors."""
        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"
            record.msg = f"{color}{record.msg}{Colors.RESET}"
        return super().format(record)


def setup_logging(
    output_dir: Optional[str] = None,
    level: int = logging.INFO,
    log_file: str = "training.log",
    use_colors: bool = True,
) -> logging.Logger:
    """
    Setup logging for training.

    Args:
        output_dir: Directory for log files. If None, only console logging.
        level: Logging level
        log_file: Name of log file
        use_colors: Whether to use colored console output

    Returns:
        Configured root logger
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        use_colors=use_colors,
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            output_path / log_file,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TensorBoardLogger:
    """
    TensorBoard logging wrapper.

    Provides convenient methods for logging training metrics.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        flush_secs: int = 30,
    ):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Optional experiment name (creates subdirectory)
            flush_secs: How often to flush to disk
        """
        if not HAS_TENSORBOARD:
            raise ImportError(
                "TensorBoard not available. Install with: pip install tensorboard"
            )

        if experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = str(Path(log_dir) / f"{experiment_name}_{timestamp}")

        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self.log_dir = log_dir
        self._step = 0

        logging.getLogger(__name__).info(f"TensorBoard logging to: {log_dir}")

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a scalar value.

        Args:
            tag: Metric name
            value: Metric value
            step: Global step (uses internal counter if not provided)
        """
        step = step if step is not None else self._step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log multiple scalars under a common tag.

        Args:
            main_tag: Main tag name
            tag_scalar_dict: Dict of sub-tag to value
            step: Global step
        """
        step = step if step is not None else self._step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(
        self,
        tag: str,
        values: Any,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a histogram.

        Args:
            tag: Histogram name
            values: Values to histogram
            step: Global step
        """
        step = step if step is not None else self._step
        self.writer.add_histogram(tag, values, step)

    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None,
    ) -> None:
        """
        Log text.

        Args:
            tag: Text tag
            text: Text content
            step: Global step
        """
        step = step if step is not None else self._step
        self.writer.add_text(tag, text, step)

    def log_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, float],
    ) -> None:
        """
        Log hyperparameters and metrics.

        Args:
            hparam_dict: Hyperparameter dictionary
            metric_dict: Metric dictionary
        """
        self.writer.add_hparams(hparam_dict, metric_dict)

    def log_training_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        loss_components: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log a training step with all relevant metrics.

        Args:
            step: Global step
            loss: Total loss
            learning_rate: Current learning rate
            loss_components: Optional dict of individual loss components
        """
        self._step = step

        self.log_scalar("train/loss", loss, step)
        self.log_scalar("train/learning_rate", learning_rate, step)

        if loss_components:
            for name, value in loss_components.items():
                self.log_scalar(f"train/loss_{name}", value, step)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log epoch-level metrics.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Optional validation loss
            metrics: Optional additional metrics
        """
        self.log_scalar("epoch/train_loss", train_loss, epoch)

        if val_loss is not None:
            self.log_scalar("epoch/val_loss", val_loss, epoch)

        if metrics:
            for name, value in metrics.items():
                self.log_scalar(f"epoch/{name}", value, epoch)

    def flush(self) -> None:
        """Flush pending events to disk."""
        self.writer.flush()

    def close(self) -> None:
        """Close the writer."""
        self.writer.close()

    def __enter__(self) -> "TensorBoardLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
