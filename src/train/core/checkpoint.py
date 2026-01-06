"""
Checkpoint management for SPLADE training.

Handles saving, loading, and cleaning up checkpoints.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""

    path: Path
    epoch: int
    step: int
    timestamp: str
    metrics: Dict[str, float]

    @property
    def filename(self) -> str:
        """Get checkpoint filename."""
        return self.path.name


class CheckpointManager:
    """
    Manager for training checkpoints.

    Handles:
    - Saving model, optimizer, and scheduler states
    - Loading from checkpoints
    - Automatic cleanup of old checkpoints
    - Tracking best model
    """

    def __init__(
        self,
        output_dir: str,
        keep_last_n: int = 3,
        save_best: bool = True,
        metric_for_best: str = "val_loss",
        metric_higher_is_better: bool = False,
    ):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            save_best: Whether to save best model separately
            metric_for_best: Metric to use for determining best model
            metric_higher_is_better: Whether higher metric is better
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.metric_for_best = metric_for_best
        self.metric_higher_is_better = metric_higher_is_better

        self.best_metric: Optional[float] = None
        self.checkpoints: List[CheckpointInfo] = []

        # Load existing checkpoint info
        self._scan_existing_checkpoints()

    def _scan_existing_checkpoints(self) -> None:
        """Scan output directory for existing checkpoints."""
        checkpoint_dirs = list(self.output_dir.glob("checkpoint_*"))

        for ckpt_dir in checkpoint_dirs:
            info_file = ckpt_dir / "checkpoint_info.json"
            if info_file.exists():
                with open(info_file, "r") as f:
                    info = json.load(f)
                    self.checkpoints.append(CheckpointInfo(
                        path=ckpt_dir,
                        epoch=info["epoch"],
                        step=info["step"],
                        timestamp=info["timestamp"],
                        metrics=info.get("metrics", {}),
                    ))

        # Sort by step
        self.checkpoints.sort(key=lambda x: x.step)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Learning rate scheduler to save
            epoch: Current epoch
            step: Current global step
            metrics: Training metrics
            config: Training configuration

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}"
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = checkpoint_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Save optimizer state
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)

        # Save scheduler state
        if scheduler is not None:
            scheduler_path = checkpoint_dir / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)

        # Save checkpoint info
        info = {
            "epoch": epoch,
            "step": step,
            "timestamp": timestamp,
            "metrics": metrics,
        }
        if config:
            info["config"] = config

        info_path = checkpoint_dir / "checkpoint_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        # Track checkpoint
        ckpt_info = CheckpointInfo(
            path=checkpoint_dir,
            epoch=epoch,
            step=step,
            timestamp=timestamp,
            metrics=metrics,
        )
        self.checkpoints.append(ckpt_info)

        logger.info(f"Saved checkpoint: {checkpoint_dir}")

        # Check if this is the best model
        if self.save_best and self.metric_for_best in metrics:
            current_metric = metrics[self.metric_for_best]
            is_best = False

            if self.best_metric is None:
                is_best = True
            elif self.metric_higher_is_better:
                is_best = current_metric > self.best_metric
            else:
                is_best = current_metric < self.best_metric

            if is_best:
                self.best_metric = current_metric
                self._save_best_model(model, metrics)

        # Cleanup old checkpoints
        self._cleanup()

        return checkpoint_dir

    def _save_best_model(
        self,
        model: torch.nn.Module,
        metrics: Dict[str, float],
    ) -> None:
        """Save the best model."""
        best_dir = self.output_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = best_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Save info
        info = {
            "metrics": metrics,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        info_path = best_dir / "best_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"New best model saved with {self.metric_for_best}: {self.best_metric:.4f}")

    def _cleanup(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        if len(self.checkpoints) <= self.keep_last_n:
            return

        # Sort by step and remove oldest
        self.checkpoints.sort(key=lambda x: x.step)
        to_remove = self.checkpoints[:-self.keep_last_n]

        for ckpt in to_remove:
            if ckpt.path.exists():
                import shutil
                shutil.rmtree(ckpt.path)
                logger.debug(f"Removed old checkpoint: {ckpt.path}")

        self.checkpoints = self.checkpoints[-self.keep_last_n:]

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Load the latest checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load tensors to

        Returns:
            Dict with epoch, step, and metrics from checkpoint

        Raises:
            FileNotFoundError: If no checkpoints exist
        """
        if not self.checkpoints:
            raise FileNotFoundError("No checkpoints found")

        return self.load(self.checkpoints[-1].path, model, optimizer, scheduler, device)

    def load(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Load a specific checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load tensors to

        Returns:
            Dict with epoch, step, and metrics from checkpoint
        """
        checkpoint_path = Path(checkpoint_path)

        # Load model
        model_path = checkpoint_path / "model.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Load optimizer
        if optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pt"
            if optimizer_path.exists():
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

        # Load scheduler
        if scheduler is not None:
            scheduler_path = checkpoint_path / "scheduler.pt"
            if scheduler_path.exists():
                scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))

        # Load info
        info_path = checkpoint_path / "checkpoint_info.json"
        with open(info_path, "r") as f:
            info = json.load(f)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"  Epoch: {info['epoch']}, Step: {info['step']}")

        return info

    def load_best(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Load the best model.

        Args:
            model: Model to load state into
            device: Device to load tensors to

        Returns:
            Dict with metrics from best model

        Raises:
            FileNotFoundError: If no best model exists
        """
        best_dir = self.output_dir / "best_model"
        if not best_dir.exists():
            raise FileNotFoundError("No best model found")

        model_path = best_dir / "model.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))

        info_path = best_dir / "best_info.json"
        with open(info_path, "r") as f:
            info = json.load(f)

        logger.info(f"Loaded best model from {best_dir}")
        return info

    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get path to latest checkpoint, or None if no checkpoints exist."""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1].path

    def has_checkpoint(self) -> bool:
        """Check if any checkpoints exist."""
        return len(self.checkpoints) > 0
