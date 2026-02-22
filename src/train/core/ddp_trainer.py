"""DDP-wrapped SPLADE Trainer for multi-GPU training."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.train.config.base import BaseConfig
from src.train.core.checkpoint import CheckpointManager
from src.train.core.hooks import TrainingHook
from src.train.core.trainer import SPLADETrainer

logger = logging.getLogger(__name__)


class DDPSPLADETrainer(SPLADETrainer):
    """
    DDP-wrapped SPLADE trainer for multi-GPU training.

    Extends SPLADETrainer with:
    - DistributedDataParallel model wrapping
    - DistributedSampler for data sharding
    - Rank-aware logging (only rank 0 logs)
    - Rank-aware checkpoint saving (only rank 0 saves)
    - Gradient synchronization across GPUs
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        loss_fn: torch.nn.Module,
        config: BaseConfig,
        checkpoint_manager: Optional[CheckpointManager] = None,
        hooks: Optional[List[TrainingHook]] = None,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize DDP trainer.

        Args:
            model: SPLADE model to train
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            loss_fn: Loss function
            config: Training configuration
            checkpoint_manager: Optional checkpoint manager
            hooks: Optional list of training hooks
            local_rank: Local GPU rank
            world_size: Total number of GPUs
        """
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main_process = local_rank == 0

        # Override device before super().__init__
        config.device = f"cuda:{local_rank}"

        # Extract train sampler before super init
        self.train_sampler: Optional[DistributedSampler] = None
        if isinstance(train_dataloader.sampler, DistributedSampler):
            self.train_sampler = train_dataloader.sampler

        # Initialize base trainer
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_fn=loss_fn,
            config=config,
            checkpoint_manager=checkpoint_manager,
            hooks=hooks,
        )

        # Untie shared weights to avoid DDP inplace errors
        # XLMRobertaForMaskedLM shares embeddings and lm_head weights
        base_model = self.model
        if hasattr(base_model, "model") and hasattr(base_model.model, "lm_head"):
            lm_model = base_model.model
            if hasattr(lm_model.lm_head, "decoder"):
                if (
                    lm_model.lm_head.decoder.weight
                    is lm_model.roberta.embeddings.word_embeddings.weight
                ):
                    lm_model.lm_head.decoder.weight = (
                        torch.nn.Parameter(
                            lm_model.lm_head.decoder.weight.clone()
                        )
                    )
                    logger.info("Untied lm_head/embedding weights for DDP")

        # Wrap model with DDP
        # broadcast_buffers=False prevents position_ids buffer sync
        # that causes inplace modification errors with multi-forward-pass
        self.model = DDP(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

        if self.is_main_process:
            logger.info(
                f"DDP initialized: world_size={world_size}, "
                f"local_rank={local_rank}"
            )

    @classmethod
    def setup_distributed(
        cls,
        local_rank: int,
        world_size: int,
        backend: str = "nccl",
    ) -> None:
        """
        Initialize distributed process group.

        Args:
            local_rank: Local GPU rank
            world_size: Total number of processes
            backend: Distributed backend (nccl for GPU)
        """
        torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                rank=local_rank,
                world_size=world_size,
            )

        logger.info(
            f"Process group initialized: "
            f"rank={local_rank}/{world_size}, "
            f"backend={backend}"
        )

    def train(self) -> Dict[str, Any]:
        """
        Run DDP training loop with barriers.

        Returns:
            Dict with training results and final metrics
        """
        # Synchronize before training
        dist.barrier()

        results = super().train()

        # Synchronize after training
        dist.barrier()

        return results

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with DDP synchronization.

        Sets sampler epoch for proper shuffling and
        synchronizes metrics across ranks.

        Args:
            epoch: Current epoch number

        Returns:
            Dict of training metrics
        """
        # Set epoch on sampler for proper shuffling
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        # Run base training epoch
        metrics = super()._train_epoch(epoch)

        # Synchronize metrics across ranks
        metrics = self._sync_metrics(metrics)

        return metrics

    def _sync_metrics(
        self,
        metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Synchronize metrics across all ranks via all_reduce.

        Computes the mean of each metric across all processes.

        Args:
            metrics: Local metrics dict

        Returns:
            Averaged metrics across all ranks
        """
        synced: Dict[str, float] = {}

        for key, value in metrics.items():
            tensor = torch.tensor(
                value, dtype=torch.float64, device=self.device
            )
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            synced[key] = tensor.item() / self.world_size

        return synced

    def _save_final_checkpoint(self) -> None:
        """Save final checkpoint on rank 0 only."""
        if not self.is_main_process:
            dist.barrier()
            return

        logger.info("Saving final checkpoint (rank 0)...")

        # Unwrap DDP model for saving
        model_to_save = self.model.module
        self.checkpoint_manager.save(
            model=model_to_save,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.global_step,
            metrics={"best_val_loss": self.best_val_loss},
        )

        dist.barrier()

    def resume_from_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Resume training from a checkpoint on all ranks.

        Loads checkpoint and synchronizes across ranks.

        Args:
            checkpoint_path: Optional path to specific checkpoint.
                If None, resumes from latest.
        """
        # Unwrap DDP model for loading
        model_to_load = self.model.module

        if checkpoint_path:
            info = self.checkpoint_manager.load(
                Path(checkpoint_path),
                model_to_load,
                self.optimizer,
                self.scheduler,
                str(self.device),
            )
        else:
            info = self.checkpoint_manager.load_latest(
                model_to_load,
                self.optimizer,
                self.scheduler,
                str(self.device),
            )

        self.current_epoch = info["epoch"]
        self.global_step = info["step"]
        self._resume_epoch = info["epoch"]
        self.best_val_loss = info.get(
            "metrics", {}
        ).get("best_val_loss", float("inf"))

        # Synchronize all ranks after loading
        dist.barrier()

        if self.is_main_process:
            logger.info(
                f"Resumed from epoch {self.current_epoch}, "
                f"step {self.global_step}"
            )

    def cleanup(self) -> None:
        """Destroy distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info(
                f"Process group destroyed (rank {self.local_rank})"
            )
