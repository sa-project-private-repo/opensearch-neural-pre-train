"""Trainer for Neural Sparse model."""

import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.neural_sparse_encoder import NeuralSparseEncoder
from src.training.losses import CombinedLoss


class NeuralSparseTrainer:
    """
    Trainer for Neural Sparse model with mixed precision support.
    """

    def __init__(
        self,
        model: NeuralSparseEncoder,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        loss_fn: Optional[CombinedLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        output_dir: str = "outputs",
        save_steps: int = 1000,
        eval_steps: int = 500,
        logging_steps: int = 100,
    ):
        """
        Initialize trainer.

        Args:
            model: Neural Sparse encoder model
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            use_amp: Use automatic mixed precision
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            output_dir: Directory to save checkpoints
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            logging_steps: Log metrics every N steps
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Loss function
        self.loss_fn = loss_fn or CombinedLoss()

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=2e-5,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Mixed precision
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Training config
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Logging and saving
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        print(f"Initialized NeuralSparseTrainer:")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Output directory: {output_dir}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch dict

        Returns:
            Loss dict
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass with autocast
        with autocast(enabled=self.use_amp):
            # Encode query
            query_outputs = self.model(
                input_ids=batch["query_input_ids"],
                attention_mask=batch["query_attention_mask"],
            )
            query_rep = query_outputs["sparse_rep"]

            # Encode positive document
            pos_outputs = self.model(
                input_ids=batch["pos_doc_input_ids"],
                attention_mask=batch["pos_doc_attention_mask"],
            )
            pos_rep = pos_outputs["sparse_rep"]

            # Encode negative documents
            batch_size, num_neg, seq_len = batch["neg_doc_input_ids"].shape

            # Flatten negatives
            neg_input_ids = batch["neg_doc_input_ids"].view(batch_size * num_neg, seq_len)
            neg_attention_mask = batch["neg_doc_attention_mask"].view(
                batch_size * num_neg, seq_len
            )

            neg_outputs = self.model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
            )
            neg_rep = neg_outputs["sparse_rep"]

            # Reshape negatives
            neg_rep = neg_rep.view(batch_size, num_neg, -1)

            # Encode synonym pairs if present
            korean_rep = None
            english_rep = None
            if "korean_input_ids" in batch and "english_input_ids" in batch:
                korean_outputs = self.model(
                    input_ids=batch["korean_input_ids"],
                    attention_mask=batch["korean_attention_mask"],
                )
                korean_rep = korean_outputs["sparse_rep"]

                english_outputs = self.model(
                    input_ids=batch["english_input_ids"],
                    attention_mask=batch["english_attention_mask"],
                )
                english_rep = english_outputs["sparse_rep"]

            # Compute loss
            losses = self.loss_fn(
                query_rep=query_rep,
                pos_doc_rep=pos_rep,
                neg_doc_reps=neg_rep,
                korean_rep=korean_rep,
                english_rep=english_rep,
            )

            total_loss = losses["total_loss"]

        # Scale loss for gradient accumulation
        scaled_loss = total_loss / self.gradient_accumulation_steps

        # Backward pass
        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Return losses as floats
        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average losses for epoch
        """
        self.model.train()
        epoch_losses = {
            "total_loss": 0.0,
            "ranking_loss": 0.0,
            "cross_lingual_loss": 0.0,
            "sparsity_loss": 0.0,
        }
        num_batches = 0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            # Training step
            losses = self.train_step(batch)

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            num_batches += 1

            # Optimizer step (with gradient accumulation)
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_loss = epoch_losses["total_loss"] / num_batches
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                # Evaluation
                if self.val_dataloader and self.global_step % self.eval_steps == 0:
                    val_losses = self.evaluate()
                    print(f"\nValidation at step {self.global_step}:")
                    print(f"  Val loss: {val_losses['total_loss']:.4f}")

                    # Save best model
                    if val_losses["total_loss"] < self.best_val_loss:
                        self.best_val_loss = val_losses["total_loss"]
                        self.save_checkpoint("best_model")
                        print(f"  New best model saved!")

                    self.model.train()

                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

        # Average losses
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        return avg_losses

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Returns:
            Average validation losses
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        val_losses = {
            "total_loss": 0.0,
            "ranking_loss": 0.0,
            "cross_lingual_loss": 0.0,
            "sparsity_loss": 0.0,
        }
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Validation"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            query_outputs = self.model(
                input_ids=batch["query_input_ids"],
                attention_mask=batch["query_attention_mask"],
            )
            query_rep = query_outputs["sparse_rep"]

            pos_outputs = self.model(
                input_ids=batch["pos_doc_input_ids"],
                attention_mask=batch["pos_doc_attention_mask"],
            )
            pos_rep = pos_outputs["sparse_rep"]

            # Encode negatives
            batch_size, num_neg, seq_len = batch["neg_doc_input_ids"].shape
            neg_input_ids = batch["neg_doc_input_ids"].view(batch_size * num_neg, seq_len)
            neg_attention_mask = batch["neg_doc_attention_mask"].view(
                batch_size * num_neg, seq_len
            )

            neg_outputs = self.model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
            )
            neg_rep = neg_outputs["sparse_rep"].view(batch_size, num_neg, -1)

            # Compute loss
            losses = self.loss_fn(
                query_rep=query_rep,
                pos_doc_rep=pos_rep,
                neg_doc_reps=neg_rep,
            )

            # Accumulate
            for key in val_losses:
                val_losses[key] += losses[key].item()
            num_batches += 1

        # Average
        avg_losses = {k: v / num_batches for k, v in val_losses.items()}
        return avg_losses

    def train(self, num_epochs: int) -> None:
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Total training steps: {len(self.train_dataloader) * num_epochs}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_losses = self.train_epoch(epoch)

            print(f"\nEpoch {epoch} completed:")
            print(f"  Train loss: {epoch_losses['total_loss']:.4f}")
            print(f"  Ranking loss: {epoch_losses['ranking_loss']:.4f}")
            print(f"  Cross-lingual loss: {epoch_losses['cross_lingual_loss']:.4f}")
            print(f"  Sparsity loss: {epoch_losses['sparsity_loss']:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch-{epoch}")

        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, checkpoint_name: str) -> None:
        """
        Save model checkpoint.

        Args:
            checkpoint_name: Name of checkpoint
        """
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(str(checkpoint_dir))

        # Save trainer state
        torch.save(
            {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "best_val_loss": self.best_val_loss,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
            },
            checkpoint_dir / "trainer_state.pt",
        )

        print(f"Checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    print("Trainer module loaded successfully.")
    print("Use this module by importing NeuralSparseTrainer in your training script.")
