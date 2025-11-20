"""
Production training script for SPLADE-doc model.

This script trains the model on the full dataset from notebooks 01-05:
- Korean Wikipedia paired data (01)
- NamuWiki paired data (01)
- Pre-training datasets: S2ORC, WikiAnswers, GOOAQ, etc. (03)
- Hard negatives mined with BM25 (04)
- MS MARCO for fine-tuning (05)

Usage:
    python train.py --config configs/pretrain.yaml
    python train.py --config configs/finetune.yaml
"""

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import yaml

from src.model.splade_model import create_splade_model, SPLADEDoc
from src.model.losses import SPLADELoss
from src.data.dataset import create_dataloaders

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Production trainer for SPLADE-doc."""

    def __init__(self, config: Dict):
        """
        Initialize trainer.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )

        logger.info(f"Initializing trainer on device: {self.device}")

        # Initialize model
        self.model = self._init_model()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

        # Initialize dataloaders
        self.train_loader, self.val_loader = self._init_dataloaders()

        # Initialize loss
        self.loss_fn = self._init_loss()

        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self._init_optimizer()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Mixed precision training
        self.mixed_precision = config['training'].get('mixed_precision', None)
        self.scaler = None
        if self.mixed_precision == 'fp16':
            self.scaler = GradScaler()
            logger.info("Using FP16 mixed precision training")
        elif self.mixed_precision == 'bf16':
            # BF16 doesn't need GradScaler
            logger.info("Using BF16 mixed precision training")

        # Output directory
        self.output_dir = Path(config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Trainer initialized successfully")

    def _init_model(self) -> SPLADEDoc:
        """Initialize SPLADE model."""
        model_config = self.config['model']

        model = create_splade_model(
            model_name=model_config['name'],
            use_idf=model_config.get('use_idf', False),
            dropout=model_config.get('dropout', 0.1),
        )

        model = model.to(self.device)

        # Load checkpoint if resuming
        if 'resume_from' in model_config and model_config['resume_from']:
            checkpoint_path = Path(model_config['resume_from'])
            if checkpoint_path.exists():
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Model: {model_config['name']}")
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def _init_dataloaders(self) -> tuple:
        """Initialize data loaders."""
        data_config = self.config['data']

        # Get data files
        train_files = self._get_data_files(data_config['train_patterns'])
        val_files = self._get_data_files(data_config['val_patterns'])

        logger.info(f"Train files: {len(train_files)}")
        logger.info(f"Val files: {len(val_files)}")

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_files=train_files,
            val_files=val_files if val_files else None,
            tokenizer=self.tokenizer,
            batch_size=data_config['batch_size'],
            max_length=data_config['max_length'],
            num_negatives=data_config.get('num_negatives', 7),
            num_workers=data_config.get('num_workers', 4),
            use_hard_negatives=data_config.get('use_hard_negatives', False),
        )

        logger.info(f"Train batches: {len(train_loader)}")
        if val_loader:
            logger.info(f"Val batches: {len(val_loader)}")

        return train_loader, val_loader

    def _get_data_files(self, patterns: List[str]) -> List[str]:
        """Get data files matching patterns."""
        files = []
        for pattern in patterns:
            matched = glob.glob(pattern)
            files.extend(matched)
        return sorted(files)

    def _init_loss(self) -> SPLADELoss:
        """Initialize loss function."""
        loss_config = self.config['loss']

        loss_fn = SPLADELoss(
            temperature=loss_config.get('temperature', 0.05),
            lambda_flops=loss_config.get('lambda_flops', 1e-4),
            lambda_idf=loss_config.get('lambda_idf', 1e-3),
            lambda_kd=loss_config.get('lambda_kd', 0.5),
            use_kd=loss_config.get('use_kd', False),
            use_idf_penalty=loss_config.get('use_idf_penalty', False),
        )

        loss_fn = loss_fn.to(self.device)
        return loss_fn

    def _init_optimizer(self) -> tuple:
        """Initialize optimizer and scheduler."""
        train_config = self.config['training']

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config.get('weight_decay', 0.01),
        )

        total_steps = (
            len(self.train_loader) * train_config['num_epochs']
            // train_config['gradient_accumulation_steps']
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_config.get('warmup_steps', 1000),
            num_training_steps=total_steps,
        )

        logger.info(f"Optimizer: AdamW (lr={train_config['learning_rate']})")
        logger.info(f"Total training steps: {total_steps}")

        return optimizer, scheduler

    def train(self):
        """Main training loop."""
        train_config = self.config['training']
        num_epochs = train_config['num_epochs']

        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info("=" * 80)
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {self.config['data']['batch_size']}")
        logger.info(f"Gradient accumulation: {train_config['gradient_accumulation_steps']}")
        logger.info("=" * 80)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train epoch
            train_loss = self._train_epoch()

            # Validate
            if self.val_loader:
                val_loss = self._validate()

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint('best_model')

            # Save epoch checkpoint
            if (epoch + 1) % train_config.get('save_epochs', 1) == 0:
                self._save_checkpoint(f'epoch_{epoch + 1}')

        logger.info("\n" + "=" * 80)
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 80)

    def _train_epoch(self) -> float:
        """Train one epoch."""
        self.model.train()
        train_config = self.config['training']

        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training (Step {self.global_step})"
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            loss, loss_dict = self._train_step(batch)

            # Backward
            loss = loss / train_config['gradient_accumulation_steps']

            if self.scaler is not None:
                # FP16 with GradScaler
                self.scaler.scale(loss).backward()
            else:
                # BF16 or FP32
                loss.backward()

            # Update
            if (step + 1) % train_config['gradient_accumulation_steps'] == 0:
                if self.scaler is not None:
                    # FP16: unscale before clipping
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    train_config['max_grad_norm']
                )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % train_config['log_steps'] == 0:
                    self._log_metrics(loss_dict)

            total_loss += loss.item() * train_config['gradient_accumulation_steps']
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss / num_batches:.4f}"
            })

        return total_loss / num_batches

    def _train_step(self, batch: Dict) -> tuple:
        """Single training step with optional mixed precision."""
        # Determine autocast dtype
        autocast_dtype = None
        if self.mixed_precision == 'bf16':
            autocast_dtype = torch.bfloat16
        elif self.mixed_precision == 'fp16':
            autocast_dtype = torch.float16

        # Use autocast if mixed precision is enabled
        with autocast(device_type='cuda', dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
            # Encode query
            query_repr, _ = self.model(
                batch['query_input_ids'],
                batch['query_attention_mask']
            )

            # Encode positive document
            pos_doc_repr, _ = self.model(
                batch['pos_doc_input_ids'],
                batch['pos_doc_attention_mask']
            )

            # Encode negative documents
            batch_size, num_neg, seq_len = batch['neg_doc_input_ids'].shape
            neg_input_ids = batch['neg_doc_input_ids'].view(batch_size * num_neg, seq_len)
            neg_attention_mask = batch['neg_doc_attention_mask'].view(batch_size * num_neg, seq_len)

            neg_doc_repr_flat, _ = self.model(neg_input_ids, neg_attention_mask)
            neg_doc_repr = neg_doc_repr_flat.view(batch_size, num_neg, -1)

            # Compute loss
            loss, loss_dict = self.loss_fn(
                query_repr,
                pos_doc_repr,
                neg_doc_repr,
            )

        return loss, loss_dict

    @torch.no_grad()
    def _validate(self) -> float:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            _, loss_dict = self._train_step(batch)
            total_loss += loss_dict['total']
            num_batches += 1

        avg_loss = total_loss / num_batches
        logger.info(f"Validation loss: {avg_loss:.4f}")

        return avg_loss

    def _save_checkpoint(self, name: str):
        """Save checkpoint."""
        checkpoint_path = self.output_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(
            {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'config': self.config,
            },
            checkpoint_path / 'checkpoint.pt'
        )

        logger.info(f"✓ Saved checkpoint: {checkpoint_path}")

    def _log_metrics(self, metrics: Dict):
        """Log metrics."""
        log_file = self.output_dir / 'training_log.jsonl'
        log_entry = {
            'step': self.global_step,
            'epoch': self.current_epoch,
            **metrics
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train SPLADE-doc model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize trainer
    trainer = Trainer(config)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
