#!/usr/bin/env python3
"""Fine-tune pko-t5-base on KorQuAD for answer-free Korean question generation.

Downloads KorQuAD 1.0 (squad_kor_v1) from HuggingFace, extracts (context, question)
pairs ignoring answers, and trains a seq2seq model with input format:
    "generate question: {context}" -> "{question}"

The resulting model can be used by expand_documents.py to perform doc2query
document expansion for improving neural sparse retrieval recall.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_PREFIX = "generate question: "


def load_korquad() -> DatasetDict:
    """Download and return the KorQuAD 1.0 dataset from HuggingFace.

    Returns:
        DatasetDict with 'train' and 'validation' splits.
    """
    logger.info("Loading KorQuAD 1.0 (squad_kor_v1) from HuggingFace...")
    dataset: DatasetDict = load_dataset("squad_kor_v1")
    logger.info(
        f"KorQuAD loaded: {len(dataset['train'])} train, "
        f"{len(dataset['validation'])} validation samples"
    )
    return dataset


def extract_context_question_pairs(
    dataset: DatasetDict,
) -> DatasetDict:
    """Extract (context, question) pairs from KorQuAD, ignoring answers.

    Maps each example to a dict with 'context' and 'question' keys.
    Answers are intentionally dropped for answer-free question generation.

    Args:
        dataset: KorQuAD DatasetDict with 'train' and 'validation' splits.

    Returns:
        DatasetDict containing only 'context' and 'question' columns.
    """
    logger.info("Extracting (context, question) pairs (answer-free)...")

    def _extract(example: dict[str, Any]) -> dict[str, str]:
        return {
            "context": example["context"],
            "question": example["question"],
        }

    extracted = dataset.map(
        _extract,
        remove_columns=[
            col
            for col in dataset["train"].column_names
            if col not in ("context", "question")
        ],
        desc="Extracting pairs",
    )
    logger.info("Extraction complete")
    return extracted


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_input_length: int,
    max_output_length: int,
) -> DatasetDict:
    """Tokenize (context, question) pairs for seq2seq training.

    Input format: "generate question: {context}" truncated to max_input_length.
    Target: "{question}" truncated to max_output_length.

    Args:
        dataset: DatasetDict with 'context' and 'question' columns.
        tokenizer: Tokenizer from the pre-trained model.
        max_input_length: Maximum number of input tokens.
        max_output_length: Maximum number of output tokens.

    Returns:
        DatasetDict with 'input_ids', 'attention_mask', 'labels' columns.
    """
    logger.info(
        f"Tokenizing dataset (max_input={max_input_length}, max_output={max_output_length})..."
    )

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        inputs = [INPUT_PREFIX + ctx for ctx in batch["context"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["question"],
                max_length=max_output_length,
                truncation=True,
                padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=["context", "question"],
        desc="Tokenizing",
    )
    logger.info("Tokenization complete")
    return tokenized


def build_training_args(
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_ratio: float,
) -> Seq2SeqTrainingArguments:
    """Build Seq2SeqTrainingArguments with sane defaults.

    Args:
        output_dir: Directory where checkpoints and logs are saved.
        epochs: Number of training epochs.
        batch_size: Per-device train and eval batch size.
        lr: Learning rate.
        warmup_ratio: Fraction of total steps used for linear warmup.

    Returns:
        Configured Seq2SeqTrainingArguments instance.
    """
    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        predict_with_generate=True,
        generation_max_length=64,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
        save_total_limit=2,
        dataloader_num_workers=4,
        report_to="none",
        push_to_hub=False,
    )


def train(
    model_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_ratio: float,
    max_input_length: int,
    max_output_length: int,
) -> None:
    """Run the full fine-tuning pipeline.

    Downloads KorQuAD, tokenizes data, and trains pko-t5-base with
    Seq2SeqTrainer. Best checkpoint is saved to output_dir.

    Args:
        model_name: HuggingFace model identifier (default: paust/pko-t5-base).
        output_dir: Directory for saving checkpoints and the final model.
        epochs: Number of training epochs.
        batch_size: Per-device batch size for training and evaluation.
        lr: Peak learning rate.
        warmup_ratio: Fraction of steps for learning rate warmup.
        max_input_length: Maximum tokenized context length.
        max_output_length: Maximum tokenized question length.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        model_name
    )

    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )

    dataset = load_korquad()
    dataset = extract_context_question_pairs(dataset)
    tokenized = tokenize_dataset(dataset, tokenizer, max_input_length, max_output_length)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    training_args = build_training_args(
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        warmup_ratio=warmup_ratio,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    logger.info(f"Saving best model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Fine-tuning complete")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune pko-t5-base on KorQuAD for answer-free question generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/doc2query_ko"),
        help="Directory to save fine-tuned model and checkpoints",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="paust/pko-t5-base",
        help="HuggingFace model identifier for the base T5 model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device batch size for training and evaluation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of total steps used for linear learning rate warmup",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=512,
        help="Maximum number of input tokens (context)",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=64,
        help="Maximum number of output tokens (generated question)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for doc2query fine-tuning."""
    args = parse_args()
    train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
    )


if __name__ == "__main__":
    main()
