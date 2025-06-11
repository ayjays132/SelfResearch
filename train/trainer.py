from __future__ import annotations

"""Simple training utilities for language models."""

from dataclasses import dataclass
from typing import Optional
import argparse
import math

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from data.dataset_loader import load_dataset_splits


@dataclass
class TrainingConfig:
    """Configuration for language model training.

    Attributes:
        model_name: Model identifier from the HuggingFace Hub.
        dataset_name: Name of the dataset to load.
        train_split: Dataset split used for training.
        eval_split: Optional evaluation split.
        text_column: Column containing text samples.
        output_dir: Directory to save trained models.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        lr: Learning rate.
        grad_accum: Steps for gradient accumulation.
        device: Optional override for computation device.
    """

    model_name: str
    dataset_name: str
    train_split: str
    eval_split: Optional[str] = None
    text_column: str = "text"
    output_dir: str = "./model_output"
    epochs: int = 1
    batch_size: int = 2
    lr: float = 5e-5
    grad_accum: int = 2
    device: Optional[str] = None


def train_model(config: TrainingConfig) -> None:
    """Train a causal language model using HuggingFace Trainer."""
    device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_ds, eval_ds = load_dataset_splits(
        config.dataset_name,
        config.model_name,
        train_split=config.train_split,
        eval_split=config.eval_split,
        text_column=config.text_column,
    )

    model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        logging_steps=10,
        save_steps=50,
        gradient_accumulation_steps=config.grad_accum,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=50 if eval_ds is not None else None,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        shift_logits = torch.tensor(logits[:, :-1, :])
        shift_labels = torch.tensor(labels[:, 1:])
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        perplexity = math.exp(loss.item())
        return {"perplexity": perplexity}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_ds is not None else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_ds is not None else None,
    )

    trainer.train()
    if eval_ds is not None:
        trainer.evaluate()
    trainer.save_model(config.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a causal language model")
    parser.add_argument("model_name", help="Model identifier from the HuggingFace hub")
    parser.add_argument("dataset", help="Dataset name from the HuggingFace hub")
    parser.add_argument("train_split", help="Training split, e.g. 'train[:100]'")
    parser.add_argument("--eval-split", help="Optional evaluation split")
    parser.add_argument("--text-column", default="text", help="Name of the text column")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--output-dir", default="./model_output", help="Directory to save the model")
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model_name,
        dataset_name=args.dataset,
        train_split=args.train_split,
        eval_split=args.eval_split,
        text_column=args.text_column,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )
    train_model(config)


if __name__ == "__main__":
    main()
