from __future__ import annotations

"""Simple training utilities for language models."""

from dataclasses import dataclass
from typing import Optional
import argparse

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from data.dataset_loader import load_and_tokenize


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_name: str
    dataset_name: str
    split: str
    output_dir: str = "./model_output"
    epochs: int = 1
    batch_size: int = 2
    lr: float = 5e-5
    device: Optional[str] = None


def train_model(config: TrainingConfig) -> None:
    """Train a causal language model using HuggingFace Trainer."""
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenized_ds = load_and_tokenize(config.dataset_name, config.split, config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        logging_steps=10,
        save_steps=50,
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_ds, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(config.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a causal language model")
    parser.add_argument("model_name", help="Model identifier from the HuggingFace hub")
    parser.add_argument("dataset", help="Dataset name from the HuggingFace hub")
    parser.add_argument("split", help="Dataset split, e.g. 'train[:100]'")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--output-dir", default="./model_output", help="Directory to save the model")
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model_name,
        dataset_name=args.dataset,
        split=args.split,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    train_model(config)


if __name__ == "__main__":
    main()
