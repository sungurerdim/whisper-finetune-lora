"""LoRA FP16 fine-tuning for Whisper using HuggingFace Seq2SeqTrainer.

Usage:
    python train_lora.py --language tr --config config.yaml
    python train_lora.py --language tr --resume-from-checkpoint ./output/tr/checkpoints/checkpoint-500
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate as hf_evaluate
import torch
import yaml
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper with LoRA.",
    )
    parser.add_argument("--language", type=str, required=True, help="ISO 639-1 language code.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class DataCollatorSpeechSeq2Seq:
    """Collate input features and labels for Whisper training."""

    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 so it's ignored by loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if it was prepended
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    language = args.language

    model_name = config["model"]["name"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    output_base = Path(config["output"]["base_dir"]) / language

    # Load processor and model
    logger.info("Loading model: %s", model_name)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Set forced decoder IDs for the target language
    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe",
    )

    # Apply LoRA
    logger.info("Applying LoRA config: r=%d, alpha=%d, modules=%s", lora_cfg["r"], lora_cfg["alpha"], lora_cfg["target_modules"])
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load prepared data
    data_dir = output_base / "data"
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        logger.error("Run prepare_data.py first.")
        raise SystemExit(1)

    logger.info("Loading prepared dataset from %s", data_dir)
    dataset = DatasetDict.load_from_disk(str(data_dir))

    # WER metric
    wer_metric = hf_evaluate.load("wer")

    def compute_metrics(pred: Any) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # Replace -100 with pad token for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Data collator
    data_collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Training arguments
    checkpoint_dir = output_base / "checkpoints"
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        fp16=train_cfg["fp16"],
        evaluation_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        logging_dir=str(output_base / "logs"),
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("val", dataset.get("test")),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=train_cfg["early_stopping_patience"],
            ),
        ],
    )

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save best model
    best_dir = checkpoint_dir / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    logger.info("Best model saved to %s", best_dir)

    # Final evaluation
    logger.info("Running final evaluation...")
    metrics = trainer.evaluate()
    logger.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
