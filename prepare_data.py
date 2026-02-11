"""Download, preprocess, and optionally augment datasets for Whisper fine-tuning.

Usage:
    python prepare_data.py --language tr --config config.yaml
    python prepare_data.py --list-languages
    python prepare_data.py --language tr --sources common_voice fleurs --no-augment
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from transformers import WhisperProcessor

from data_sources import get_sources_for_language, list_all_languages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Whisper LoRA fine-tuning.",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="ISO 639-1 language code (e.g. tr, de, en).",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List all available languages and exit.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Only use these sources (short names: common_voice, fleurs, etc.).",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Skip data augmentation.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of parallel workers for dataset processing.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_and_normalize(
    source: dict,
    split_name: str,
    sample_rate: int,
    max_duration: float,
) -> "Dataset | None":
    """Download a single split from a source and normalize columns."""
    dataset_id = source["dataset_id"]
    hf_split = source["split_map"].get(split_name)
    if hf_split is None:
        return None

    logger.info("Loading %s [%s] ...", dataset_id, hf_split)
    try:
        ds = load_dataset(
            dataset_id,
            split=hf_split,
            **source["load_kwargs"],
        )
    except Exception:
        logger.warning("Failed to load %s [%s], skipping.", dataset_id, hf_split)
        return None

    # Rename columns to standard names
    audio_col = source["audio_column"]
    text_col = source["text_column"]
    if audio_col != "audio":
        ds = ds.rename_column(audio_col, "audio")
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")

    # Keep only audio + text
    keep_cols = {"audio", "text"}
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)

    # Resample audio
    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))

    # Filter by duration
    def is_valid(example: dict) -> bool:
        audio = example["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        return 0.5 <= duration <= max_duration

    ds = ds.filter(is_valid, num_proc=1)

    # Filter empty transcripts
    ds = ds.filter(lambda x: bool(x["text"] and x["text"].strip()), num_proc=1)

    logger.info("  %s [%s]: %d samples after filtering.", dataset_id, hf_split, len(ds))
    return ds


def apply_augmentation(dataset: "Dataset", config: dict) -> "Dataset":
    """Apply audio augmentation using audiomentations."""
    try:
        import audiomentations as am
        import numpy as np
    except ImportError:
        logger.warning("audiomentations not installed, skipping augmentation.")
        return dataset

    aug_cfg = config.get("augmentation", {})
    speed_rates = aug_cfg.get("speed_perturbation", [0.9, 1.0, 1.1])
    noise_snrs = aug_cfg.get("noise_snr_db", [10, 15, 20])

    transforms = am.Compose([
        am.TimeStretch(
            min_rate=min(speed_rates),
            max_rate=max(speed_rates),
            p=0.5,
        ),
        am.AddGaussianNoise(
            min_amplitude=1e-4,
            max_amplitude=5e-3,
            p=0.3,
        ),
    ])

    def augment(example: dict) -> dict:
        audio = example["audio"]
        samples = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        augmented = transforms(samples=samples, sample_rate=sr)
        example["audio"] = {"array": augmented, "sampling_rate": sr}
        return example

    logger.info("Applying augmentation to %d samples...", len(dataset))
    return dataset.map(augment, num_proc=1)


def prepare_features(
    dataset: "Dataset",
    processor: WhisperProcessor,
    language: str,
    num_proc: int,
) -> "Dataset":
    """Extract Whisper features (log-mel spectrogram + tokenized labels)."""

    def process(batch: dict) -> dict:
        audio = batch["audio"]
        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="np",
        )
        batch["input_features"] = inputs.input_features[0]

        labels = processor.tokenizer(
            batch["text"],
            return_tensors="np",
        )
        batch["labels"] = labels.input_ids[0]
        return batch

    logger.info("Extracting features for %d samples...", len(dataset))
    dataset = dataset.map(
        process,
        remove_columns=["audio", "text"],
        num_proc=num_proc,
    )
    return dataset


def main() -> None:
    args = parse_args()

    if args.list_languages:
        langs = list_all_languages()
        print(f"\n{'Language':<10} {'Sources':>3}  Details")
        print("-" * 60)
        for lang, sources in langs.items():
            print(f"{lang:<10} {len(sources):>3}  {', '.join(sources)}")
        print(f"\nTotal: {len(langs)} languages")
        sys.exit(0)

    if not args.language:
        print("Error: --language is required (or use --list-languages).")
        sys.exit(1)

    config = load_config(args.config)
    language = args.language
    sample_rate = config["data"]["sample_rate"]
    max_duration = config["data"]["max_duration_seconds"]

    # Get sources for this language
    sources = get_sources_for_language(language)
    if not sources:
        print(f"Error: No datasets found for language '{language}'.")
        print("Run --list-languages to see available languages.")
        sys.exit(1)

    # Filter by user-specified sources
    if args.sources:
        sources = [s for s in sources if s["short_name"] in args.sources]
        if not sources:
            print(f"Error: None of {args.sources} available for '{language}'.")
            sys.exit(1)

    logger.info(
        "Language: %s | Sources: %s",
        language,
        ", ".join(s["short_name"] for s in sources),
    )

    # Download and merge datasets for each split
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for source in sources:
        for split_name in splits:
            ds = download_and_normalize(source, split_name, sample_rate, max_duration)
            if ds is not None and len(ds) > 0:
                splits[split_name].append(ds)

    merged = {}
    for split_name, ds_list in splits.items():
        if ds_list:
            merged[split_name] = concatenate_datasets(ds_list)
            logger.info("Merged %s: %d samples", split_name, len(merged[split_name]))
        else:
            logger.warning("No data for split: %s", split_name)

    if not merged:
        print("Error: No data loaded for any split.")
        sys.exit(1)

    # If only train split exists, create val/test from it
    if "train" in merged and "val" not in merged:
        logger.info("No validation split found, splitting train 80/10/10.")
        train_test = merged["train"].train_test_split(test_size=0.2, seed=42)
        val_test = train_test["test"].train_test_split(test_size=0.5, seed=42)
        merged["train"] = train_test["train"]
        merged["val"] = val_test["train"]
        merged["test"] = val_test["test"]

    # Apply augmentation to training set only
    if not args.no_augment and "train" in merged:
        merged["train"] = apply_augmentation(merged["train"], config)

    # Extract features
    model_name = config["model"]["name"]
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task="transcribe")

    for split_name in merged:
        merged[split_name] = prepare_features(
            merged[split_name],
            processor,
            language,
            args.num_proc,
        )

    # Save
    output_dir = Path(config["output"]["base_dir"]) / language / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = DatasetDict(merged)
    dataset_dict.save_to_disk(str(output_dir))
    logger.info("Dataset saved to %s", output_dir)

    # Summary
    for split_name, ds in merged.items():
        logger.info("  %s: %d samples", split_name, len(ds))


if __name__ == "__main__":
    main()
