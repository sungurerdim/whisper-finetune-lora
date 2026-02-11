"""Merge LoRA weights and convert to CTranslate2 (faster-whisper) format.

Usage:
    python merge_and_convert.py --language tr --config config.yaml
    python merge_and_convert.py --language tr --checkpoint ./output/tr/checkpoints/checkpoint-1000
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights and convert to CTranslate2 format.",
    )
    parser.add_argument("--language", type=str, required=True, help="ISO 639-1 language code.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint path (default: best).")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the best checkpoint directory."""
    best = checkpoint_dir / "best"
    if best.exists():
        return best

    # Fall back to latest checkpoint by number
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        logger.error("No checkpoints found in %s", checkpoint_dir)
        sys.exit(1)
    return checkpoints[-1]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    language = args.language

    model_name = config["model"]["name"]
    output_base = Path(config["output"]["base_dir"]) / language
    quantization = config["output"]["ct2_quantization"]

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_best_checkpoint(output_base / "checkpoints")

    logger.info("Using checkpoint: %s", checkpoint_path)

    # Step 1: Load base model + LoRA adapter and merge
    logger.info("Loading base model: %s", model_name)
    base_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task="transcribe")

    logger.info("Loading LoRA adapter from %s", checkpoint_path)
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))

    logger.info("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Step 2: Save merged model in HuggingFace format
    merged_dir = output_base / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving merged model to %s", merged_dir)
    model.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))

    # Step 3: Convert to CTranslate2 format
    ct2_dir = output_base / f"faster-whisper-{language}"
    logger.info("Converting to CTranslate2 format (%s)...", quantization)

    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", str(merged_dir),
        "--output_dir", str(ct2_dir),
        "--quantization", quantization,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
    ]

    try:
        # Try the Python module approach first
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to ct2-transformers-converter CLI
        logger.info("Falling back to ct2-transformers-converter CLI...")
        cmd_cli = [
            "ct2-transformers-converter",
            "--model", str(merged_dir),
            "--output_dir", str(ct2_dir),
            "--quantization", quantization,
            "--copy_files", "tokenizer.json", "preprocessor_config.json",
        ]
        subprocess.run(cmd_cli, check=True)

    # Step 4: Copy additional tokenizer files
    for fname in ["tokenizer_config.json", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "added_tokens.json",
                   "normalizer.json"]:
        src = merged_dir / fname
        if src.exists():
            shutil.copy2(src, ct2_dir / fname)

    logger.info("CTranslate2 model saved to %s", ct2_dir)
    logger.info("Done! Model is ready for faster-whisper inference.")


if __name__ == "__main__":
    main()
