"""Evaluate fine-tuned Whisper model: WER, CER, and RTFx benchmarks.

Usage:
    python evaluate.py --language tr --config config.yaml
    python evaluate.py --language tr --config config.yaml --compare-baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import yaml
from jiwer import cer, wer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Whisper model.",
    )
    parser.add_argument("--language", type=str, required=True, help="ISO 639-1 language code.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare with original large-v3.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def transcribe_dataset(
    model_path: str,
    test_data: list[dict],
    *,
    device: str = "auto",
    compute_type: str = "float16",
) -> tuple[list[str], float, float]:
    """Transcribe all test samples and return predictions + timing info.

    Returns:
        Tuple of (predictions, total_audio_duration, total_inference_time).
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_path, device=device, compute_type=compute_type)

    predictions = []
    total_audio_duration = 0.0
    total_inference_time = 0.0

    for i, sample in enumerate(test_data):
        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]
        audio_duration = len(audio_array) / sample_rate
        total_audio_duration += audio_duration

        start = time.perf_counter()
        segments, _info = model.transcribe(
            audio_array,
            language=None,  # Let model detect or use forced language from config
        )
        text = " ".join(seg.text.strip() for seg in segments)
        elapsed = time.perf_counter() - start
        total_inference_time += elapsed

        predictions.append(text)

        if (i + 1) % 100 == 0:
            logger.info("  Transcribed %d/%d samples...", i + 1, len(test_data))

    return predictions, total_audio_duration, total_inference_time


def compute_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute WER and CER."""
    # Filter out empty pairs
    valid = [
        (p, r) for p, r in zip(predictions, references)
        if r.strip()
    ]
    if not valid:
        return {"wer": 0.0, "cer": 0.0}

    preds, refs = zip(*valid)
    return {
        "wer": wer(list(refs), list(preds)),
        "cer": cer(list(refs), list(preds)),
    }


def print_results(
    label: str,
    metrics: dict[str, float],
    audio_duration: float,
    inference_time: float,
    num_samples: int,
) -> None:
    """Print evaluation results as a formatted table."""
    rtfx = audio_duration / inference_time if inference_time > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Samples:     {num_samples}")
    print(f"  WER:         {metrics['wer']:.2%}")
    print(f"  CER:         {metrics['cer']:.2%}")
    print(f"  Audio:       {audio_duration:.1f}s")
    print(f"  Inference:   {inference_time:.1f}s")
    print(f"  RTFx:        {rtfx:.1f}x")
    print(f"{'=' * 50}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    language = args.language

    output_base = Path(config["output"]["base_dir"]) / language
    model_path = str(output_base / f"faster-whisper-{language}")

    if not Path(model_path).exists():
        logger.error("Model not found: %s", model_path)
        logger.error("Run merge_and_convert.py first.")
        raise SystemExit(1)

    # Load test dataset
    from datasets import DatasetDict

    data_dir = output_base / "data"
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        raise SystemExit(1)

    dataset = DatasetDict.load_from_disk(str(data_dir))
    if "test" not in dataset:
        logger.error("No test split found in dataset.")
        raise SystemExit(1)

    # We need the raw audio for faster-whisper, so reload without feature extraction
    # The test set references are in the processed dataset labels
    # For evaluation, we reload the original test data
    logger.info("Loading test data...")

    from data_sources import get_sources_for_language
    from datasets import Audio, concatenate_datasets, load_dataset

    sample_rate = config["data"]["sample_rate"]
    sources = get_sources_for_language(language)

    test_datasets = []
    for source in sources:
        hf_split = source["split_map"].get("test")
        if not hf_split:
            continue
        try:
            ds = load_dataset(
                source["dataset_id"],
                split=hf_split,
                **source["load_kwargs"],
            )
            audio_col = source["audio_column"]
            text_col = source["text_column"]
            if audio_col != "audio":
                ds = ds.rename_column(audio_col, "audio")
            if text_col != "text":
                ds = ds.rename_column(text_col, "text")
            keep = {"audio", "text"}
            ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
            ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
            ds = ds.filter(lambda x: bool(x["text"] and x["text"].strip()))
            test_datasets.append(ds)
        except Exception:
            logger.warning("Failed to load test data from %s", source["dataset_id"])

    if not test_datasets:
        logger.error("Could not load any test data.")
        raise SystemExit(1)

    test_data = concatenate_datasets(test_datasets)
    test_list = list(test_data)
    references = [s["text"] for s in test_list]

    logger.info("Evaluating fine-tuned model (%d samples)...", len(test_list))
    predictions, audio_dur, infer_time = transcribe_dataset(model_path, test_list)
    metrics = compute_metrics(predictions, references)
    rtfx = audio_dur / infer_time if infer_time > 0 else 0

    print_results(
        f"Fine-tuned Model ({language})",
        metrics,
        audio_dur,
        infer_time,
        len(test_list),
    )

    results = {
        "language": language,
        "model": f"faster-whisper-{language}",
        "num_samples": len(test_list),
        "wer": metrics["wer"],
        "cer": metrics["cer"],
        "rtfx": rtfx,
        "audio_duration_s": audio_dur,
        "inference_time_s": infer_time,
    }

    # Baseline comparison
    if args.compare_baseline:
        logger.info("Evaluating baseline (original large-v3)...")
        base_preds, base_audio, base_time = transcribe_dataset(
            config["model"]["name"],
            test_list,
        )
        base_metrics = compute_metrics(base_preds, references)
        base_rtfx = base_audio / base_time if base_time > 0 else 0

        print_results(
            f"Baseline large-v3 ({language})",
            base_metrics,
            base_audio,
            base_time,
            len(test_list),
        )

        wer_improvement = base_metrics["wer"] - metrics["wer"]
        cer_improvement = base_metrics["cer"] - metrics["cer"]

        print(f"\n  WER improvement: {wer_improvement:+.2%}")
        print(f"  CER improvement: {cer_improvement:+.2%}")

        results["baseline"] = {
            "wer": base_metrics["wer"],
            "cer": base_metrics["cer"],
            "rtfx": base_rtfx,
        }
        results["improvement"] = {
            "wer": wer_improvement,
            "cer": cer_improvement,
        }

    # Save results
    results_path = output_base / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
