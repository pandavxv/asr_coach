from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from presentation_coach.asr import (
    DEFAULT_LORA_ADAPTER_DIR,
    DEFAULT_MODEL_ID,
    PHOWHISPER_SMALL_MODEL_ID,
    has_local_lora_adapter,
    load_asr_pipeline,
    transcribe_audio,
)
from presentation_coach.evaluation import cer, wer


DATASET_ID = "thanhnew2001/VietSuperSpeech"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PhoWhisper-base and optional LoRA adapter on VietSuperSpeech."
    )
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-dir", default=str(DEFAULT_LORA_ADAPTER_DIR))
    parser.add_argument("--mode", choices=["base", "lora", "both"], default="base")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=18.0)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--audio-cache-dir", default="cache/vietsuperspeech")
    return parser.parse_args()


def levenshtein(reference_units: list[str], hypothesis_units: list[str]) -> int:
    if len(reference_units) < len(hypothesis_units):
        reference_units, hypothesis_units = hypothesis_units, reference_units

    previous_row = list(range(len(hypothesis_units) + 1))
    for index, reference_unit in enumerate(reference_units, start=1):
        current_row = [index]
        for column, hypothesis_unit in enumerate(hypothesis_units, start=1):
            current_row.append(
                min(
                    current_row[column - 1] + 1,
                    previous_row[column] + 1,
                    previous_row[column - 1] + (reference_unit != hypothesis_unit),
                )
            )
        previous_row = current_row
    return previous_row[-1]


def raw_wer(reference: str, hypothesis: str) -> float:
    reference_words = str(reference).split()
    hypothesis_words = str(hypothesis).split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return levenshtein(reference_words, hypothesis_words) / len(reference_words)


def clean_text(text: Any) -> str:
    return " ".join(str(text).strip().split())


def load_eval_subset(args: argparse.Namespace):
    dataset = load_dataset(args.dataset_id, split="validation")
    if "duration" in dataset.column_names:
        dataset = dataset.filter(
            lambda item: args.min_duration
            <= float(item["duration"])
            <= args.max_duration
        )
    sample_count = min(args.samples, len(dataset))
    return dataset.shuffle(seed=args.seed + 1).select(range(sample_count))


def resolve_audio_path(
    dataset_id: str,
    audio_value: Any,
    audio_cache_dir: str | Path,
) -> str:
    if isinstance(audio_value, dict):
        if audio_value.get("path"):
            return str(audio_value["path"])
        raise ValueError("Audio dict does not contain a local path.")

    audio_path = str(audio_value)
    if Path(audio_path).exists():
        return audio_path

    Path(audio_cache_dir).mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        filename=audio_path,
        local_dir=audio_cache_dir,
    )


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "samples": len(rows),
        "raw_wer": sum(row["raw_wer"] for row in rows) / len(rows),
        "normalized_wer": sum(row["normalized_wer"] for row in rows) / len(rows),
        "normalized_cer": sum(row["normalized_cer"] for row in rows) / len(rows),
    }


def model_label(model_id: str, adapter_dir: str | None = None) -> str:
    label = "phowhisper-final"
    if model_id != PHOWHISPER_SMALL_MODEL_ID:
        label = model_id.rsplit("/", maxsplit=1)[-1].lower()
    if adapter_dir:
        return f"{label}-lora"
    return label


def evaluate_model(
    model_label: str,
    args: argparse.Namespace,
    dataset,
    adapter_dir: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, example in enumerate(dataset, start=1):
        audio_path = resolve_audio_path(
            args.dataset_id,
            example["audio"],
            args.audio_cache_dir,
        )
        reference = clean_text(example["text"])
        prediction = transcribe_audio(
            audio_path,
            model_id=args.model_id,
            adapter_dir=adapter_dir,
        )

        row = {
            "model": model_label,
            "sample_index": index,
            "audio": str(example["audio"]),
            "duration": float(example.get("duration", 0.0)),
            "reference": reference,
            "prediction": prediction,
            "raw_wer": raw_wer(reference, prediction),
            "normalized_wer": wer(reference, prediction),
            "normalized_cer": cer(reference, prediction),
        }
        rows.append(row)
        print(
            f"{model_label} {index}/{len(dataset)} "
            f"raw_wer={row['raw_wer']:.3f} norm_wer={row['normalized_wer']:.3f}"
        )
    return rows


def save_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"vietsuperspeech_eval_{args.mode}_{args.samples}_{timestamp}"

    csv_path = output_dir / f"{base_name}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_id": args.dataset_id,
        "model_id": args.model_id,
        "mode": args.mode,
        "samples": args.samples,
        "seed": args.seed,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "audio_cache_dir": args.audio_cache_dir,
        "metrics": {},
    }
    for model_label in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model_label]
        summary["metrics"][model_label] = metric_summary(model_rows)

    json_path = output_dir / f"{base_name}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved predictions to: {csv_path}")
    print(f"Saved summary to: {json_path}")


def main() -> None:
    args = parse_args()
    dataset = load_eval_subset(args)

    all_rows: list[dict[str, Any]] = []
    if args.mode in {"base", "both"}:
        all_rows.extend(evaluate_model(model_label(args.model_id), args, dataset))
        load_asr_pipeline.cache_clear()

    if args.mode in {"lora", "both"}:
        if not has_local_lora_adapter(args.adapter_dir):
            raise FileNotFoundError(f"LoRA adapter not found: {args.adapter_dir}")
        all_rows.extend(
            evaluate_model(
                model_label(args.model_id, args.adapter_dir),
                args,
                dataset,
                adapter_dir=args.adapter_dir,
            )
        )
        load_asr_pipeline.cache_clear()

    save_outputs(args, all_rows)


if __name__ == "__main__":
    main()
