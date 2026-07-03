from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

import evaluate
import librosa
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)


DATASET_ID = "thanhnew2001/VietSuperSpeech"
DEFAULT_MODEL_ID = "vinai/PhoWhisper-base"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick LoRA fine-tune PhoWhisper-base on VietSuperSpeech."
    )
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default="training_outputs/phowhisper-vss-lora")
    parser.add_argument("--train-samples", type=int, default=300)
    parser.add_argument("--eval-samples", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-duration", type=float, default=18.0)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def select_subset(dataset, sample_count: int, seed: int):
    if sample_count <= 0:
        return dataset.select([])
    sample_count = min(sample_count, len(dataset))
    return dataset.shuffle(seed=seed).select(range(sample_count))


def load_vietsuperspeech(args: argparse.Namespace):
    train = load_dataset(args.dataset_id, split="train")
    eval_dataset = load_dataset(args.dataset_id, split="validation")

    if "duration" in train.column_names:
        train = train.filter(
            lambda item: args.min_duration <= float(item["duration"]) <= args.max_duration
        )
        eval_dataset = eval_dataset.filter(
            lambda item: args.min_duration <= float(item["duration"]) <= args.max_duration
        )

    return {
        "train": select_subset(train, args.train_samples, args.seed),
        "eval": select_subset(eval_dataset, args.eval_samples, args.seed + 1),
    }


def resolve_audio_path(dataset_id: str, audio_value: Any) -> str:
    if isinstance(audio_value, dict):
        if audio_value.get("path"):
            return str(audio_value["path"])
        raise ValueError("Audio dict does not contain a local path.")

    audio_path = str(audio_value)
    if Path(audio_path).exists():
        return audio_path

    return hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        filename=audio_path,
    )


def prepare_example(
    example: dict[str, Any],
    processor: Any,
    dataset_id: str,
) -> dict[str, Any]:
    audio_path = resolve_audio_path(dataset_id, example["audio"])
    audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)

    example["input_features"] = processor.feature_extractor(
        audio_array,
        sampling_rate=16000,
    ).input_features[0]
    example["labels"] = processor.tokenizer(clean_text(example["text"])).input_ids
    return example


def build_training_arguments(args: argparse.Namespace) -> Seq2SeqTrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": min(10, max(1, args.max_steps // 10)),
        "max_steps": args.max_steps,
        "gradient_checkpointing": True,
        "fp16": torch.cuda.is_available(),
        "predict_with_generate": True,
        "generation_max_length": 225,
        "logging_steps": 10,
        "save_steps": args.max_steps,
        "save_total_limit": 1,
        "report_to": ["tensorboard"],
        "remove_unused_columns": False,
        "label_names": ["labels"],
    }

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"

    return Seq2SeqTrainingArguments(**kwargs)


def build_trainer(
    args: argparse.Namespace,
    model,
    train_dataset,
    eval_dataset,
    processor: Any,
    compute_metrics,
) -> Seq2SeqTrainer:
    kwargs = {
        "args": build_training_arguments(args),
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        ),
        "compute_metrics": compute_metrics,
    }

    signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = processor
    elif "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = processor.feature_extractor

    return Seq2SeqTrainer(**kwargs)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        language="vi",
        task="transcribe",
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="vi",
        task="transcribe",
    )
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = model.config.forced_decoder_ids
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_vietsuperspeech(args)
    keep_columns = ["input_features", "labels"]
    train_dataset = dataset["train"].map(
        lambda item: prepare_example(item, processor, args.dataset_id),
        remove_columns=[
            column for column in dataset["train"].column_names if column not in keep_columns
        ],
    )
    eval_dataset = dataset["eval"].map(
        lambda item: prepare_example(item, processor, args.dataset_id),
        remove_columns=[
            column for column in dataset["eval"].column_names if column not in keep_columns
        ],
    )

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True,
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids,
            skip_special_tokens=True,
        )
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    trainer = build_trainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved LoRA adapter and processor to: {output_dir}")


if __name__ == "__main__":
    main()
