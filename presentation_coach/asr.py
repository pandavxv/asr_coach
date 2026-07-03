from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from presentation_coach.audio import load_audio_array

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


DEFAULT_MODEL_ID = "vinai/PhoWhisper-base"
PHOWHISPER_SMALL_MODEL_ID = "vinai/PhoWhisper-small"
DEFAULT_LORA_ADAPTER_DIR = Path(__file__).resolve().parents[1] / "phowhisper-vss-lora"
PHOWHISPER_SMALL_LORA_ADAPTER_DIR = (
    Path(__file__).resolve().parents[1] / "phowhisper-small-vss-lora"
)


LZMA_IMPORT_HINT = (
    "Python environment is missing the LZMA runtime used by Hugging Face "
    "Transformers. On Windows, recreate the conda env with `conda create -n "
    "asr_coach python=3.11 pip`, or run `conda install -n asr_coach -c "
    "conda-forge liblzma`, then reinstall requirements."
)


def _is_lzma_import_error(exc: ImportError) -> bool:
    return exc.name in {"_lzma", "lzma"} or "_lzma" in str(exc)


def has_local_lora_adapter(adapter_dir: str | Path = DEFAULT_LORA_ADAPTER_DIR) -> bool:
    adapter_path = Path(adapter_dir)
    return (
        (adapter_path / "adapter_config.json").exists()
        and (adapter_path / "adapter_model.safetensors").exists()
    )


@lru_cache(maxsize=4)
def load_asr_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    adapter_dir: str | None = None,
):
    try:
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration
        from transformers import pipeline
    except ImportError as exc:
        if _is_lzma_import_error(exc):
            raise RuntimeError(LZMA_IMPORT_HINT) from exc
        raise

    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if device == 0 else torch.float32

    if adapter_dir:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "Local LoRA adapter requires `peft`. Install it with `pip install peft`."
            ) from exc

        adapter_path = Path(adapter_dir)
        processor = AutoProcessor.from_pretrained(adapter_path)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
        model.eval()

        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            chunk_length_s=30,
            batch_size=1,
        )

    kwargs = {
        "model": model_id,
        "device": device,
        "chunk_length_s": 30,
        "batch_size": 1,
    }
    if device == 0:
        kwargs["torch_dtype"] = torch_dtype

    return pipeline("automatic-speech-recognition", **kwargs)


def transcribe_audio(
    audio_path: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    adapter_dir: str | Path | None = None,
) -> str:
    asr_pipeline = load_asr_pipeline(
        model_id=model_id,
        adapter_dir=str(adapter_dir) if adapter_dir else None,
    )
    path = str(audio_path)
    audio_array, sample_rate = load_audio_array(path)
    pipeline_input = {"array": audio_array, "sampling_rate": sample_rate}

    try:
        result = asr_pipeline(
            pipeline_input,
            generate_kwargs={"language": "vi", "task": "transcribe"},
        )
    except (TypeError, ValueError):
        result = asr_pipeline(pipeline_input)

    if isinstance(result, dict):
        return result.get("text", "").strip()
    return str(result).strip()
