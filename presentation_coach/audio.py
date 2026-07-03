from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


FFMPEG_MISSING_HINT = (
    "FFmpeg is required to read this audio file. Install it with "
    "`conda install -c conda-forge ffmpeg -y`, then restart the app."
)


def can_read_audio_info(audio_path: str | Path) -> bool:
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return info.frames > 0 and info.samplerate > 0
    except Exception:
        return False


def load_audio_array(
    audio_path: str | Path,
    target_sample_rate: int = 16000,
):
    path = str(audio_path)

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        import librosa

        audio, sample_rate = librosa.load(path, sr=None, mono=True)

    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    if sample_rate != target_sample_rate:
        import librosa

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=target_sample_rate,
        )
        sample_rate = target_sample_rate

    return audio, sample_rate


def convert_audio_to_wav(
    audio_path: str | Path,
    sample_rate: int = 16000,
) -> Path:
    source_path = Path(audio_path)
    if not source_path.exists():
        raise RuntimeError(f"Audio input does not exist: {source_path}")
    if source_path.stat().st_size == 0:
        raise RuntimeError("Audio input is empty. Record or upload the audio again.")
    if source_path.suffix.lower() in {".wav", ".wave"} and can_read_audio_info(
        source_path
    ):
        return source_path

    fd, output_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    output_path = Path(output_name)

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        str(output_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(FFMPEG_MISSING_HINT) from exc
    except subprocess.CalledProcessError as exc:
        output_path.unlink(missing_ok=True)
        output = "\n".join(
            part.strip()
            for part in [exc.stderr, exc.stdout]
            if part and part.strip()
        )
        detail = output or (
            f"ffmpeg exited with code {exc.returncode}; "
            f"input={source_path}; input_size={source_path.stat().st_size} bytes; "
            f"ffmpeg={shutil.which('ffmpeg') or 'not found on PATH'}"
        )
        raise RuntimeError(f"Could not convert audio file with FFmpeg: {detail}") from exc

    return output_path


def get_audio_duration_seconds(audio_path: str | Path) -> float:
    path = str(audio_path)

    try:
        import soundfile as sf

        info = sf.info(path)
        return float(info.frames / info.samplerate)
    except Exception:
        pass

    try:
        import librosa

        audio, sample_rate = librosa.load(path, sr=None, mono=True)
        return float(len(audio) / sample_rate)
    except Exception as exc:
        raise RuntimeError(f"Could not read audio duration from {path}") from exc
