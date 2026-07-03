from pathlib import Path
import tempfile

import streamlit as st

from presentation_coach.asr import (
    DEFAULT_LORA_ADAPTER_DIR,
    DEFAULT_MODEL_ID,
    PHOWHISPER_SMALL_MODEL_ID,
    PHOWHISPER_SMALL_LORA_ADAPTER_DIR,
    has_local_lora_adapter,
    transcribe_audio,
)
from presentation_coach.audio import convert_audio_to_wav, get_audio_duration_seconds
from presentation_coach.report import build_report


st.set_page_config(
    page_title="Vietnamese Presentation ASR Coach",
    layout="wide",
)

st.title("Vietnamese Presentation ASR Coach")

model_options = {
    "PhoWhisper-base": (DEFAULT_MODEL_ID, None),
    "PhoWhisper-final": (PHOWHISPER_SMALL_MODEL_ID, None),
}
if has_local_lora_adapter():
    model_options["PhoWhisper-base + LoRA fine-tuned"] = (
        DEFAULT_MODEL_ID,
        DEFAULT_LORA_ADAPTER_DIR,
    )
if has_local_lora_adapter(PHOWHISPER_SMALL_LORA_ADAPTER_DIR):
    model_options["PhoWhisper-final + LoRA fine-tuned"] = (
        PHOWHISPER_SMALL_MODEL_ID,
        PHOWHISPER_SMALL_LORA_ADAPTER_DIR,
    )
if len(model_options) == 2:
    st.info("Chua thay LoRA adapter local, app se chi dung model pretrained.")

model_mode = st.radio(
    "Che do model",
    list(model_options.keys()),
    horizontal=True,
)
selected_model_id, selected_adapter_dir = model_options[model_mode]
st.caption(f"ASR model: {model_mode}")

input_mode = st.radio(
    "Nguon audio",
    ["Record truc tiep", "Upload file"],
    horizontal=True,
)

recorded_audio = None
uploaded_audio = None
if input_mode == "Record truc tiep":
    recorded_audio = st.audio_input("Record audio")
else:
    uploaded_audio = st.file_uploader(
        "Audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
    )

selected_audio = recorded_audio or uploaded_audio
reference_text = st.text_area(
    "Transcript chuan (tuy chon)",
    height=140,
)


def save_audio_to_temp(audio_file) -> Path:
    suffix = Path(getattr(audio_file, "name", "")).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(audio_file.getbuffer())
        return Path(temp_file.name)


def render_report(transcript: str, report: dict[str, object]) -> None:
    metrics = report["metrics"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Thoi luong", f"{metrics.duration_seconds:.1f}s")
    col2.metric("So tu", metrics.word_count)
    col3.metric("WPM", f"{metrics.wpm:.1f}")
    col4.metric("Filler", metrics.filler_total)

    if report["wer"] is not None and report["cer"] is not None:
        col5, col6 = st.columns(2)
        col5.metric("WER", f"{report['wer'] * 100:.2f}%")
        col6.metric("CER", f"{report['cer'] * 100:.2f}%")

    st.subheader("Transcript")
    st.write(transcript or "_Khong co transcript._")

    st.subheader("Filler words")
    if metrics.filler_counts:
        st.table(
            [
                {"filler": filler, "count": count}
                for filler, count in metrics.filler_counts.items()
            ]
        )
    else:
        st.write("Khong phat hien filler word.")

    st.subheader("Feedback")
    for item in report["feedback"]:
        st.write(f"- {item}")


if st.button("Phan tich", type="primary", disabled=selected_audio is None):
    temp_audio_path = save_audio_to_temp(selected_audio)
    wav_audio_path = None

    try:
        with st.spinner(f"Dang chay {model_mode}..."):
            wav_audio_path = convert_audio_to_wav(temp_audio_path)
            duration_seconds = get_audio_duration_seconds(wav_audio_path)
            transcript = transcribe_audio(
                wav_audio_path,
                model_id=selected_model_id,
                adapter_dir=selected_adapter_dir,
            )
            report = build_report(
                transcript=transcript,
                duration_seconds=duration_seconds,
                reference_text=reference_text,
            )

        render_report(transcript, report)
    finally:
        temp_audio_path.unlink(missing_ok=True)
        if wav_audio_path:
            wav_audio_path.unlink(missing_ok=True)
