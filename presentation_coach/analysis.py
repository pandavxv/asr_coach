from __future__ import annotations

from dataclasses import dataclass
import re


FILLER_PHRASES = (
    "ờ",
    "ừ",
    "ừm",
    "à",
    "thì",
    "kiểu như",
    "kiểu",
    "nói chung",
)


@dataclass(frozen=True)
class SpeechMetrics:
    duration_seconds: float
    word_count: int
    wpm: float
    filler_counts: dict[str, int]
    filler_total: int
    filler_per_minute: float
    pace_label: str


def normalize_text(text: str) -> str:
    lowered = text.lower()
    without_punctuation = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", without_punctuation, flags=re.UNICODE).strip()


def tokenize_words(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


def count_fillers(
    text: str,
    filler_phrases: tuple[str, ...] = FILLER_PHRASES,
) -> dict[str, int]:
    tokens = tokenize_words(text)
    filler_specs = sorted(
        ((phrase, tokenize_words(phrase)) for phrase in filler_phrases),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    counts = {phrase: 0 for phrase in filler_phrases}

    index = 0
    while index < len(tokens):
        matched = False
        for phrase, phrase_tokens in filler_specs:
            if not phrase_tokens:
                continue
            end = index + len(phrase_tokens)
            if tokens[index:end] == phrase_tokens:
                counts[phrase] += 1
                index = end
                matched = True
                break
        if not matched:
            index += 1

    return {phrase: count for phrase, count in counts.items() if count > 0}


def words_per_minute(word_count: int, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    return word_count / (duration_seconds / 60)


def classify_pace(wpm: float) -> str:
    if wpm == 0:
        return "khong du du lieu"
    if wpm < 120:
        return "hoi cham"
    if wpm > 180:
        return "hoi nhanh"
    return "on dinh"


def build_speech_metrics(transcript: str, duration_seconds: float) -> SpeechMetrics:
    words = tokenize_words(transcript)
    filler_counts = count_fillers(transcript)
    filler_total = sum(filler_counts.values())
    wpm = words_per_minute(len(words), duration_seconds)
    filler_per_minute = words_per_minute(filler_total, duration_seconds)

    return SpeechMetrics(
        duration_seconds=duration_seconds,
        word_count=len(words),
        wpm=wpm,
        filler_counts=filler_counts,
        filler_total=filler_total,
        filler_per_minute=filler_per_minute,
        pace_label=classify_pace(wpm),
    )

