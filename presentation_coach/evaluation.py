from __future__ import annotations

from presentation_coach.analysis import normalize_text


def levenshtein_distance(reference_units: list[str], hypothesis_units: list[str]) -> int:
    if len(reference_units) < len(hypothesis_units):
        reference_units, hypothesis_units = hypothesis_units, reference_units

    previous_row = list(range(len(hypothesis_units) + 1))
    for i, reference_unit in enumerate(reference_units, start=1):
        current_row = [i]
        for j, hypothesis_unit in enumerate(hypothesis_units, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (reference_unit != hypothesis_unit)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row

    return previous_row[-1]


def error_rate(reference_units: list[str], hypothesis_units: list[str]) -> float:
    if not reference_units:
        return 0.0 if not hypothesis_units else 1.0
    return levenshtein_distance(reference_units, hypothesis_units) / len(reference_units)


def wer(reference: str, hypothesis: str) -> float:
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    return error_rate(reference_words, hypothesis_words)


def cer(reference: str, hypothesis: str) -> float:
    reference_chars = list(normalize_text(reference).replace(" ", ""))
    hypothesis_chars = list(normalize_text(hypothesis).replace(" ", ""))
    return error_rate(reference_chars, hypothesis_chars)

