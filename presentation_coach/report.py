from __future__ import annotations

from presentation_coach.analysis import SpeechMetrics, build_speech_metrics
from presentation_coach.evaluation import cer, wer


def _top_filler(metrics: SpeechMetrics) -> str | None:
    if not metrics.filler_counts:
        return None
    return max(metrics.filler_counts.items(), key=lambda item: item[1])[0]


def build_feedback(metrics: SpeechMetrics, wer_value: float | None = None) -> list[str]:
    feedback: list[str] = []

    if metrics.word_count == 0:
        return [
            "Chưa có transcript để phân tích. Hãy thử record lại một đoạn ngắn 20-30 giây, nói rõ từng ý và để micro gần hơn một chút."
        ]

    feedback.append(
        f"Tổng quan: bạn nói {metrics.word_count} từ trong {metrics.duration_seconds:.1f} giây, tốc độ khoảng {metrics.wpm:.0f} WPM. Mình sẽ tập trung vào nhịp nói, filler words và độ rõ của transcript."
    )

    if metrics.pace_label == "hoi cham":
        feedback.append(
            "Nhịp nói: hơi chậm. Điểm tốt là người nghe có thời gian theo dõi, nhưng nếu kéo dài cả bài thì bài thuyết trình có thể bị mất năng lượng. Lần sau hãy thử tăng nhịp ở các câu giải thích và giữ pause cho những ý quan trọng."
        )
    elif metrics.pace_label == "hoi nhanh":
        feedback.append(
            "Nhịp nói: hơi nhanh. Nói nhanh giúp bài có năng lượng, nhưng người nghe có thể bỏ lỡ ý chính. Hãy chèn pause 1 giây sau mỗi câu chốt và giảm tốc ở các đoạn có thuật ngữ."
        )
    else:
        feedback.append(
            "Nhịp nói: đang ổn định cho một bài thuyết trình. Hãy giữ nhịp này, nhưng chú ý thêm những khoảng dừng ngắn sau các ý chính để câu nói có điểm nhấn hơn."
        )

    if metrics.filler_total == 0:
        feedback.append(
            "Filler words: rất sạch, hệ thống không phát hiện filler word nào. Đây là điểm mạnh vì bài nói nghe gọn và tự tin hơn."
        )
    elif metrics.filler_per_minute <= 3:
        feedback.append(
            f"Filler words: có {metrics.filler_total} lần, khoảng {metrics.filler_per_minute:.1f} lần/phút. Mức này vẫn ổn; lần sau thử thay filler bằng một khoảng dừng rất ngắn để nghe tự nhiên hơn."
        )
    else:
        top_filler = _top_filler(metrics)
        filler_note = f" Từ xuất hiện nhiều nhất là '{top_filler}'." if top_filler else ""
        feedback.append(
            f"Filler words: đang xuất hiện khá nhiều, {metrics.filler_total} lần, khoảng {metrics.filler_per_minute:.1f} lần/phút.{filler_note} Lần tập tiếp theo, chỉ cần tập trung giảm một filler phổ biến nhất trước."
        )

    if wer_value is not None:
        if wer_value <= 0.15:
            feedback.append(
                "Độ rõ transcript: ASR gần với transcript chuẩn, cho thấy audio và cách phát âm đang khá tốt."
            )
        elif wer_value <= 0.35:
            feedback.append(
                "Độ rõ transcript: ASR có sai khác vừa phải so với transcript chuẩn. Hãy xem lại những từ bị nhận sai, thường do nói nhanh, âm cuối không rõ hoặc thuật ngữ tiếng Anh."
            )
        else:
            feedback.append(
                "Độ rõ transcript: ASR sai khác nhiều so với transcript chuẩn. Nên thử record trong môi trường yên tĩnh hơn, nói gần micro hơn và tách câu dài thành các cụm ngắn."
            )
    else:
        feedback.append(
            "Độ rõ transcript: nếu muốn đánh giá chính xác hơn, hãy nhập transcript chuẩn để hệ thống tính WER/CER và chỉ ra mức sai khác của ASR."
        )

    if metrics.pace_label == "hoi nhanh":
        feedback.append(
            "Bài tập tiếp theo: đọc lại cùng đoạn này, mỗi khi hết một ý hãy dừng 1 giây rồi mới nói tiếp."
        )
    elif metrics.filler_per_minute > 3:
        feedback.append(
            "Bài tập tiếp theo: record lại một lần nữa và cố ý thay filler bằng im lặng ngắn. Mục tiêu là giảm filler xuống dưới 3 lần/phút."
        )
    else:
        feedback.append(
            "Bài tập tiếp theo: record thêm một bản 60 giây, giữ nhịp hiện tại và tập nhấn mạnh rõ hơn ở câu mở đầu và câu kết."
        )

    return feedback


def build_report(
    transcript: str,
    duration_seconds: float,
    reference_text: str | None = None,
) -> dict[str, object]:
    metrics = build_speech_metrics(transcript, duration_seconds)
    has_reference = bool(reference_text and reference_text.strip())
    wer_value = wer(reference_text, transcript) if has_reference else None
    cer_value = cer(reference_text, transcript) if has_reference else None

    return {
        "metrics": metrics,
        "wer": wer_value,
        "cer": cer_value,
        "feedback": build_feedback(metrics, wer_value),
    }
