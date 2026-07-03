import unittest

from presentation_coach.analysis import SpeechMetrics, build_speech_metrics
from presentation_coach.report import build_feedback


class ReportTest(unittest.TestCase):
    def test_feedback_includes_coaching_sections(self):
        metrics = build_speech_metrics(
            "xin chao cac ban hom nay em trinh bay ve asr",
            duration_seconds=5,
        )

        feedback = build_feedback(metrics)
        joined_feedback = "\n".join(feedback)

        self.assertIn("Tổng quan:", joined_feedback)
        self.assertIn("Nhịp nói:", joined_feedback)
        self.assertIn("Filler words:", joined_feedback)
        self.assertIn("Bài tập tiếp theo:", joined_feedback)

    def test_feedback_mentions_top_filler_when_filler_rate_is_high(self):
        metrics = SpeechMetrics(
            duration_seconds=60,
            word_count=140,
            wpm=140,
            filler_counts={"um": 5, "ah": 2},
            filler_total=7,
            filler_per_minute=7,
            pace_label="on dinh",
        )

        feedback = build_feedback(metrics)

        self.assertTrue(any("'um'" in item for item in feedback))


if __name__ == "__main__":
    unittest.main()
