import unittest

from presentation_coach.analysis import build_speech_metrics, count_fillers


class AnalysisTest(unittest.TestCase):
    def test_build_speech_metrics_counts_basic_values(self):
        transcript = "Ờ hôm nay thì em trình bày, nói chung là hơi nhanh."

        metrics = build_speech_metrics(transcript, duration_seconds=60)

        self.assertEqual(metrics.word_count, 12)
        self.assertEqual(metrics.filler_total, 3)
        self.assertEqual(metrics.filler_counts["ờ"], 1)
        self.assertEqual(metrics.filler_counts["thì"], 1)
        self.assertEqual(metrics.filler_counts["nói chung"], 1)
        self.assertEqual(metrics.wpm, 12)

    def test_longer_filler_phrase_is_not_double_counted(self):
        counts = count_fillers("Nói chung là kiểu như vậy.")

        self.assertEqual(counts["nói chung"], 1)
        self.assertEqual(counts["kiểu như"], 1)
        self.assertNotIn("kiểu", counts)


if __name__ == "__main__":
    unittest.main()

