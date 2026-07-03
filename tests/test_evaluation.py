import unittest

from presentation_coach.evaluation import cer, wer


class EvaluationTest(unittest.TestCase):
    def test_wer_counts_deleted_word(self):
        self.assertEqual(wer("xin chào các bạn", "xin chào bạn"), 0.25)

    def test_cer_is_zero_for_equal_normalized_text(self):
        self.assertEqual(cer("Xin chào, các bạn!", "xin chào các bạn"), 0)


if __name__ == "__main__":
    unittest.main()

