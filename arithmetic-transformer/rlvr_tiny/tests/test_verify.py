import unittest

from rlvr_tiny.verify import check_local_steps, eval_expr, parse_trace, score_trace


class VerifyTests(unittest.TestCase):
    def test_parse_valid(self):
        parts, ok, err = parse_trace("22+9=20+2+9=20+11=31")
        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertEqual(parts[-1], "31")

    def test_wrong_final(self):
        scored = score_trace("22+9", "20+2+9=20+11=32", fmt="B")
        self.assertFalse(scored["final_ok"])

    def test_wrong_middle_step(self):
        scored = score_trace("22+9", "20+2+9=20+10=31", fmt="B")
        self.assertLess(scored["valid_step_fraction"], 1.0)

    def test_malformed_token_order(self):
        parts, ok, err = parse_trace("22++9=31")
        self.assertTrue(ok)
        self.assertIsNone(err)
        with self.assertRaises(ValueError):
            eval_expr("22++9")

    def test_empty_segment(self):
        parts, ok, err = parse_trace("22+9==31")
        self.assertFalse(ok)
        self.assertEqual(err, "empty_segment")

    def test_doubled_operators(self):
        with self.assertRaises(ValueError):
            eval_expr("20++11")

    def test_leading_separator(self):
        with self.assertRaises(ValueError):
            eval_expr("+20+11")

    def test_trailing_separator(self):
        with self.assertRaises(ValueError):
            eval_expr("20+11+")

    def test_adversarial_parseable_case(self):
        scored = score_trace("58+67", "40+18+7=65", fmt="C")
        self.assertTrue(scored["parse_ok"])
        self.assertFalse(scored["final_ok"])


if __name__ == "__main__":
    unittest.main()
