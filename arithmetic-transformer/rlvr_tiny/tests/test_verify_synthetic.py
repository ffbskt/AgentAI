import unittest

from rlvr_tiny.tests.synthetic_outputs import (
    CORNER_CASE_CORRECT_TRACE,
    CORNER_CASE_INCORRECT_TRACE,
    CORNER_CASE_PROBLEM,
)
from rlvr_tiny.verify import (
    RewardConfig,
    check_local_steps,
    eval_expr,
    extract_final_answer,
    parse_trace,
    score_trace,
)


class VerifySyntheticOutputsTests(unittest.TestCase):
    def setUp(self):
        self.correct_parts, self.correct_ok, self.correct_error = parse_trace(
            CORNER_CASE_CORRECT_TRACE
        )
        self.incorrect_parts, self.incorrect_ok, self.incorrect_error = parse_trace(
            CORNER_CASE_INCORRECT_TRACE
        )

    def test_parse_trace_on_corner_cases(self):
        self.assertTrue(self.correct_ok)
        self.assertIsNone(self.correct_error)
        self.assertEqual(
            self.correct_parts, ["99+1", "90+9+1", "90+10", "100"]
        )

        self.assertTrue(self.incorrect_ok)
        self.assertIsNone(self.incorrect_error)
        self.assertEqual(
            self.incorrect_parts, ["99+1", "90+9+1", "90+11", "101"]
        )

    def test_eval_expr_on_parts(self):
        self.assertEqual(eval_expr(self.correct_parts[0]), 100)
        self.assertEqual(eval_expr(self.correct_parts[1]), 100)
        self.assertEqual(eval_expr(self.correct_parts[2]), 100)
        self.assertEqual(eval_expr(self.correct_parts[3]), 100)

        self.assertEqual(eval_expr(self.incorrect_parts[0]), 100)
        self.assertEqual(eval_expr(self.incorrect_parts[1]), 100)
        self.assertEqual(eval_expr(self.incorrect_parts[2]), 101)
        self.assertEqual(eval_expr(self.incorrect_parts[3]), 101)

    def test_check_local_steps(self):
        correct_flags, correct_fraction = check_local_steps(self.correct_parts, fmt="B")
        incorrect_flags, incorrect_fraction = check_local_steps(
            self.incorrect_parts, fmt="B"
        )

        self.assertEqual(correct_flags, [True, True, True])
        self.assertEqual(correct_fraction, 1.0)

        self.assertEqual(incorrect_flags, [True, False, True])
        self.assertAlmostEqual(incorrect_fraction, 2.0 / 3.0)

    def test_extract_final_answer(self):
        self.assertEqual(extract_final_answer(self.correct_parts), "100")
        self.assertEqual(extract_final_answer(self.incorrect_parts), "101")

    def test_score_trace_correct_output(self):
        result = score_trace(
            CORNER_CASE_PROBLEM,
            CORNER_CASE_CORRECT_TRACE,
            reward_cfg=RewardConfig(),
            fmt="B",
        )
        self.assertTrue(result["parse_ok"])
        self.assertTrue(result["final_ok"])
        self.assertEqual(result["num_steps"], 3)
        self.assertEqual(result["valid_step_fraction"], 1.0)
        self.assertEqual(result["step_valid_flags"], [True, True, True])
        self.assertTrue(result["exact_trace_correct"])
        self.assertEqual(result["final_answer"], "100")

    def test_score_trace_incorrect_output(self):
        result = score_trace(
            CORNER_CASE_PROBLEM,
            CORNER_CASE_INCORRECT_TRACE,
            reward_cfg=RewardConfig(),
            fmt="B",
        )
        self.assertTrue(result["parse_ok"])
        self.assertFalse(result["final_ok"])
        self.assertEqual(result["num_steps"], 3)
        self.assertEqual(result["step_valid_flags"], [True, False, True])
        self.assertAlmostEqual(result["valid_step_fraction"], 2.0 / 3.0)
        self.assertFalse(result["exact_trace_correct"])
        self.assertEqual(result["final_answer"], "101")


if __name__ == "__main__":
    unittest.main()
