import unittest

from utils.sequence import encode_c9_matrix


def _seq_with_replacements(replacements):
    seq = ["A"] * 23
    for pos_1_based, value in replacements.items():
        seq[pos_1_based - 1] = value
    return "".join(seq)


class C9EncoderRuleTests(unittest.TestCase):
    def test_no_mismatch_has_zero_continuous_state(self):
        matrix = encode_c9_matrix("A" * 23, "A" * 23)

        self.assertTrue(all(row[5:7] == [0, 0] for row in matrix))
        self.assertTrue(all(row[7:9] == [0, 0] for row in matrix))

    def test_singleton_mismatch_is_01_only_at_that_position(self):
        matrix = encode_c9_matrix("A" * 23, _seq_with_replacements({5: "T"}))

        self.assertEqual(matrix[4][5:7], [1, 0])
        self.assertEqual(matrix[4][7:9], [0, 1])
        self.assertEqual(matrix[3][7:9], [0, 0])
        self.assertEqual(matrix[5][7:9], [0, 0])

    def test_two_base_mismatch_run_marks_both_positions_as_10(self):
        matrix = encode_c9_matrix("A" * 23, _seq_with_replacements({5: "T", 6: "T"}))

        self.assertEqual(matrix[4][7:9], [1, 0])
        self.assertEqual(matrix[5][7:9], [1, 0])
        self.assertEqual(matrix[3][7:9], [0, 0])
        self.assertEqual(matrix[6][7:9], [0, 0])

    def test_three_or_more_mismatch_run_marks_all_run_positions_as_11(self):
        matrix = encode_c9_matrix("A" * 23, _seq_with_replacements({1: "T", 2: "T", 3: "T", 4: "T"}))

        self.assertEqual([row[7:9] for row in matrix[:4]], [[1, 1], [1, 1], [1, 1], [1, 1]])
        self.assertEqual(matrix[4][7:9], [0, 0])

    def test_separated_runs_are_labeled_independently(self):
        matrix = encode_c9_matrix(
            "A" * 23,
            _seq_with_replacements({2: "T", 5: "T", 6: "T", 9: "T", 10: "T", 11: "T"}),
        )

        expected = [[0, 0] for _ in range(23)]
        expected[1] = [0, 1]
        expected[4] = [1, 0]
        expected[5] = [1, 0]
        expected[8] = [1, 1]
        expected[9] = [1, 1]
        expected[10] = [1, 1]
        self.assertEqual([row[7:9] for row in matrix], expected)

    def test_indels_keep_event_bits_but_do_not_create_continuous_mismatch_state(self):
        deletion = encode_c9_matrix("A" * 23, _seq_with_replacements({4: "-"}))
        self.assertEqual(deletion[3][5:7], [1, 1])
        self.assertEqual(deletion[3][7:9], [0, 0])

        insertion = encode_c9_matrix(_seq_with_replacements({7: "-"}), _seq_with_replacements({7: "T"}))
        self.assertEqual(insertion[6][5:7], [0, 1])
        self.assertEqual(insertion[6][7:9], [0, 0])


if __name__ == "__main__":
    unittest.main()
