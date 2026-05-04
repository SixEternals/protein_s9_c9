import unittest

from models.deepfocus import _r9_feature_vector
from utils.sequence import encode_r9_matrix, region_bits


def _seq_with_replacements(replacements):
    seq = ["A"] * 23
    for pos_1_based, value in replacements.items():
        seq[pos_1_based - 1] = value
    return "".join(seq)


class R9EncoderRuleTests(unittest.TestCase):
    def test_region_bits_follow_deepfocus_paper_mapping(self):
        self.assertEqual(region_bits(1), (0, 1))
        self.assertEqual(region_bits(15), (0, 1))
        self.assertEqual(region_bits(16), (1, 0))
        self.assertEqual(region_bits(20), (1, 0))
        self.assertEqual(region_bits(21), (0, 0))
        self.assertEqual(region_bits(23), (0, 0))

    def test_deepfocus_feature_vector_counts_paper_region_codes(self):
        matrix = encode_r9_matrix("A" * 23, _seq_with_replacements({5: "T", 18: "T", 22: "T"}))
        features = _r9_feature_vector(matrix)

        self.assertEqual(matrix[4][7:9], [0, 1])
        self.assertEqual(matrix[17][7:9], [1, 0])
        self.assertEqual(matrix[21][7:9], [0, 0])
        self.assertAlmostEqual(features[2], 1 / 23, places=6)
        self.assertAlmostEqual(features[3], 1 / 23, places=6)
        self.assertAlmostEqual(features[4], 1 / 23, places=6)


if __name__ == "__main__":
    unittest.main()
