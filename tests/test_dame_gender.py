import unittest
import collections

from opengender.dame_gender import Gender
from opengender.paths import PARTIAL_PATH

collections.Callable = collections.abc.Callable


class TddInPythonExample(unittest.TestCase):
    def test_dame_gender_features_int(self):
        g = Gender()
        features_int = g.features_int("David")
        #        self.assertTrue(features_int["first_letter"] == 100)
        self.assertTrue(features_int["last_letter"] == 100)
        self.assertTrue(features_int["vocals"] == 2)
        self.assertTrue(features_int["consonants"] == 2)
        #        self.assertTrue(features_int["first_letter_vocal"] == 0)
        self.assertTrue(features_int["last_letter_vocal"] == 0)
        #        self.assertTrue(features_int["first_letter_consonant"] == 1)
        self.assertTrue(features_int["last_letter_consonant"] == 1)
        self.assertTrue(features_int["last_letter_a"] == 0)
        self.assertEqual(len(features_int), 36)

    def test_dame_gender_csv2gender_list(self):
        g = Gender()
        gl = g.csv2gender_list(path=PARTIAL_PATH)
        self.assertEqual(
            gl, [1, 1, 1, 1, 2, 1, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
        )
        self.assertEqual(len(gl), 21)
        self.assertEqual(g.females, 3)
        self.assertEqual(g.males, 16)
        self.assertEqual(g.unknown, 2)

    def test_dame_gender_features_list(self):
        g = Gender()
        fl = g.features_list()
        self.assertTrue(len(fl) > 20)
