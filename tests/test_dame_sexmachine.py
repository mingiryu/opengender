import unittest
import numpy as np

from opengender.dame_sexmachine import (
    features_int,
    guess,
    forest_load,
    svc_load,
    csv2gender_list,
)
from opengender.paths import PARTIAL_PATH


class TddInPythonExample(unittest.TestCase):
    def test_sexmachine_features_int(self):
        dicc = features_int("David")
        self.assertEqual(chr(dicc["last_letter"]), "d")
        self.assertEqual(chr(dicc["first_letter"]), "d")
        self.assertEqual(dicc["count(a)"], 1)
        self.assertEqual(dicc["count(b)"], 0)
        self.assertEqual(dicc["count(c)"], 0)
        self.assertEqual(dicc["count(d)"], 2)
        self.assertEqual(dicc["count(e)"], 0)
        self.assertEqual(dicc["count(f)"], 0)
        self.assertEqual(dicc["count(h)"], 0)
        self.assertEqual(dicc["count(i)"], 1)
        self.assertEqual(dicc["count(v)"], 1)
        self.assertTrue(dicc["count(a)"] > 0)
        self.assertTrue(dicc["vocals"], 2)
        self.assertTrue(dicc["consonants"], 3)
        self.assertEqual(dicc["first_letter_vocal"], 0)
        self.assertEqual(dicc["last_letter_vocal"], 0)
        self.assertTrue(len(dicc.values()) > 30)

    def test_dame_sexmachine_guess(self):
        self.assertEqual(guess("David"), "male")
        self.assertEqual(guess("Laura"), "female")
        self.assertEqual(guess("David", binary=True), 1)
        self.assertEqual(guess("Laura", binary=True), 0)
        self.assertEqual(guess("David", binary=True, ml="svc"), 1)
        self.assertEqual(guess("Laura", binary=True, ml="svc"), 0)
        self.assertEqual(guess("Palabra", binary=True, ml="svc"), 1)
        self.assertEqual(guess("Nadiccionaria", binary=True), 0)
        self.assertEqual(guess("Nadiccionaria"), "female")
        #        With accents:
        self.assertEqual(guess("Inés"), "female")
        #        Without accents:
        self.assertEqual(guess("Ines"), "female")

    def test_dame_gender_csv2gender_list(self):
        gl = csv2gender_list(path=PARTIAL_PATH)
        self.assertEqual(
            gl, [1, 1, 1, 1, 2, 1, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
        )
        self.assertEqual(len(gl), 21)

    def test_sexmachine_forest_load(self):
        m = forest_load()
        predicted = m.predict(
            [
                [
                    0,
                    0,
                    1,
                    0,
                    21,
                    0,
                    0,
                    0,
                    0,
                    34,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    0,
                    0,
                    0,
                    34,
                    1,
                    0,
                    1,
                ]
            ]
        )
        a = np.array([0.65])
        self.assertEqual(predicted, a)

    def test_sexmachine_svc_load(self):
        m = svc_load()
        predicted = m.predict(
            [
                [
                    0,
                    0,
                    1,
                    0,
                    21,
                    0,
                    0,
                    0,
                    0,
                    34,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    0,
                    0,
                    0,
                    34,
                    1,
                    0,
                    1,
                ]
            ]
        )
        n = np.array([1])
        self.assertTrue(np.array_equal(predicted, n))
