import unittest
import numpy as np
import os.path
import collections

from opengender.dame_sexmachine import DameSexmachine
from opengender.paths import DATA_DIR, ALL_PATH


collections.Callable = collections.abc.Callable


class TddInPythonExample(unittest.TestCase):
    def test_dame_sexmachine_features_int(self):
        s = DameSexmachine()
        f = s.features_int("David")
        self.assertTrue(len(f) > 0)

    def test_dame_sexmachine_guess(self):
        s = DameSexmachine()
        self.assertEqual(s.guess("David"), "male")
        self.assertEqual(s.guess("Laura"), "female")
        self.assertEqual(s.guess("David", binary=True), 1)
        self.assertEqual(s.guess("Laura", binary=True), 0)
        self.assertEqual(s.guess("David", binary=True, ml="svc"), 1)
        self.assertEqual(s.guess("Laura", binary=True, ml="svc"), 0)
        self.assertEqual(s.guess("Palabra", binary=True, ml="svc"), 1)
        self.assertEqual(s.guess("Nadiccionaria", binary=True), 0)
        self.assertEqual(s.guess("Nadiccionaria"), "female")
        #        With accents:
        self.assertEqual(s.guess("Inés"), "female")
        #        Without accents:
        self.assertEqual(s.guess("Ines"), "female")

    def test_sexmachine_features_int(self):
        s = DameSexmachine()
        dicc = s.features_int("David")
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

    def test_sexmachine_features_list(self):
        s = DameSexmachine()
        fl = s.features_list()
        self.assertTrue(len(fl) > 20)

    def test_sexmachine_features_list_all(self):
        s = DameSexmachine()
        fl = s.features_list(path=ALL_PATH)
        self.assertTrue(len(fl) > 1000)

    def test_sexmachine_forest(self):
        self.assertTrue(os.path.isfile(DATA_DIR / "forest_model.sav"))

    def test_sexmachine_forest_load(self):
        s = DameSexmachine()
        m = s.forest_load()
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
        s = DameSexmachine()
        m = s.svc_load()
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
