import unittest
import collections

from opengender.dame_gender import Gender


collections.Callable = collections.abc.Callable


class TddInPythonExample(unittest.TestCase):
    def test_dame_gender_features(self):
        g = Gender()
        f = g.features("David")
        self.assertEqual(f["has(a)"], True)
        self.assertEqual(f["count(i)"], 1)
        self.assertEqual(f["count(v)"], 1)
        self.assertEqual(f["last_letter"], "d")
        self.assertEqual(f["first_letter"], "d")

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

    def test_dame_gender_guess(self):
        g = Gender()
        r = g.guess(name="David", binary=True, dataset="ine")
        self.assertEqual(r, 1)
        r = g.guess(name="Andrea", binary=True)
        self.assertEqual(r, 0)
        r = g.guess(name="David", binary=False)
        self.assertEqual(r, "male")
        r = g.guess(name="Laura", binary=True)
        self.assertEqual(r, 0)
        r = g.guess(name="Laura", binary=False)
        self.assertEqual(r, "female")
        r = g.guess(name="Andrea", binary=True)
        self.assertEqual(r, 0)
        r = g.guess(name="ANA-MARIA", binary=True, dataset="inter")
        self.assertEqual(r, 0)
        r = g.guess(
            name="ANA-MARIA", binary=True, dataset="inter", force_whitespaces=True
        )
        self.assertEqual(r, 0)

    def test_dame_gender_csv2gender_list(self):
        g = Gender()
        gl = g.csv2gender_list(path="files/names/partial.csv")
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

    def test_dame_gender_name_frec(self):
        g = Gender()
        frec1 = g.name_frec("INES", dataset="ine")
        self.assertEqual(int(frec1["females"]), 63378)
        self.assertEqual(int(frec1["males"]), 0)
        frec2 = g.name_frec("BEATRIZ", dataset="ine")
        self.assertEqual(int(frec2["females"]), 122917)
        frec3 = g.name_frec("ALMUDENA", dataset="ine")
        self.assertEqual(int(frec3["females"]), 30517)
        frec5 = g.name_frec("ELIZABETH", dataset="us")
        self.assertEqual(int(frec5["females"]), 1655053)
        frec5n = g.name_frec("ELISABETH", dataset="us")
        self.assertEqual(int(frec5n["females"]), 46811)
        frec6 = g.name_frec("MARIA", dataset="gb")
        self.assertEqual(int(frec6["females"]), 10401)
        frec7 = g.name_frec("JULIAN", dataset="gb")
        self.assertEqual(int(frec7["males"]), 1713)
        frec8 = g.name_frec("A", dataset="gb")
        self.assertEqual(int(frec8["males"]), 42)
        frec6 = g.name_frec("MARIA", dataset="nz")
        self.assertEqual(int(frec6["females"]), 5541)
        frec6 = g.name_frec("MARIA", dataset="ca")
        self.assertEqual(int(frec6["females"]), 1725)
        frec7 = g.name_frec("MARIA", dataset="si")
        self.assertEqual(int(frec7["females"]), 2867)
        frec37 = g.name_frec("ANA-MARIA", dataset="inter")
        self.assertEqual(int(frec37["females"]), 4130)
        self.assertEqual(int(frec37["males"]), 0)
        frec38 = g.name_frec("ANA-MARIA", force_whitespaces=True, dataset="inter")
        self.assertEqual(int(frec38["females"]), 277337)
        self.assertEqual(int(frec38["males"]), 5)
