import unittest
import collections

from opengender.dame_utils import DameUtils


collections.Callable = collections.abc.Callable


class TddInPythonExample(unittest.TestCase):
    def test_drop_dots(self):
        u = DameUtils()
        self.assertEqual(1212, int(u.drop_dots(12.12)))

    def test_force_whitespaces(self):
        u = DameUtils()
        self.assertEqual("JUAN DAVID", u.force_whitespaces("JUAN-DAVID"))
        self.assertEqual("JUAN DAVID", u.force_whitespaces("JUAN_DAVID"))

    def test_drop_accents(self):
        u = DameUtils()
        self.assertEqual("Ines", u.drop_accents("Inés"))

    def test_dicc_dataset(self):
        du = DameUtils()
        dicc = du.dicc_dataset("male")
        self.assertEqual(dicc["at"], "files/names/names_at/atmales.csv")
        dicc = du.dicc_dataset("female")
        self.assertEqual(dicc["at"], "files/names/names_at/atfemales.csv")
