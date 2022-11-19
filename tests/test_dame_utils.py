import unittest

from opengender.dame_utils import drop_accents, drop_dots, force_whitespaces


class TddInPythonExample(unittest.TestCase):
    def test_drop_dots(self):
        self.assertEqual(1212, int(drop_dots(12.12)))

    def test_force_whitespaces(self):
        self.assertEqual("JUAN DAVID", force_whitespaces("JUAN-DAVID"))
        self.assertEqual("JUAN DAVID", force_whitespaces("JUAN_DAVID"))

    def test_drop_accents(self):
        self.assertEqual("Ines", drop_accents("Inés"))
