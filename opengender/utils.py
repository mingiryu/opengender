import unicodedata

from opengender.paths import INTERALL_PATH


def dicc_dataset():
    return {"inter": INTERALL_PATH}

def drop_dots(s):
    # given s removes dots symbols in the string
    aux = ""
    for c in unicodedata.normalize("NFD", str(s)):
        if c != ".":
            aux = aux + c
    return aux

def force_whitespaces(s):
    # replace underscore, hyphens, ... by white spaces
    aux = ""
    for c in unicodedata.normalize("NFD", str(s)):
        if (c == "_") or (c == "-"):
            aux = aux + " "
        else:
            aux = aux + c
    return aux

def drop_accents(s):
    # given a string s delete accents
    return "".join(
        (
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )
    )
