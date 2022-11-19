import csv
import pickle

from opengender.paths import DATA_DIR, ALL_PATH


csv.field_size_limit(3000000)


def features_int(name):
    # features method created to check the scikit classifiers
    features = {}
    features["first_letter"] = ord(name[0].lower())
    features["last_letter"] = ord(name[-1].lower())
    for letter in "abcdefghijklmnopqrstuvwxyz":
        n = name.lower().count(letter)
        features["count({})".format(letter)] = n
    features["vocals"] = 0
    for letter in "aeiou":
        features["vocals"] = features["vocals"] + 1
    features["consonants"] = 0
    for letter in "bcdfghjklmnpqrstvwxyz":
        features["consonants"] = features["consonants"] + 1
    if chr(features["first_letter"]) in "aeiou":
        features["first_letter_vocal"] = 1
    else:
        features["first_letter_vocal"] = 0
    if chr(features["last_letter"]) in "aeiou":
        features["last_letter_vocal"] = 1
    else:
        features["last_letter_vocal"] = 0
    # h = hyphen.Hyphenator('en_US')
    # features["syllables"] = len(h.syllables(name))
    if ord(name[-1].lower()) == "a":
        features["last_letter_a"] = 1
    else:
        features["last_letter_a"] = 0
    return features


def features_list(path=ALL_PATH):
    flist = []
    with open(path) as csvfile:
        sexreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        next(sexreader, None)
        for row in sexreader:
            name = row[0].title()
            name = name.replace('"', "")
            flist.append(list(features_int(name).values()))
    return flist


def csv2gender_list(path):
    # generating a list of 0, 1, 2 as females, males and unknows
    # TODO: ISO/IEC 5218 proposes a norm about coding gender:
    # ``0 as not know'',``1 as male'', ``2 as female''
    # and ``9 as not applicable''
    gender_column = 4
    gender_f_chars = "f"
    gender_m_chars = "m"
    glist = []
    with open(path) as csvfile:
        sexreader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(sexreader, None)
        gender = ""
        for row in sexreader:
            try:
                gender = row[gender_column]
            except IndexError:
                print("The method csv2gender_list has not row[%s]" % str(gender_column))
                print("To review that gender row is set in the input")
                # os.kill(os.getpid(), signal.SIGUSR1)
            if gender == gender_f_chars:
                g = 0
            elif gender == gender_m_chars:
                g = 1
            else:
                g = 2
            glist.append(g)

    return glist


def svc_load():
    pkl_file = open(DATA_DIR / "svc_model.sav", "rb")
    clf = pickle.load(pkl_file)
    pkl_file.close()
    return clf


def forest_load():
    pkl_file = open(DATA_DIR / "forest_model.sav", "rb")
    clf = pickle.load(pkl_file)
    pkl_file.close()
    return clf


def guess(name, binary=False, ml="svc"):
    # guess method to check names dictionary and nltk classifier
    # TODO: ISO/IEC 5218 proposes a norm about coding gender:
    # ``0 as not know'',``1 as male'', ``2 as female''
    # and ``9 as not applicable''
    guess = 2

    vector = features_int(name)
    if (guess == "unknown") | (guess == 2):
        vector = list(features_int(name).values())
        if ml == "svc":
            m = svc_load()
            predicted = m.predict([vector])
            guess = predicted[0]
        elif ml == "forest":
            m = forest_load()
            predicted = m.predict([vector])
            guess = predicted[0]

        if binary:
            if guess == "female":
                guess = 0
            elif guess == "male":
                guess = 1
            elif guess == "unkwnon":
                guess = 2
        else:
            if guess == 0:
                guess = "female"
            elif guess == 1:
                guess = "male"
            elif guess == 2:
                guess = "unknown"
    return guess
