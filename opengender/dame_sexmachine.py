import csv
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification

from sklearn import svm

from opengender.dame_gender import Gender


class DameSexmachine(Gender):
    def __init__(self):
        self.males = 0
        self.females = 0
        self.unknown = 0

    def features(self, name):
        # features method created to check the nltk classifier
        features = {}
        features["first_letter"] = name[0].lower()
        features["last_letter"] = name[-1].lower()
        for letter in "abcdefghijklmnopqrstuvwxyz":
            features["count({})".format(letter)] = name.lower().count(letter)
            features["has({})".format(letter)] = letter in name.lower()
        return features

    def features_int(self, name):
        # features method created to check the scikit classifiers
        features_int = {}
        features_int["first_letter"] = ord(name[0].lower())
        features_int["last_letter"] = ord(name[-1].lower())
        for letter in "abcdefghijklmnopqrstuvwxyz":
            n = name.lower().count(letter)
            features_int["count({})".format(letter)] = n
        features_int["vocals"] = 0
        for letter in "aeiou":
            features_int["vocals"] = features_int["vocals"] + 1
        features_int["consonants"] = 0
        for letter in "bcdfghjklmnpqrstvwxyz":
            features_int["consonants"] = features_int["consonants"] + 1
        if chr(features_int["first_letter"]) in "aeiou":
            features_int["first_letter_vocal"] = 1
        else:
            features_int["first_letter_vocal"] = 0
        if chr(features_int["last_letter"]) in "aeiou":
            features_int["last_letter_vocal"] = 1
        else:
            features_int["last_letter_vocal"] = 0
        # h = hyphen.Hyphenator('en_US')
        # features_int["syllables"] = len(h.syllables(name))
        if ord(name[-1].lower()) == "a":
            features_int["last_letter_a"] = 1
        else:
            features_int["last_letter_a"] = 0
        return features_int

    def svc(self):
        # Scikit svc classifier
        X = np.array(self.features_list(path="files/names/all.csv"))
        y = self.csv2gender_list(path="files/names/all.csv")
        clf = svm.SVC()
        clf.fit(X, y)
        filename = "files/datamodels/svc_model.sav"
        pickle.dump(clf, open(filename, "wb"))
        return clf

    def svc_load(self):
        pkl_file = open("files/datamodels/svc_model.sav", "rb")
        clf = pickle.load(pkl_file)
        pkl_file.close()
        return clf

    def forest(self):
        # Scikit forest classifier
        X = np.array(self.features_list(path="files/names/all.csv"))
        y = np.array(self.csv2gender_list(path="files/names/all.csv"))
        X, y = make_classification(
            n_samples=7000,
            n_features=33,
            n_informative=33,
            n_redundant=0,
            random_state=0,
            shuffle=False,
        )
        rf = RandomForestRegressor(n_estimators=20, random_state=0)
        rf.fit(X, y)
        filename = "files/datamodels/forest_model.sav"
        pickle.dump(rf, open(filename, "wb"))
        return rf

    def forest_load(self):
        pkl_file = open("files/datamodels/forest_model.sav", "rb")
        clf = pickle.load(pkl_file)
        pkl_file.close()
        return clf

    def guess(self, name, binary=False, ml="svc", *args, **kwargs):
        # guess method to check names dictionary and nltk classifier
        # TODO: ISO/IEC 5218 proposes a norm about coding gender:
        # ``0 as not know'',``1 as male'', ``2 as female''
        # and ``9 as not applicable''
        dataset = kwargs.get("dataset", "us")
        guess = 2
        guess = super().guess(name, binary, dataset)
        vector = self.features_int(name)
        if (guess == "unknown") | (guess == 2):
            vector = list(self.features_int(name).values())
            if ml == "svc":
                m = self.svc_load()
                predicted = m.predict([vector])
                guess = predicted[0]
            elif ml == "forest":
                m = self.forest_load()
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

    def guess_list(
        self, path="files/names/partial.csv", binary=False, ml="nltk", *args, **kwargs
    ):
        # guess list method
        dataset = kwargs.get("dataset", "us")
        slist = []
        with open(path) as csvfile:
            sexreader = csv.reader(csvfile, delimiter=",", quotechar="|")
            next(sexreader, None)
            for row in sexreader:
                name = row[0].title()
                name = name.replace('"', "")
                slist.append(self.guess(name, binary, ml=ml, dataset=dataset))
        return slist
