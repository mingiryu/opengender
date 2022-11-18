import csv
import unidecode
import configparser


csv.field_size_limit(3000000)


class Gender(object):
    # That's the root class in the heritage,
    # apis classes and sexmachine is a gender
    def __init__(self):
        self.config = configparser.RawConfigParser()
        self.config.read("config.cfg")
        self.males = 0
        self.females = 0
        self.unknown = 0

    # FEATURES METHODS #

    def features_int(self, name):
        # features method created to check the scikit classifiers
        features_int = {}
        features_int["first_letter"] = ord(name[0].lower())
        features_int["last_letter"] = ord(name[-1].lower())
        for letter in "abcdefghijklmnopqrstuvwxyz":
            num = name.lower().count(letter)
            features_int["count({})".format(letter)] = num
        features_int["vocals"] = 0
        for letter1 in "aeiou":
            for letter2 in name:
                if letter1 == letter2:
                    features_int["vocals"] = features_int["vocals"] + 1
        features_int["consonants"] = 0
        for letter1 in "bcdfghjklmnpqrstvwxyz":
            for letter2 in name:
                if letter1 == letter2:
                    features_int["consonants"] = features_int["consonants"] + 1
        # FIRST LETTER
        if name[0].lower() in "aeiou":
            features_int["first_letter_vocal"] = 1
        else:
            features_int["first_letter_vocal"] = 0
        if name[0].lower() in "bcdfghjklmnpqrstvwxyz":
            features_int["first_letter_consonant"] = 1
        else:
            features_int["first_letter_consonant"] = 0
        # LAST LETTER
        if name[-1].lower() in "aeiou":
            features_int["last_letter_vocal"] = 1
        else:
            features_int["last_letter_vocal"] = 0
        if name[-1].lower() in "bcdfghjklmnpqrstvwxyz":
            features_int["last_letter_consonant"] = 1
        else:
            features_int["last_letter_consonant"] = 0
        # h = hyphen.Hyphenator('en_US')
        # features_int["syllables"] = len(h.syllables(name))
        if name[-1].lower() == "a":
            features_int["last_letter_a"] = 1
        else:
            features_int["last_letter_a"] = 0
        if name[-1].lower() == "o":
            features_int["last_letter_o"] = 1
        else:
            features_int["last_letter_o"] = 0
        return features_int

    def features_list(self, path="files/names/partial.csv", sexdataset=""):
        flist = []
        with open(path) as csvfile:
            sexreader = csv.reader(csvfile, delimiter=",", quotechar="|")
            next(sexreader, None)
            for row in sexreader:
                name = row[0].title()
                name = name.replace('"', "")
                flist.append(list(self.features_int(name).values()))
        return flist

    # DATASETS METHODS #

    def name_frec(self, name, *args, **kwargs):
        # guess list method
        dataset = kwargs.get("dataset", "us")
        force_whitespaces = kwargs.get("force_whitespaces", False)
        du = DameUtils()
        name = du.drop_accents(name)
        if force_whitespaces:
            name = du.force_whitespaces(name)
        dicc_males = du.dicc_dataset("male")
        path_males = dicc_males[dataset]
        file_males = open(path_males, "r")
        readerm = csv.reader(file_males, delimiter=",", quotechar="|")
        males = 0
        for row in readerm:
            if (len(row) > 1) and (row[0].lower() == name.lower()):
                males = row[1]
                males = du.drop_dots(males)

        dicc_females = du.dicc_dataset("female")
        path_females = dicc_females[dataset]
        file_females = open(path_females, "r")
        readerf = csv.reader(file_females, delimiter=",", quotechar="|")
        females = 0
        for row in readerf:
            if (len(row) > 1) and (row[0].lower() == name.lower()):
                females = row[1]
                females = du.drop_dots(females)
        dicc = {"females": females, "males": males}

        return dicc

    # GUESS #

    def guess(self, name, binary=False, dataset="us", *args, **kwargs):
        # guess method to check names dictionary
        nonamerange = kwargs.get("nonamerange", 0)
        force_whitespaces = kwargs.get("force_whitespaces", False)
        guess = ""
        name = unidecode.unidecode(name).title()
        name.replace(name, "")
        dicc = self.name_frec(
            name, dataset=dataset, force_whitespaces=force_whitespaces
        )
        m = int(dicc["males"])
        f = int(dicc["females"])
        # nonamerange must be greater than 500
        # otherwise where are considering that
        # name is a nick
        if (m > nonamerange) or (f > nonamerange):
            if (m == 0) and (f == 0):
                if binary:
                    guess = 2
                else:
                    guess = "unknown"
            elif m > f:
                if binary:
                    guess = 1
                else:
                    guess = "male"
            elif f > m:
                if binary:
                    guess = 0
                else:
                    guess = "female"
            else:
                if binary:
                    guess = 2
                else:
                    guess = "unknown"
        else:
            if binary:
                guess = 2
            else:
                guess = "unknown"
        return guess

    def csv2gender_list(self, path, *args, **kwargs):
        # generating a list of 0, 1, 2 as females, males and unknows
        # TODO: ISO/IEC 5218 proposes a norm about coding gender:
        # ``0 as not know'',``1 as male'', ``2 as female''
        # and ``9 as not applicable''
        header = kwargs.get("header", True)
        gender_column = kwargs.get("gender_column", 4)
        gender_f_chars = kwargs.get("gender_f_chars", "f")
        gender_m_chars = kwargs.get("gender_m_chars", "m")
        delimiter = kwargs.get("delimiter", ",")
        glist = []
        with open(path) as csvfile:
            sexreader = csv.reader(csvfile, delimiter=delimiter, quotechar='"')
            if header:
                next(sexreader, None)
            count_females = 0
            count_males = 0
            count_unknown = 0
            gender = ""
            for row in sexreader:
                try:
                    gender = row[gender_column]
                except IndexError:
                    print(
                        "The method csv2gender_list has not row[%s]"
                        % str(gender_column)
                    )
                    print("To review that gender row is set in the input")
                    # os.kill(os.getpid(), signal.SIGUSR1)
                if gender == gender_f_chars:
                    g = 0
                    count_females = count_females + 1
                elif gender == gender_m_chars:
                    g = 1
                    count_males = count_males + 1
                else:
                    g = 2
                    count_unknown = count_unknown + 1
                glist.append(g)
        self.females = count_females
        self.males = count_males
        self.unknown = count_unknown
        return glist
