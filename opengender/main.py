import os
import re
import argparse

from opengender.dame_sexmachine import DameSexmachine

parser = argparse.ArgumentParser()
parser.add_argument("name", help="display the gender")
parser.add_argument('--ml', choices=['nltk', 'svc', 'sgd', 'gaussianNB',
                                     'multinomialNB', 'bernoulliNB',
                                     'forest', 'tree', 'mlp'])
parser.add_argument('--total', default="inter",
                    choices=['ar', 'at', 'au', 'be', 'ca', 'ch',
                             'cn', 'de', 'dk', 'es', 'fi',
                             'fr', 'gb', 'ie', 'is', 'it', 'no', 'nz',
                             'mx', 'pt', 'ru', 'ru_ru',
                             'ru_en', 'se', 'si',
                             'uy', 'us', 'namdict', 'inter'])
# More about iso codes on https://www.iso.org/obp/ui/
# You can set alphabet with sufix:
# So russian in latin alphabet would be ru_en
parser.add_argument('--version', action='version', version='0.4')
parser.add_argument('--force_whitespaces', default=False, action="store_true")
parser.add_argument('--verbose', default=False, action="store_true")
args = parser.parse_args()

results = []

s = DameSexmachine()

if (args.total == "namdict"):
    name = args.name.capitalize()
    name = name.replace(" ", "+")
    cmd = 'grep -i " ' + name
    cmd = cmd + ' " files/names/nam_dict.txt > files/logs/grep.tmp'
    os.system(cmd)
    results = []
    for i in open('files/logs/grep.tmp', 'r').readlines():
        if (i not in results):
            results.append(i)
    male = 0
    female = 0
    for i in results:
        regex = "(M|F|=|\\?|1)( |M|F)?( )(" + name + ")( )"
        r = re.match(regex, i)
        if (r is not None):
            prob = r.group(1) + r.group(2)
        else:
            prob = ""
        bool1 = ('F' == prob) or ('F ' == prob)
        bool1 = bool1 or (prob == '?F') or (prob == '1F')
        bool2 = ('M' == prob) or ('M ' == prob)
        bool2 = bool2 or ('?M' == prob) or ('1M' == prob)
        if bool1:
            female = female + 1
        elif bool2:
            male = male + 1
        if (prob != ""):
            if (female > male):
                print("gender: female")
            if (male > female):
                print("gender: male")
            elif (male == female):
                print("gender: unknown")
                print("you can try predict with --ml")

elif ((args.verbose) or (args.total == "all")):
    n_males = s.name_frec(args.name, dataset="ar",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="ar",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Argentina statistics"
          % (n_males, args.name))
    print("%s females for %s from Argentina statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="at",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="at",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Austria statistics"
          % (n_males, args.name))
    print("%s females for %s from Austria statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="au",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="au",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Australia statistics"
          % (n_males, args.name))
    print("%s females for %s from Australia statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="be",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="be",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Belgium statistics"
          % (n_males, args.name))
    print("%s females for %s from Belgium statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="ca",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="ca",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Canada statistics"
          % (n_males, args.name))
    print("%s females for %s from Canada statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="ch",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="ch",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Swiss statistics"
          % (n_males, args.name))
    print("%s females for %s from Swiss statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="cn",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="cn",
                            force_whitespaces=args.force_whitespaces)['females']
    print("Warning: China option only runs with China alphabet")
    print("%s males for %s from China statistics"
          % (n_males, args.name))
    print("%s females for %s from China statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="de",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="de",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Deutchsland statistics"
          % (n_males, args.name))
    print("%s females for %s from Deutchsland statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="dk",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="dk",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Denmark statistics"
          % (n_males, args.name))
    print("%s females for %s from Denmark statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="ine",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="ine",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Spain statistics (INE.es)"
          % (n_males, args.name))
    print("%s females for %s from Spain statistics (INE.es)"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="ie",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="ie",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Ireland statistics"
          % (n_males, args.name))
    print("%s females for %s from Ireland statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="is",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="is",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Iceland statistics"
          % (n_males, args.name))
    print("%s females for %s from Iceland statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="it",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="it",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Italy statistics"
          % (n_males, args.name))
    print("%s females for %s from Italy statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="fi",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="fi",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Finland statistics"
          % (n_males, args.name))
    print("%s females for %s from Finland statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="fr",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="fr",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from France statistics"
          % (n_males, args.name))
    print("%s females for %s from France statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="gb",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="gb",
                            force_whitespaces=args.force_whitespaces)['females']
    gb = "United Kingdom of Great Britain and Northern Ireland"
    print("%s males for %s from %s statistics"
          % (n_males, args.name, gb))
    print("%s females for %s from %s statistics"
          % (n_females, args.name, gb))
    n_males = s.name_frec(args.name, dataset="mx",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="mx",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Mexico statistics"
          % (n_males, args.name))
    print("%s females for %s from Mexico statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="no",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="no",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Norway statistics"
          % (n_males, args.name))
    print("%s females for %s from Norway statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="nz",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="nz",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from New Zealand statistics"
          % (n_males, args.name))
    print("%s females for %s from New Zealand statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="pt",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="pt",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Portugal statistics"
          % (n_males, args.name))
    print("%s females for %s from Portugal statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="ru",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="ru",
                            force_whitespaces=args.force_whitespaces)['females']
    print("Warning: russian option only runs with russian alphabet")
    print("%s males for %s from Russia statistics"
          % (n_males, args.name))
    print("%s females for %s from Russia statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="se",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="se",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Sweden statistics"
          % (n_males, args.name))
    print("%s females for %s from Sweden statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="si",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="si",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Slovenia statistics"
          % (n_males, args.name))
    print("%s females for %s from Slovenia statistics"
          % (n_females, args.name))
    # n_males = s.name_frec(args.name, dataset="tr")['males']
    # n_females = s.name_frec(args.name, dataset="tr")['females']
    # print("%s males for %s from Turkish statistics"
    #       % (n_males, args.name))
    # print("%s females for %s from Turkish statistics"
    #       % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="us",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="us",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from United States of America statistics"
          % (n_males, args.name))
    print("%s females for %s from United States of America statistics"
          % (n_females, args.name))
    n_males = s.name_frec(args.name, dataset="uy",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="uy",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from Uruguay statistics"
          % (n_males, args.name))
    print("%s females for %s from Uruguay statistics"
          % (n_females, args.name))

    n_males = s.name_frec(args.name, dataset="inter",
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset="inter",
                            force_whitespaces=args.force_whitespaces)['females']
    print("%s males for %s from international statistics"
          % (n_males, args.name))
    print("%s females for %s from international statistics"
          % (n_females, args.name))

else:
    s = DameSexmachine()
    n_males = s.name_frec(args.name, dataset=args.total,
                          force_whitespaces=args.force_whitespaces)['males']
    n_females = s.name_frec(args.name, dataset=args.total,
                            force_whitespaces=args.force_whitespaces)['females']
    if (int(n_males) > int(n_females)):
        print("%s's gender is male" % (str(args.name)))
        prob = int(n_males) / (int(n_males) + int(n_females))
        print("probability: %s" % str(prob))
    elif (int(n_males) < int(n_females)):
        print("%s's gender is female" % (str(args.name)))
        prob = int(n_females) / (int(n_females) + int(n_males))
        print("probability: %s" % str(prob))
    elif ((int(n_males) == 0) and (int(n_females) == 0)):
        args.ml = 'nltk'

    if (args.ml):
        if (args.ml == "nltk"):
            guess = s.guess(args.name, binary=True, ml="nltk",
                            force_whitespaces=args.force_whitespaces)
        if (args.ml == "sgd"):
            guess = s.guess(args.name, binary=True, ml="sgd",
                            force_whitespaces=args.force_whitespaces)
        elif (args.ml == "svc"):
            guess = s.guess(args.name, binary=True, ml="svc",
                            force_whitespaces=args.force_whitespaces)
        elif (args.ml == "gaussianNB"):
            guess = s.guess(args.name, binary=True, ml="gaussianNB",
                            force_whitespaces=args.force_whitespaces)
        elif (args.ml == "multinomialNB"):
            guess = s.guess(args.name, binary=True, ml="multinomialNB",
                            force_whitespaces=args.force_whitespaces)
        elif (args.ml == "bernoulliNB"):
            guess = s.guess(args.name, binary=True, ml="bernoulliNB",
                            force_whitespaces=args.force_whitespaces)
        elif (args.ml == "forest"):
            guess = s.guess(args.name, binary=True, ml="forest",
                            force_whitespaces=args.force_whitespaces)
        elif (args.ml == "xgboost"):
            guess = s.guess(args.name, binary=True, ml="xgboost",
                            force_whitespaces=args.force_whitespaces)
        elif (args.ml == "tree"):
            guess = s.guess(args.name, binary=True, ml="tree",
                            force_whitespaces=args.force_whitespaces)
        elif (args.ml == "mlp"):
            guess = s.guess(args.name, binary=True, ml="mlp",
                            force_whitespaces=args.force_whitespaces)
        if (guess == 1):
            sex = "male"
        elif (guess == 0):
            sex = "female"
        elif (guess == 2):
            sex = "unknown"
        print("%s gender predicted is %s" % (str(args.name), sex))

    if (args.total == "ar"):
        print("%s males for %s from Argentina statistics"
              % (n_males, args.name))
        print("%s females for %s from Argentina statistics"
              % (n_females, args.name))
    elif (args.total == "at"):
        print("%s males for %s from Austria statistics"
              % (n_males, args.name))
        print("%s females for %s from Austria statistics"
              % (n_females, args.name))
    elif (args.total == "au"):
        print("%s males for %s from Australia statistics"
              % (n_males, args.name))
        print("%s females for %s from Australia statistics"
              % (n_females, args.name))
    elif (args.total == "be"):
        print("%s males for %s from Belgium statistics"
              % (n_males, args.name))
        print("%s females for %s from Belgium statistics"
              % (n_females, args.name))
    elif (args.total == "ca"):
        print("%s males for %s from Canada statistics"
              % (n_males, args.name))
        print("%s females for %s from Canada statistics"
              % (n_females, args.name))
    elif (args.total == "ch"):
        print("%s males for %s from Swiss statistics"
              % (n_males, args.name))
        print("%s females for %s from Swiss statistics"
              % (n_females, args.name))
    elif (args.total == "cn"):
        print("Warning: total=cn only runs with chinese alphabet")
        print("%s males for %s from China statistics"
              % (n_males, args.name))
        print("%s females for %s from China statistics"
              % (n_females, args.name))
    elif (args.total == "de"):
        print("%s males for %s from Deutchsland statistics"
              % (n_males, args.name))
        print("%s females for %s from Deutchsland statistics"
              % (n_females, args.name))
    elif (args.total == "dk"):
        print("%s males for %s from Denmark statistics"
              % (n_males, args.name))
        print("%s females for %s from Denmark statistics"
              % (n_females, args.name))
    elif (args.total == "ie"):
        print("%s males for %s from Ireland statistics"
              % (n_males, args.name))
        print("%s females for %s from Ireland statistics"
              % (n_females, args.name))
    elif (args.total == "is"):
        print("%s males for %s from Iceland statistics"
              % (n_males, args.name))
        print("%s females for %s from Iceland statistics"
              % (n_females, args.name))
    elif (args.total == "it"):
        print("%s males for %s from Italy statistics"
              % (n_males, args.name))
        print("%s females for %s from Italy statistics"
              % (n_females, args.name))
    elif ((args.total == "ine") or (args.total == "es")):
        print("%s males for %s from Spain statistics (INE.es)"
              % (n_males, args.name))
        print("%s females for %s from Spain statistics (INE.es)"
              % (n_females, args.name))
    elif (args.total == "inter"):
        print("%s males for %s from international statistics"
              % (n_males, args.name))
        print("%s females for %s from international statistics"
              % (n_females, args.name))
    elif (args.total == "fi"):
        print("%s males for %s from Finland statistics"
              % (n_males, args.name))
        print("%s females for %s from Finland statistics"
              % (n_females, args.name))
    elif (args.total == "fr"):
        print("%s males for %s from France statistics"
              % (n_males, args.name))
        print("%s females for %s from France statistics"
              % (n_females, args.name))
    elif (args.total == "gb"):
        gb = "United Kingdom of Great Britain and Northern Ireland statistics"
        print("%s males for %s from %s"
              % (n_males, args.name, gb))
        print("%s females for %s from %s"
              % (n_females, args.name, gb))
    elif (args.total == "mx"):
        print("%s males for %s from Mexico statistics"
              % (n_males, args.name))
        print("%s females for %s from Mexico statistics"
              % (n_females, args.name))
    elif (args.total == "no"):
        print("%s males for %s from Norway statistics"
              % (n_males, args.name))
        print("%s females for %s from Norway statistics"
              % (n_females, args.name))
    elif (args.total == "nz"):
        print("%s males for %s from New Zealand statistics"
              % (n_males, args.name))
        print("%s females for %s from New Zealand statistics"
              % (n_females, args.name))
    elif (args.total == "pt"):
        print("%s males for %s from Portugal statistics"
              % (n_males, args.name))
        print("%s females for %s from Portugal statistics"
              % (n_females, args.name))
    elif ((args.total == "ru") | (args.total == "ru_ru")):
        print("Warning: total=ru only runs with russian alphabet")
        print("If you want latin alphabet, choose ru_en")
        print("%s males for %s from Russia statistics"
              % (n_males, args.name))
        print("%s females for %s from Russia statistics"
              % (n_females, args.name))
    elif (args.total == "ru_en"):
        print("%s males for %s from Russia statistics"
              % (n_males, args.name))
        print("%s females for %s from Russia statistics"
              % (n_females, args.name))
    elif (args.total == "se"):
        print("%s males for %s from Sweden statistics"
              % (n_males, args.name))
        print("%s females for %s from Sweden statistics"
              % (n_females, args.name))
    elif (args.total == "si"):
        print("%s males for %s from Slovenia statistics"
              % (n_males, args.name))
        print("%s females for %s from Slovenia statistics"
              % (n_females, args.name))
    elif (args.total == "tr"):
        print("%s males for %s from Turkish statistics"
              % (n_males, args.name))
        print("%s females for %s from Turkish statistics"
              % (n_females, args.name))
    elif (args.total == "uy"):
        print("%s males for %s from Uruguay statistics"
              % (n_males, args.name))
        print("%s females for %s from Uruguay statistics"
              % (n_females, args.name))
    elif (args.total == "us"):
        print("%s males for %s from USA statistics"
              % (n_males, args.name))
        print("%s females for %s from USA statistics"
              % (n_females, args.name))
