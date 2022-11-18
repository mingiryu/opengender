import os
import re
import argparse

from opengender.dame_sexmachine import DameSexmachine

parser = argparse.ArgumentParser()
parser.add_argument("name", help="display the gender")
parser.add_argument(
    "--ml",
    choices=[
        "svc",
        "forest",
    ],
)
parser.add_argument(
    "--total",
    default="inter",
    choices=[
        "namdict",
        "inter",
    ],
)
# More about iso codes on https://www.iso.org/obp/ui/
# You can set alphabet with sufix:
# So russian in latin alphabet would be ru_en
parser.add_argument("--version", action="version", version="0.4")
parser.add_argument("--force_whitespaces", default=False, action="store_true")
args = parser.parse_args()

results = []

s = DameSexmachine()
n_males = s.name_frec(
    args.name, dataset=args.total, force_whitespaces=args.force_whitespaces
)["males"]
n_females = s.name_frec(
    args.name, dataset=args.total, force_whitespaces=args.force_whitespaces
)["females"]
if int(n_males) > int(n_females):
    print("%s's gender is male" % (str(args.name)))
    prob = int(n_males) / (int(n_males) + int(n_females))
    print("probability: %s" % str(prob))
elif int(n_males) < int(n_females):
    print("%s's gender is female" % (str(args.name)))
    prob = int(n_females) / (int(n_females) + int(n_males))
    print("probability: %s" % str(prob))
elif (int(n_males) == 0) and (int(n_females) == 0):
    args.ml = "nltk"

if args.ml:
    if args.ml == "svc":
        guess = s.guess(
            args.name,
            binary=True,
            ml="svc",
            force_whitespaces=args.force_whitespaces,
        )
    elif args.ml == "forest":
        guess = s.guess(
            args.name,
            binary=True,
            ml="forest",
            force_whitespaces=args.force_whitespaces,
        )

    if guess == 1:
        sex = "male"
    elif guess == 0:
        sex = "female"
    elif guess == 2:
        sex = "unknown"
    print("%s gender predicted is %s" % (str(args.name), sex))
