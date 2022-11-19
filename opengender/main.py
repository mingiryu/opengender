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

# More about iso codes on https://www.iso.org/obp/ui/
# You can set alphabet with sufix:
# So russian in latin alphabet would be ru_en
parser.add_argument("--version", action="version", version="0.4")
parser.add_argument("--force_whitespaces", default=False, action="store_true")
args = parser.parse_args()

results = []

s = DameSexmachine()

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
