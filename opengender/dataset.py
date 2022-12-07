from datasets import load_dataset


SEED = 7


def load_wiki_gendersort():
    dataset = load_dataset("mingiryu/wiki_gendersort")
    dataset = dataset.shuffle(seed=SEED)

    df = dataset["train"].to_pandas()
    df = df[df.name.notna()]

    df["X"] = df.name.str.lower()
    df["y"] = df.gender.str.lower().str[0]

    df = df[df.y != "u"]
    df = df[df.y != "i"]

    return df


def load_name_gender_inference():
    dataset = load_dataset("mingiryu/name_gende_inference")
    dataset = dataset.shuffle(seed=SEED)

    df = dataset["train"].to_pandas()
    df = df[df.first_name.notna()]
    
    df["X"] = df.first_name
    df["y"] = df.gender

    df = df[df.y != "u"]

    return df


def load_damegender():
    dataset = load_dataset("mingiryu/damegender")
    dataset = dataset.shuffle(seed=SEED)

    df = dataset["train"].to_pandas()
    df = df[df.name.notna()]

    df["X"] = df.name.str.lower()
    df["y"] = df.apply(lambda row: "m" if row.male > row.female else "f", axis=1)
    df["y"] = df.apply(lambda row: "u" if row.male == row.female else row.y, axis=1)

    df = df[df.y != "u"]

    return df
