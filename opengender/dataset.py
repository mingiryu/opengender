from datasets import load_dataset


SEED = 42


def load_wiki_gendersort():
    dataset = load_dataset("mingiryu/wiki_gendersort")
    dataset = dataset.shuffle(seed=SEED)

    df = dataset["train"].to_pandas()

    df["X"] = df.name.str.lower()
    df["y"] = df.gender.str.lower().str[0]

    return df.dropna()


def load_name_gender_inference():
    dataset = load_dataset("mingiryu/name_gende_inference")
    dataset = dataset.shuffle(seed=SEED)

    df = dataset["train"].to_pandas()

    df["X"] = df.first_name
    df["y"] = df.gender

    return df.dropna()


def load_damegender():
    dataset = load_dataset("mingiryu/damegender")
    dataset = dataset.shuffle(seed=SEED)

    df = dataset["train"].to_pandas()

    df["X"] = df.name.str.lower()
    df["y"] = df.apply(lambda row: "m" if row.male > row.female else "f", axis=1)
    df["y"] = df.apply(lambda row: "u" if row.male == row.female else row.y, axis=1)

    return df.dropna()
