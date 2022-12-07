import pandas as pd

from datasets import load_dataset

from opengender import DATA_DIR


RANDOM_STATE = 7
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


def load_wiki_gendersort():
    dataset = load_dataset("mingiryu/wiki_gendersort")
    dataset = dataset.shuffle(seed=RANDOM_STATE)

    df = dataset["train"].to_pandas()
    df = df[df.name.notna()]

    df["X"] = df.name
    df["y"] = df.gender.str.lower().str[0]

    df = df[df.y != "u"]
    df = df[df.y != "i"]

    return df


def load_name_gender_inference():
    dataset = load_dataset("mingiryu/name_gende_inference")
    dataset = dataset.shuffle(seed=RANDOM_STATE)

    df = dataset["train"].to_pandas()
    df = df[df.first_name.notna()]

    df["X"] = df.first_name
    df["y"] = df.gender

    df = df[df.y != "u"]

    return df


def load_damegender():
    dataset = load_dataset("mingiryu/damegender")
    dataset = dataset.shuffle(seed=RANDOM_STATE)

    df = dataset["train"].to_pandas()
    df = df[df.name.notna()]

    df["X"] = df.name
    df["y"] = df.apply(
        lambda row: "m" if row.male > row.female else "f", axis=1
    )  # noqa
    df["y"] = df.apply(
        lambda row: "u" if row.male == row.female else row.y, axis=1
    )  # noqa

    df = df[df.y != "u"]

    return df


def splits():
    train = pd.concat(
        [load_damegender()[["X", "y"]], load_wiki_gendersort()[["X", "y"]]]
    )
    test = load_name_gender_inference()[["X", "y"]]

    # Make sure test data is unseen during training
    train = train[~train.X.isin(test.X)]
    train = train.dropna()

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)


if __name__ == "__main__":
    splits()
