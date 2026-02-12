import re
import pandas as pd

# Common abbreviations and expansions
ABBR = {
    "mktplace": "marketplace",
    "pmt": "payment",
    "txn": "transaction",
    "eccom": "ecommerce",
    "acc": "accessory",
    "elec": "electronics",
    "dept": "department",
    "mart": "market",
}


def clean_text(text: str) -> str:
    """Clean and normalise raw transaction text."""
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Expand abbreviations
    tokens = []
    for tok in text.split():
        if tok in ABBR:
            tokens.append(ABBR[tok])
        else:
            tokens.append(tok)

    return " ".join(tokens)


def load_and_clean(path: str = "../data/transactions.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].apply(clean_text)
    return df


if __name__ == "__main__":
    df = load_and_clean("../data/transactions.csv")
    print(df.head())
