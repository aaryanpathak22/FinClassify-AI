import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import torch
from transformers import AutoTokenizer, AutoModel

from data_prep import load_and_clean


class DistilBertEmbedder:
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Return a 2D numpy array of embeddings for a list of texts."""
        embs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=64,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**enc)
            # Use [CLS]-like first token embedding
            cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
            embs.append(cls_emb)

        return np.vstack(embs)


def main() -> None:
    print("Loading and cleaning data...")
    df = load_and_clean("../data/transactions.csv")

    texts = df["clean_text"].tolist()
    labels = df["category"].tolist()

    print("Encoding labels...")
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print("Generating DistilBERT embeddings...")
    embedder = DistilBertEmbedder()
    X = embedder.encode(texts)

    print("Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training Logistic Regression (multinomial)...")
    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_val)
    report = classification_report(
        y_val,
        y_pred,
        target_names=le.classes_,
        digits=3,
    )
    print(report)

    print("Saving model artifacts...")
    joblib.dump(clf, "../models/finclassify_model.pkl")
    joblib.dump(le, "../models/finclassify_labels.pkl")

    print("Training complete.")


if __name__ == "__main__":
    main()
