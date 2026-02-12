import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import joblib
import numpy as np

from data_prep import clean_text
from train_model import DistilBertEmbedder


MODELS_DIR = Path("../models")
DATA_DIR = Path("../data")
FEEDBACK_LOG = DATA_DIR / "feedback_log.csv"


class FinClassifyService:
    """
    FinClassify AI inference + XAI + feedback logging service.
    """

    def __init__(self) -> None:
        self.embedder = DistilBertEmbedder()
        self.model = joblib.load(MODELS_DIR / "finclassify_model.pkl")
        self.label_encoder = joblib.load(MODELS_DIR / "finclassify_labels.pkl")
        self.class_names: List[str] = list(self.label_encoder.classes_)

    def _embed(self, text: str) -> np.ndarray:
        clean = clean_text(text)
        emb = self.embedder.encode([clean])
        return clean, emb

    def predict(self, raw_text: str) -> Dict[str, Any]:
        """
        Predict category + probability distribution.
        """
        clean, emb = self._embed(raw_text)
        proba = self.model.predict_proba(emb)[0]
        idx = int(np.argmax(proba))
        category = str(self.label_encoder.inverse_transform([idx])[0])
        confidence = float(proba[idx])

        scores = {
            str(self.label_encoder.inverse_transform([i])[0]): float(p)
            for i, p in enumerate(proba)
        }

        return {
            "input": raw_text,
            "clean_text": clean,
            "predicted_category": category,
            "confidence": confidence,
            "scores": scores,
        }

    def explain(self, raw_text: str, top_k: int = 5) -> List[Dict[str, float]]:
        """
        Simple word-importance explanation:
        - Compute baseline confidence of predicted class
        - For each unique token, remove it and measure drop in confidence
        - Higher drop => more important word
        """
        base_pred = self.predict(raw_text)
        base_conf = base_pred["confidence"]
        tokens = base_pred["clean_text"].split()

        if len(tokens) <= 1:
            return []

        importances: List[Dict[str, float]] = []
        seen = set()

        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)

            masked_tokens = [t for t in tokens if t != tok]
            masked_text = " ".join(masked_tokens) if masked_tokens else ""
            masked_pred = self.predict(masked_text) if masked_text else base_pred
            masked_conf = masked_pred["confidence"]
            delta = base_conf - masked_conf

            importances.append(
                {
                    "token": tok,
                    "delta_confidence": float(delta),
                }
            )

        # Sort by confidence drop (descending) and keep top_k positive
        importances = [
            imp for imp in importances if imp["delta_confidence"] > 0
        ]
        importances.sort(key=lambda x: x["delta_confidence"], reverse=True)

        return importances[:top_k]

    def log_feedback(
        self,
        raw_text: str,
        predicted_category: str,
        correct_category: str,
    ) -> None:
        """
        Append feedback to feedback_log.csv for future retraining.
        """
        FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        is_new_file = not FEEDBACK_LOG.exists()

        with FEEDBACK_LOG.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(
                    ["timestamp", "text", "predicted_category", "correct_category"]
                )
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    raw_text,
                    predicted_category,
                    correct_category,
                ]
            )


if __name__ == "__main__":
    service = FinClassifyService()
    example = "Starbud Gas #2345 NY"
    pred = service.predict(example)
    print("Prediction:", pred)
    print("Top explanation tokens:", service.explain(example))
