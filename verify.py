import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from data import PREDICATES


def generate_hard_negatives(triple: dict, entities: list[dict]) -> list[dict]:
    hard_negs = []
    subj = triple.get("subject", "")
    pred = triple.get("predicate", "")
    obj = triple.get("object", "")

    other_preds = [p for p in PREDICATES if p != pred]
    if other_preds:
        hard_negs.append({
            "subject": subj, "predicate": random.choice(other_preds), "object": obj,
            "neg_type": "wrong_predicate"
        })

    ent_texts = [e.get("text", "") for e in entities if e.get("text", "") != subj]
    if ent_texts:
        hard_negs.append({
            "subject": random.choice(ent_texts), "predicate": pred, "object": obj,
            "neg_type": "wrong_subject"
        })

    ent_texts = [e.get("text", "") for e in entities if e.get("text", "") != obj]
    if ent_texts:
        hard_negs.append({
            "subject": subj, "predicate": pred, "object": random.choice(ent_texts),
            "neg_type": "wrong_object"
        })

    return hard_negs


def build_discriminator_data(results: list[dict]) -> tuple[list[dict], list[int]]:
    samples, labels = [], []
    for r in results:
        text = r.get("text", "")
        gold_rels = r.get("gold_rels", [])
        pred_ents = r.get("pred_ents", [])

        for rel in gold_rels:
            samples.append({"triple": rel, "text": text})
            labels.append(1)
            for neg in generate_hard_negatives(rel, pred_ents):
                samples.append({"triple": neg, "text": text})
                labels.append(0)

    return samples, labels


class TriVerifier:
    def __init__(self, embed_model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.clf = LogisticRegression(random_state=42)
        self.is_trained = False

    def _encode_sample(self, sample: dict) -> np.ndarray:
        triple = sample["triple"]
        text = sample["text"]
        triple_text = f"{triple.get('subject', '')} {triple.get('predicate', '')} {triple.get('object', '')}"

        emb_triple = self.embed_model.encode([triple_text])[0]
        emb_text = self.embed_model.encode([text])[0]
        cos_sim = np.dot(emb_triple, emb_text) / (np.linalg.norm(emb_triple) * np.linalg.norm(emb_text) + 1e-8)

        return np.array([cos_sim])

    def train(self, samples: list[dict], labels: list[int]):
        if len(samples) < 10:
            print("Not enough samples for TriVerifier training")
            return

        X = np.array([self._encode_sample(s) for s in samples])
        y = np.array(labels)
        self.clf.fit(X, y)
        self.is_trained = True

    def predict(self, triple: dict, text: str) -> tuple[bool, float]:
        if not self.is_trained:
            return True, 1.0

        sample = {"triple": triple, "text": text}
        X = self._encode_sample(sample).reshape(1, -1)
        prob = self.clf.predict_proba(X)[0][1]
        return prob >= 0.5, prob

    def verify_relations(self, relations: list[dict], text: str, threshold: float = 0.5) -> list[dict]:
        verified = []
        for rel in relations:
            is_valid, conf = self.predict(rel, text)
            if conf >= threshold:
                rel["tri_verifier_conf"] = conf
                verified.append(rel)
        return verified
