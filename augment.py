import re
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from data import defang_ioc, refang_ioc
from config import get_client, get_model
from cache import cached_completion

ERROR_INSTRUCTIONS = {
    "boundary": "改写以下句子，改变实体前后的修饰词和上下文，使实体边界更加清晰明确。保持原有实体和关系不变。",
    "type": "改写以下句子，添加明确的类型指示词（如'恶意软件'、'攻击组织'、'漏洞'等），使实体类型更容易识别。保持原有实体和关系不变。",
    "missing": "改写以下句子，使用并列结构或分句形式，使所有实体都能被突出显示。保持原有实体和关系不变。",
}

MULTIVIEW_INSTRUCTIONS = {
    "paraphrase": "保持语义和所有实体关系完全不变，用不同的表达方式改写这个句子。",
    "formal": "将这个句子改写为更正式的技术报告风格，保持所有实体和关系不变。",
    "simplify": "简化句子结构，使实体和关系更清晰易识别，但不改变原有信息。",
}


def rewrite_for_error(text: str, error_type: str, n: int = 3) -> list[str]:
    instruction = ERROR_INSTRUCTIONS.get(error_type, ERROR_INSTRUCTIONS["missing"])
    prompt = f"{instruction}\n\n原文：{text}\n\n请生成{n}个不同的改写版本，每行一个："

    try:
        content = cached_completion(get_client(), get_model(), [{"role": "user", "content": prompt}])
        candidates = content.strip().split('\n')
        return [c for c in (re.sub(r'^[\d]+[.、)）\s]+', '', c.strip()) for c in candidates) if c][:n]
    except Exception as e:
        print(f"Rewrite error: {e}")
        return []


def generate_ioc_variants(text: str) -> list[str]:
    variants = []
    defanged = defang_ioc(text)
    if defanged != text:
        variants.append(defanged)
    refanged = refang_ioc(text)
    if refanged != text and refanged != defanged:
        variants.append(refanged)
    return variants


def multiview_rewrite(text: str, style: str = "paraphrase") -> str:
    instruction = MULTIVIEW_INSTRUCTIONS.get(style, MULTIVIEW_INSTRUCTIONS["paraphrase"])
    prompt = f"{instruction}\n\n原文：{text}\n\n改写："

    try:
        return cached_completion(get_client(), get_model(), [{"role": "user", "content": prompt}], max_completion_tokens=500).strip()
    except Exception as e:
        print(f"Multiview rewrite error: {e}")
        return ""


def generate_multiview_candidates(text: str, gold_output: str) -> list[dict]:
    candidates = []
    for variant in generate_ioc_variants(text):
        candidates.append({"instruction": variant, "output": gold_output, "augment_type": "ioc_variant"})

    style = random.choice(list(MULTIVIEW_INSTRUCTIONS.keys()))
    rewritten = multiview_rewrite(text, style)
    if rewritten:
        candidates.append({"instruction": rewritten, "output": gold_output, "augment_type": f"multiview_{style}"})
    return candidates


def compute_fix_evidence(original: str, candidate: str, error: dict, baseline_extract_fn) -> int:
    try:
        new_pred = baseline_extract_fn(candidate)
        gold = error["gold"]

        if error["error_type"] == "type":
            for ent in new_pred[0]:
                if ent.get("text") == gold.get("text") and ent.get("type") == gold.get("type"):
                    return 1
        elif error["error_type"] == "boundary":
            for ent in new_pred[0]:
                if ent.get("text") == gold.get("text", ""):
                    return 1
        elif error["error_type"] == "missing":
            gold_text = gold.get("text", "")
            for ent in new_pred[0]:
                ent_text = ent.get("text", "")
                if gold_text and ent_text and (gold_text in ent_text or ent_text in gold_text):
                    return 1
    except Exception:
        pass
    return 0


def compute_features(original: str, candidate: str, fix: int, embed_model: SentenceTransformer) -> np.ndarray:
    from Levenshtein import distance as levenshtein_distance

    emb_orig = embed_model.encode([original])[0]
    emb_cand = embed_model.encode([candidate])[0]
    sim = float(cosine_similarity([emb_orig], [emb_cand])[0][0])
    len_ratio = len(candidate) / len(original) if original else 1.0
    edit_dist = levenshtein_distance(original, candidate) / max(len(original), len(candidate), 1)

    return np.array([sim, len_ratio, edit_dist, fix])


def train_acceptor(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)
    return clf


def calibrate_threshold(clf, X_val: np.ndarray, y_val: np.ndarray, target_precision: float = 0.95) -> float:
    probs = clf.predict_proba(X_val)[:, 1]
    best_threshold = 1.0
    for threshold in np.linspace(0.99, 0.3, 70):
        mask = probs >= threshold
        if mask.sum() == 0:
            continue
        precision = y_val[mask].mean()
        if precision >= target_precision:
            best_threshold = threshold
    return best_threshold


def accept_candidates(clf, threshold: float, features: np.ndarray) -> np.ndarray:
    probs = clf.predict_proba(features)[:, 1]
    return probs >= threshold


def diversity_select(embeddings: np.ndarray, k: int, use_dpp: bool = True) -> list[int]:
    n = len(embeddings)
    if n <= k:
        return list(range(n))

    sim_matrix = cosine_similarity(embeddings)

    if use_dpp:
        selected = [0]
        for _ in range(k - 1):
            min_sims = []
            for i in range(n):
                if i not in selected:
                    max_sim_to_selected = max(sim_matrix[i, j] for j in selected)
                    min_sims.append((i, max_sim_to_selected))
            if min_sims:
                best_idx = min(min_sims, key=lambda x: x[1])[0]
                selected.append(best_idx)
        return selected
    else:
        selected = [random.randint(0, n - 1)]
        for _ in range(k - 1):
            best_idx, best_score = -1, -1
            for i in range(n):
                if i not in selected:
                    min_sim = min(sim_matrix[i, j] for j in selected)
                    if (1 - min_sim) > best_score:
                        best_score = 1 - min_sim
                        best_idx = i
            if best_idx >= 0:
                selected.append(best_idx)
        return selected
