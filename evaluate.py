import re
import numpy as np
from collections import defaultdict
from data import IOC_TYPES, IOC_PATTERNS, ENTITY_TYPES, PREDICATES, refang_ioc


def entity_to_tuple(e):
    if isinstance(e, dict):
        text = e.get("text", "")
        etype = e.get("type", "")
        # Handle cases where text or type is a list
        if isinstance(text, list):
            text = tuple(text) if text else ""
        if isinstance(etype, list):
            etype = tuple(etype) if etype else ""
        return (text, etype)
    if isinstance(e, list):
        return tuple(e)
    return e


def _to_hashable(val):
    """Convert a value to a hashable type."""
    if isinstance(val, list):
        return tuple(val) if val else ""
    return val if val is not None else ""


def relation_to_tuple(r):
    if isinstance(r, dict):
        return (_to_hashable(r.get("subject", "")),
                _to_hashable(r.get("predicate", "")),
                _to_hashable(r.get("object", "")))
    if isinstance(r, (list, tuple)) and len(r) >= 3:
        return tuple(_to_hashable(x) for x in r[:3])
    return r


def precision_recall_f1(pred: list, gold: list, to_tuple_fn) -> dict:
    pred_set = set(to_tuple_fn(p) for p in pred)
    gold_set = set(to_tuple_fn(g) for g in gold)

    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0
    r = tp / len(gold_set) if gold_set else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0

    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": len(pred_set) - tp, "fn": len(gold_set) - tp}


def compute_metrics_micro_macro(results: list[dict]) -> dict:
    total_tp_ent, total_fp_ent, total_fn_ent = 0, 0, 0
    total_tp_rel, total_fp_rel, total_fn_rel = 0, 0, 0
    type_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    pred_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for r in results:
        pred_ents, gold_ents = r.get("pred_ents", []), r.get("gold_ents", [])
        pred_rels, gold_rels = r.get("pred_rels", []), r.get("gold_rels", [])

        ent_result = precision_recall_f1(pred_ents, gold_ents, entity_to_tuple)
        total_tp_ent += ent_result["tp"]
        total_fp_ent += ent_result["fp"]
        total_fn_ent += ent_result["fn"]

        pred_by_type, gold_by_type = defaultdict(list), defaultdict(list)
        for e in pred_ents:
            if isinstance(e, dict):
                pred_by_type[e.get("type", "")].append(e)
        for e in gold_ents:
            if isinstance(e, dict):
                gold_by_type[e.get("type", "")].append(e)
        for etype in ENTITY_TYPES:
            type_result = precision_recall_f1(pred_by_type[etype], gold_by_type[etype], entity_to_tuple)
            type_metrics[etype]["tp"] += type_result["tp"]
            type_metrics[etype]["fp"] += type_result["fp"]
            type_metrics[etype]["fn"] += type_result["fn"]

        rel_result = precision_recall_f1(pred_rels, gold_rels, relation_to_tuple)
        total_tp_rel += rel_result["tp"]
        total_fp_rel += rel_result["fp"]
        total_fn_rel += rel_result["fn"]

        pred_by_pred, gold_by_pred = defaultdict(list), defaultdict(list)
        for rel in pred_rels:
            if isinstance(rel, dict):
                pred_by_pred[rel.get("predicate", "")].append(rel)
        for rel in gold_rels:
            if isinstance(rel, dict):
                gold_by_pred[rel.get("predicate", "")].append(rel)
        for predicate in PREDICATES:
            pred_result = precision_recall_f1(pred_by_pred[predicate], gold_by_pred[predicate], relation_to_tuple)
            pred_metrics[predicate]["tp"] += pred_result["tp"]
            pred_metrics[predicate]["fp"] += pred_result["fp"]
            pred_metrics[predicate]["fn"] += pred_result["fn"]

    def calc_prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        return {"precision": p, "recall": r, "f1": f1}

    type_prfs = [calc_prf(m["tp"], m["fp"], m["fn"]) for m in type_metrics.values() if m["tp"] + m["fp"] + m["fn"] > 0]
    pred_prfs = [calc_prf(m["tp"], m["fp"], m["fn"]) for m in pred_metrics.values() if m["tp"] + m["fp"] + m["fn"] > 0]

    entity_macro = {
        "precision": np.mean([p["precision"] for p in type_prfs]) if type_prfs else 0,
        "recall": np.mean([p["recall"] for p in type_prfs]) if type_prfs else 0,
        "f1": np.mean([p["f1"] for p in type_prfs]) if type_prfs else 0,
    }
    relation_macro = {
        "precision": np.mean([p["precision"] for p in pred_prfs]) if pred_prfs else 0,
        "recall": np.mean([p["recall"] for p in pred_prfs]) if pred_prfs else 0,
        "f1": np.mean([p["f1"] for p in pred_prfs]) if pred_prfs else 0,
    }

    per_entity_type = {etype: calc_prf(m["tp"], m["fp"], m["fn"])
                        for etype, m in type_metrics.items() if m["tp"] + m["fp"] + m["fn"] > 0}
    per_predicate = {pred: calc_prf(m["tp"], m["fp"], m["fn"])
                     for pred, m in pred_metrics.items() if m["tp"] + m["fp"] + m["fn"] > 0}

    return {
        "entity_micro": calc_prf(total_tp_ent, total_fp_ent, total_fn_ent),
        "entity_macro": entity_macro,
        "relation_micro": calc_prf(total_tp_rel, total_fp_rel, total_fn_rel),
        "relation_macro": relation_macro,
        "per_entity_type": per_entity_type,
        "per_predicate": per_predicate,
    }


def abstention_rate(total: int, accepted: int) -> float:
    return 1 - accepted / total if total else 0


def regex_verifiable_rate(entities: list[dict]) -> float:
    ioc_ents = [e for e in entities if isinstance(e, dict) and e.get("type") in IOC_TYPES]
    if not ioc_ents:
        return 1.0

    verified = 0
    for e in ioc_ents:
        etype, etext = e.get("type", ""), refang_ioc(e.get("text", ""))
        if etype in IOC_PATTERNS and re.fullmatch(IOC_PATTERNS[etype], etext):
            verified += 1
    return verified / len(ioc_ents)


def precision_coverage_curve(confidences: list[float], labels: list[int]) -> dict:
    if not confidences:
        return {"precisions": [], "coverages": [], "thresholds": []}

    scores, labels = np.array(confidences), np.array(labels)
    thresholds = np.linspace(0, 1, 100)
    precisions, coverages = [], []

    for t in thresholds:
        mask = scores >= t
        if mask.sum() == 0:
            continue
        precisions.append(float(labels[mask].mean()))
        coverages.append(float(mask.mean()))

    return {"precisions": precisions, "coverages": coverages, "thresholds": thresholds.tolist()}


def find_operating_point(confidences: list[float], labels: list[int], target_precision: float = 0.95) -> dict:
    curve = precision_coverage_curve(confidences, labels)
    for p, c, t in zip(curve["precisions"], curve["coverages"], curve["thresholds"]):
        if p >= target_precision:
            return {"threshold": t, "precision": p, "coverage": c}
    return {"threshold": 1.0, "precision": 1.0, "coverage": 0.0}
