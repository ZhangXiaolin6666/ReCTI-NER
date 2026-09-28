import json
import re
import numpy as np
import faiss
from jsonschema import validate, ValidationError
from sentence_transformers import SentenceTransformer
from data import ENTITY_TYPES, PREDICATES, IOC_PATTERNS, refang_ioc, validate_entity_type, validate_predicate
from config import get_client, get_model
from cache import cached_completion, load_index_cache, save_index_cache
embed_model = None
index = None
indexed_examples = []

def get_extraction_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "type": {"type": "string", "enum": list(ENTITY_TYPES)}
                    },
                    "required": ["text", "type"]
                }
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "predicate": {"type": "string", "enum": list(PREDICATES)},
                        "object": {"type": "string"}
                    },
                    "required": ["subject", "predicate", "object"]
                }
            }
        },
        "required": ["entities", "relations"]
    }


# Module-level reference for backward compatibility
EXTRACTION_SCHEMA = get_extraction_schema()


def validate_schema(result: dict) -> bool:
    try:
        validate(instance=result, schema=get_extraction_schema())
        return True
    except ValidationError:
        return False


def normalize_entity(e) -> dict | None:
    """归一化实体格式，支持多种 LLM 返回格式"""
    if isinstance(e, list) and len(e) >= 2:
        return {"text": str(e[0]), "type": str(e[1])}
    if isinstance(e, dict):
        text = e.get("text") or e.get("name") or e.get("value") or ""
        etype = e.get("type") or ""
        if text and etype:
            return {"text": str(text), "type": str(etype)}
    return None


def normalize_relation(r) -> dict | None:
    """归一化关系格式"""
    if isinstance(r, list) and len(r) >= 3:
        return {"subject": str(r[0]), "predicate": str(r[1]), "object": str(r[2])}
    if isinstance(r, dict):
        subj = r.get("subject") or r.get("head") or r.get("src") or ""
        pred = r.get("predicate") or r.get("relation") or r.get("rel") or ""
        obj = r.get("object") or r.get("tail") or r.get("dst") or ""
        if subj and pred and obj:
            return {"subject": str(subj), "predicate": str(pred), "object": str(obj)}
    return None


def build_index(examples: list[dict], model_name: str = "BAAI/bge-small-zh-v1.5"):
    global embed_model, index, indexed_examples
    if not examples:
        return

    # 尝试加载缓存
    cached_index, cached_examples = load_index_cache(examples)
    if cached_index is not None:
        embed_model = SentenceTransformer(model_name)
        index, indexed_examples = cached_index, cached_examples
        return

    embed_model = SentenceTransformer(model_name)
    texts = [ex.get("instruction", "") for ex in examples]
    embeddings = embed_model.encode(texts, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    indexed_examples = examples
    save_index_cache(examples, index, indexed_examples)


def retrieve_examples(query: str, k: int, strategy: str = "semantic") -> list[dict]:
    if k == 0 or index is None or len(indexed_examples) == 0:
        return []
    if strategy == "entity_aware":
        return retrieve_entity_aware(query, k)
    return retrieve_semantic(query, k)


def retrieve_semantic(query: str, k: int) -> list[dict]:
    query_emb = embed_model.encode([query], normalize_embeddings=True)
    k = min(k, len(indexed_examples))
    _, indices = index.search(query_emb.astype(np.float32), k)
    return [indexed_examples[i] for i in indices[0] if i < len(indexed_examples)]


def retrieve_entity_aware(query: str, k: int) -> list[dict]:
    query_iocs = set()
    query_normalized = refang_ioc(query)
    for pattern in IOC_PATTERNS.values():
        query_iocs.update(re.findall(pattern, query_normalized))

    query_emb = embed_model.encode([query], normalize_embeddings=True)
    search_k = min(k * 3, len(indexed_examples))
    scores, indices = index.search(query_emb.astype(np.float32), search_k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= len(indexed_examples):
            continue
        ex = indexed_examples[idx]
        ex_text = refang_ioc(ex.get("instruction", ""))
        ex_iocs = set()
        for pattern in IOC_PATTERNS.values():
            ex_iocs.update(re.findall(pattern, ex_text))

        overlap = len(query_iocs & ex_iocs) / max(len(query_iocs | ex_iocs), 1)
        combined_score = 0.7 * score + 0.3 * overlap
        candidates.append((combined_score, ex))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in candidates[:k]]


def build_prompt(text: str, examples: list[dict]) -> str:
    prompt = f"""你是一个网络安全威胁情报抽取专家。请从给定文本中抽取实体和关系。

## 实体类型（只能使用以下类型）
{', '.join(ENTITY_TYPES)}

## 关系谓词（只能使用以下谓词）
{', '.join(PREDICATES)}

## 输出格式
严格按照 JSON 格式输出，包含 entities 和 relations 两个数组。

"""
    if examples:
        prompt += "## 示例\n\n"
        for i, ex in enumerate(examples[:3], 1):
            prompt += f"### 示例 {i}\n输入：{ex.get('instruction', '')}\n输出：{ex.get('output', '')}\n\n"

    prompt += f"## 待抽取文本\n{text}\n\n请输出 JSON："
    return prompt


def icl_extract(text: str, examples: list[dict]) -> tuple[list, list]:
    prompt = build_prompt(text, examples)

    try:
        content = cached_completion(get_client(), get_model(), [{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        result = json.loads(content)

        # 归一化并过滤实体
        entities = [e for e in (normalize_entity(x) for x in result.get("entities", [])) if e]
        entities = [e for e in entities if validate_entity_type(e.get("type", ""))]

        # 归一化并过滤关系
        relations = [r for r in (normalize_relation(x) for x in result.get("relations", [])) if r]
        relations = [r for r in relations if validate_predicate(r.get("predicate", ""))]

        return entities, relations
    except Exception as e:
        print(f"ICL extract error: {e}")
        return [], []


def verify_span(entity: dict, text: str) -> tuple[bool, str]:
    ent_text = entity.get("text", "")
    ent_type = entity.get("type", "")

    if not ent_text:
        return False, "empty_entity"
    if ent_text not in text and ent_text.lower() not in text.lower():
        return False, "entity_not_in_text"
    if ent_type in IOC_PATTERNS:
        if not re.search(IOC_PATTERNS[ent_type], refang_ioc(ent_text)):
            return False, "ioc_regex_fail"
    return True, ""


def verify_nli(triple: dict, text: str) -> tuple[bool, str]:
    subj, pred, obj = triple.get("subject", ""), triple.get("predicate", ""), triple.get("object", "")
    prompt = f"判断以下假设是否能从前提中推导出来。\n\n前提：{text}\n\n假设：{subj} {pred} {obj}\n\n只回答：蕴含 / 矛盾 / 中立"

    try:
        answer = cached_completion(get_client(), get_model(), [{"role": "user", "content": prompt}], max_completion_tokens=10).strip()
        return (True, "") if "蕴含" in answer or "entail" in answer.lower() else (False, f"nli_fail:{answer}")
    except Exception:
        return True, ""


def verify_consistency(triple: dict, text: str) -> tuple[bool, str]:
    subj, pred, obj = triple.get("subject", ""), triple.get("predicate", ""), triple.get("object", "")
    prompt = f"请判断以下关系三元组是否正确描述了文本中的信息。\n\n文本：{text}\n\n三元组：({subj}, {pred}, {obj})\n\n该三元组是否准确？只回答：是 / 否"

    try:
        answer = cached_completion(get_client(), get_model(), [{"role": "user", "content": prompt}], max_completion_tokens=10).strip()
        return (True, "") if "是" in answer or "yes" in answer.lower() else (False, "consistency_fail")
    except Exception:
        return True, ""


def verify_triple(triple: dict, text: str, entities: list[dict],
                  use_nli: bool = True, use_consistency: bool = True) -> tuple[bool, str, float]:
    confidence = 1.0
    subj, obj = triple.get("subject"), triple.get("object")
    ent_texts = {e.get("text", "") for e in entities}

    if subj not in ent_texts or obj not in ent_texts:
        return False, "subject_or_object_not_entity", 0.0

    if use_nli:
        passed, _ = verify_nli(triple, text)
        if not passed:
            confidence *= 0.3

    if use_consistency:
        passed, reason = verify_consistency(triple, text)
        if not passed:
            return False, reason, confidence * 0.5

    return True, "", confidence


def verify_and_filter(entities: list, relations: list, text: str,
                      use_nli: bool = True, use_consistency: bool = True,
                      allow_abstention: bool = True) -> tuple[list, list, list]:
    verified_ents = [ent for ent in entities if verify_span(ent, text)[0]]

    verified_rels, rejected = [], []
    for rel in relations:
        passed, reason, conf = verify_triple(rel, text, verified_ents, use_nli, use_consistency)
        if passed:
            rel["confidence"] = conf
            verified_rels.append(rel)
        elif allow_abstention:
            rejected.append({"triple": rel, "reason": reason})
        else:
            rel["confidence"] = conf
            verified_rels.append(rel)

    return verified_ents, verified_rels, rejected
