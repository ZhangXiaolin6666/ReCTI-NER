import json
import pandas as pd
from tqdm import tqdm
from data import ENTITY_TYPES, PREDICATES, load_data, split_data, parse_output
from config import CONFIG, get_client, get_model
from evaluate import compute_metrics_micro_macro
from extract import build_index, retrieve_entity_aware, get_extraction_schema
from cache import cached_completion


def call_llm(prompt: str, json_mode: bool = False) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    try:
        return cached_completion(get_client(), get_model(), [{"role": "user", "content": prompt}], **kwargs)
    except Exception as e:
        print(f"LLM error: {e}")
        return "{}"


def parse_baseline_output(output: str) -> tuple[list, list]:
    """解析 baseline LLM 输出，支持多种格式"""
    try:
        data = json.loads(output)

        # 归一化实体
        entities = []
        for e in data.get("entities", []):
            if isinstance(e, list) and len(e) >= 2:
                entities.append({"text": str(e[0]), "type": str(e[1])})
            elif isinstance(e, dict):
                text = e.get("text") or e.get("name") or e.get("value") or ""
                etype = e.get("type") or ""
                if text and etype:
                    entities.append({"text": str(text), "type": str(etype)})

        # 归一化关系
        relations = []
        for r in data.get("relations", []):
            if isinstance(r, list) and len(r) >= 3:
                relations.append({"subject": str(r[0]), "predicate": str(r[1]), "object": str(r[2])})
            elif isinstance(r, dict):
                subj = r.get("subject") or r.get("head") or ""
                pred = r.get("predicate") or r.get("relation") or ""
                obj = r.get("object") or r.get("tail") or ""
                if subj and pred and obj:
                    relations.append({"subject": subj, "predicate": pred, "object": obj})

        return entities, relations
    except:
        return parse_output(output)


def run_chatie(text: str) -> tuple[list, list]:
    entities_str = call_llm(f"请列出以下网络安全文本中的所有实体（攻击者、恶意软件、IP、域名等），用JSON格式输出：\n{text}", json_mode=True)
    relations_str = call_llm(f"文本：{text}\n实体：{entities_str}\n这些实体之间有什么关系？用JSON格式输出关系三元组：", json_mode=True)
    ents, _ = parse_baseline_output(entities_str)
    _, rels = parse_baseline_output(relations_str)
    return ents, rels


def run_gpt_re(text: str, train_data: list) -> tuple[list, list]:
    examples = retrieve_entity_aware(text, 3) if train_data else []
    example_str = "".join(f"示例{i}：\n输入：{ex.get('instruction', '')}\n输出：{ex.get('output', '')}\n\n"
                         for i, ex in enumerate(examples[:2], 1))
    prompt = f"{example_str}可能的关系类型：{', '.join(PREDICATES)}\n文本：{text}\n请用JSON格式输出实体和关系三元组："
    return parse_baseline_output(call_llm(prompt, json_mode=True))


def run_gliner(text: str) -> tuple[list, list]:
    prompt = f"请识别以下文本中的命名实体。\n实体类型：{', '.join(ENTITY_TYPES)}\n文本：{text}\n用JSON格式输出："
    ents, _ = parse_baseline_output(call_llm(prompt, json_mode=True))
    return ents, []


def run_instructuie(text: str) -> tuple[list, list]:
    prompt = f"""Task: Named Entity Recognition and Relation Extraction
Domain: Cybersecurity Threat Intelligence
Entity types: {', '.join(ENTITY_TYPES)}
Relation types: {', '.join(PREDICATES)}
Input: {text}
Output in JSON format with "entities" and "relations" arrays."""
    return parse_baseline_output(call_llm(prompt, json_mode=True))


def run_odie(text: str) -> tuple[list, list]:
    prompt = f"Extract only the following types from the text:\nEntity types: {', '.join(list(ENTITY_TYPES)[:5])}\nText: {text}\nOutput in JSON format:"
    return parse_baseline_output(call_llm(prompt, json_mode=True))


def run_no_schema(text: str) -> tuple[list, list]:
    return parse_baseline_output(call_llm(f"从以下文本中抽取网络安全实体和关系，用JSON格式输出：\n{text}", json_mode=True))


def run_orellm(text: str) -> tuple[list, list]:
    prompt = f"Extract all relations from the text. Relations can be any meaningful connection between entities.\nText: {text}\nOutput in JSON format:"
    return parse_baseline_output(call_llm(prompt, json_mode=True))


def run_nature_json(text: str) -> tuple[list, list]:
    prompt = f"Extract information according to this schema:\n{json.dumps(get_extraction_schema(), indent=2)}\nText: {text}"
    return parse_baseline_output(call_llm(prompt, json_mode=True))


def run_all_baselines():
    data = load_data(CONFIG["data_path"])
    train, val = split_data(data, CONFIG["val_ratio"])
    build_index(train, CONFIG["embed_model"])

    baselines = {
        "ChatIE": run_chatie,
        "GPT-RE": lambda t: run_gpt_re(t, train),
        "GLiNER": run_gliner,
        "InstructUIE": run_instructuie,
        "ODIE": run_odie,
        "NoSchema": run_no_schema,
        "ORELLM": run_orellm,
        "NatureJSON": run_nature_json,
    }

    all_results = []
    for name, fn in baselines.items():
        print(f"Running {name}...")
        results = []
        for d in tqdm(val[:min(20, len(val))], desc=name):
            pred_ents, pred_rels = fn(d.get("instruction", ""))
            gold_ents, gold_rels = parse_output(d.get("output", ""))
            results.append({"pred_ents": pred_ents, "gold_ents": gold_ents, "pred_rels": pred_rels, "gold_rels": gold_rels})

        metrics = compute_metrics_micro_macro(results)
        all_results.append({
            "name": name,
            "ent_p": metrics["entity_micro"]["precision"],
            "ent_r": metrics["entity_micro"]["recall"],
            "ent_f1": metrics["entity_micro"]["f1"],
            "rel_p": metrics["relation_micro"]["precision"],
            "rel_r": metrics["relation_micro"]["recall"],
            "rel_f1": metrics["relation_micro"]["f1"],
        })
        print(f"{name}: Ent F1={metrics['entity_micro']['f1']:.3f}, Rel F1={metrics['relation_micro']['f1']:.3f}")

    df = pd.DataFrame(all_results)
    df.to_csv("outputs/results/baselines.csv", index=False)
    print("\nBaseline results saved to outputs/results/baselines.csv")
    print(df.to_string())
    return df
