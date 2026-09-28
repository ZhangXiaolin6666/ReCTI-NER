import re
import json

ENTITY_TYPES = {
    "attacker", "malware", "tool", "file",
    "IP", "Domain", "URL", "vulnerability", "threat type", "SHA256",
    "technique", "MD5", "attack", "application", "ransomware", "SHA1",
}

PREDICATES = {
    "use", "aka", "mount", "spread", "execute", "belong to",
    "exploit", "install", "target", "abuse", "leverage",
    "develop", "download", "drop", "communicate with",
    "hash", "used by", "resolved to", "associated with", "check",
    "deploy", "embed", "load", "include",
}

IOC_TYPES = {"IP", "Domain", "URL", "SHA256"}

IOC_PATTERNS = {
    "IP": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "Domain": r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b",
    "URL": r"https?://[^\s]+",
    "SHA256": r"\b[a-fA-F0-9]{64}\b",
    "MD5": r"\b[a-fA-F0-9]{32}\b",
}


def defang_ioc(text: str) -> str:
    text = re.sub(r'\.(?=\d)', '[.]', text)
    text = text.replace('http://', 'hxxp://').replace('https://', 'hxxps://')
    return text


def refang_ioc(text: str) -> str:
    text = text.replace('[.]', '.').replace('(.)', '.')
    text = text.replace('hxxp://', 'http://').replace('hxxps://', 'https://')
    text = text.replace('[:]', ':')
    return text


def normalize_ioc(text: str) -> str:
    text = refang_ioc(text)
    for pattern in [IOC_PATTERNS["SHA256"], IOC_PATTERNS["MD5"]]:
        text = re.sub(pattern, lambda m: m.group().lower(), text)
    return text


def load_data(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            return json.loads(content)
        return [json.loads(line) for line in content.split('\n') if line.strip()]


def parse_output(output_str: str) -> tuple[list, list]:
    entities, relations = [], []
    if not output_str:
        return entities, relations

    for line in output_str.strip().split('\n'):
        line = line.strip()
        if line.startswith('Named Entities:'):
            ent_part = line[len('Named Entities:'):].strip()
            for match in re.finditer(r'\(([^,]+),\s*([^)]+)\)', ent_part):
                entities.append({"text": match.group(1).strip(), "type": match.group(2).strip()})
        elif line.startswith('Relationships:'):
            rel_part = line[len('Relationships:'):].strip()
            for match in re.finditer(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)', rel_part):
                relations.append({
                    "subject": match.group(1).strip(),
                    "predicate": match.group(2).strip(),
                    "object": match.group(3).strip()
                })
    return entities, relations


def split_data(data: list, val_ratio: float) -> tuple[list, list]:
    split_idx = int(len(data) * (1 - val_ratio))
    return data[:split_idx], data[split_idx:]


def validate_entity_type(etype: str) -> bool:
    return etype in ENTITY_TYPES


def validate_predicate(pred: str) -> bool:
    return pred in PREDICATES


def regex_match_ioc(text: str, ioc_type: str) -> list[str]:
    if ioc_type not in IOC_PATTERNS:
        return []
    return re.findall(IOC_PATTERNS[ioc_type], text)


def load_dataset(name: str) -> list[dict]:
    from config import DATASETS
    ds = DATASETS[name]
    raw = load_data(ds["path"])

    if ds["format"] == "instruction_output":
        return raw

    if ds["format"] == "kg_triples":
        etype_map = ds.get("entity_type_map") or {}
        pred_map = ds.get("predicate_map") or {}
        # Dynamically extend ENTITY_TYPES and PREDICATES
        ENTITY_TYPES.update(etype_map.values())
        PREDICATES.update(pred_map.values())
        # Group triples by source text
        by_text = {}
        for item in raw:
            text = item.get("text", "")
            if text not in by_text:
                by_text[text] = {"entities": [], "relations": []}
            for ent in item.get("entities", []):
                mapped_type = etype_map.get(ent.get("type", ""), ent.get("type", ""))
                by_text[text]["entities"].append({"text": ent.get("text", ""), "type": mapped_type})
            for rel in item.get("relations", []):
                mapped_pred = pred_map.get(rel.get("predicate", ""), rel.get("predicate", ""))
                by_text[text]["relations"].append({
                    "subject": rel.get("subject", ""), "predicate": mapped_pred, "object": rel.get("object", "")
                })
        results = []
        for text, data in by_text.items():
            ents_str = ", ".join(f"({e['text']}, {e['type']})" for e in data["entities"])
            rels_str = ", ".join(f"({r['subject']}, {r['predicate']}, {r['object']})" for r in data["relations"])
            output = f"Named Entities: {ents_str}\nRelationships: {rels_str}"
            results.append({"instruction": text, "output": output})
        return results

    if ds["format"] == "bio_spans":
        etype_map = ds.get("entity_type_map") or {}
        ENTITY_TYPES.update(etype_map.values())
        results = []
        for item in raw:
            text = item.get("text", "")
            tokens = item.get("tokens", [])
            tags = item.get("tags", [])
            entities = []
            current_ent, current_type = [], None
            for token, tag in zip(tokens, tags):
                if tag.startswith("B-"):
                    if current_ent:
                        mapped = etype_map.get(current_type, current_type)
                        entities.append({"text": " ".join(current_ent), "type": mapped})
                    current_ent = [token]
                    current_type = tag[2:]
                elif tag.startswith("I-") and current_type == tag[2:]:
                    current_ent.append(token)
                else:
                    if current_ent:
                        mapped = etype_map.get(current_type, current_type)
                        entities.append({"text": " ".join(current_ent), "type": mapped})
                    current_ent, current_type = [], None
            if current_ent:
                mapped = etype_map.get(current_type, current_type)
                entities.append({"text": " ".join(current_ent), "type": mapped})
            ents_str = ", ".join(f"({e['text']}, {e['type']})" for e in entities)
            output = f"Named Entities: {ents_str}\nRelationships: "
            results.append({"instruction": text or " ".join(tokens), "output": output})
        return results

    raise ValueError(f"Unknown dataset format: {ds['format']}")
