import os
from dotenv import load_dotenv

load_dotenv()

os.makedirs("outputs/logs", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

DATASETS = {
    "instruction_output": {
        "path": "data/data.json",
        "format": "instruction_output",
        "entity_type_map": None,
        "predicate_map": None,
    },
    "kg_triples": {
        "path": "data/kg_triples.json",
        "format": "kg_triples",
        "entity_type_map": {
            "MALWARE": "malware", "THREAT_ACTOR": "attacker", "TOOL": "tool",
            "VULNERABILITY": "vulnerability", "IP_ADDRESS": "IP", "DOMAIN": "Domain",
            "URL": "URL", "FILE": "file", "HASH_SHA256": "SHA256", "HASH_MD5": "MD5",
            "HASH_SHA1": "SHA1", "TECHNIQUE": "technique", "ATTACK": "attack",
            "APPLICATION": "application", "RANSOMWARE": "ransomware",
        },
        "predicate_map": {
            "uses": "use", "also_known_as": "aka", "exploits": "exploit",
            "targets": "target", "downloads": "download", "drops": "drop",
            "communicates_with": "communicate with", "associated_with": "associated with",
            "installs": "install", "deploys": "deploy",
        },
    },
    "bio_spans": {
        "path": "data/bio_spans.json",
        "format": "bio_spans",
        "entity_type_map": {
            "MAL": "malware", "APT": "attacker", "TOOL": "tool",
            "VULN": "vulnerability", "IP": "IP", "DOM": "Domain",
            "URL": "URL", "FILE": "file", "HASH": "SHA256",
            "TECH": "technique", "ATT": "attack",
        },
        "predicate_map": None,
    },
}

CONFIG = {
    "dataset": "instruction_output",
    "data_path": "data/data.json",
    "val_ratio": 0.2,
    "llm_model": "qwen-plus",
    "embed_model": "BAAI/bge-small-zh-v1.5",
    "top_k": 8,
    "retrieval_strategy": "semantic",
    "augment_n": 3,
    "use_fix_evidence": True,
    "use_learned_acceptor": True,
    "use_dpp": True,
    "use_multiview": True,
    "use_nli": True,
    "use_consistency": True,
    "allow_abstention": True,
    "use_tri_verifier": True,
    "tri_verifier_threshold": 0.5,
    "use_loop": True,
    "max_loop_iter": 3,
    "target_precision": 0.95,
}

TEST_CONFIG = {
    **CONFIG,
    "test_mode": True,
    "test_n": 3,  # 每步只处理3条数据
    "use_loop": False,
    "use_multiview": False,
    "max_loop_iter": 1,
}

ABLATIONS = {
    "full": {},
    "no_fix_evidence": {"use_fix_evidence": False},
    "fixed_threshold": {"use_learned_acceptor": False},
    "no_dpp": {"use_dpp": False},
    "no_multiview": {"use_multiview": False},
    "no_tri_verifier": {"use_tri_verifier": False},
    "no_nli": {"use_nli": False},
    "no_consistency": {"use_consistency": False},
    "no_abstention": {"allow_abstention": False},
    "no_loop": {"use_loop": False},
    "top_k_0": {"top_k": 0},
    "top_k_4": {"top_k": 4},
    "top_k_16": {"top_k": 16},
    "entity_aware": {"retrieval_strategy": "entity_aware"},
}

_client = None


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


def get_model() -> str:
    return CONFIG["llm_model"]
