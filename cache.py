import json
import hashlib
import pickle
import faiss
import numpy as np
from pathlib import Path

CACHE_DIR = Path("outputs/cache")
LLM_CACHE_DIR = CACHE_DIR / "llm"
INDEX_CACHE_DIR = CACHE_DIR / "index"
INTERMEDIATE_DIR = CACHE_DIR / "intermediate"
CHECKPOINT_FILE = CACHE_DIR / "checkpoint.json"

for d in [LLM_CACHE_DIR, INDEX_CACHE_DIR, INTERMEDIATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_current_config_hash = None
_cost_stats = {"calls": 0, "cache_hits": 0, "est_prompt_tokens": 0, "est_completion_tokens": 0}


def set_config_hash(config: dict):
    """设置当前配置的 hash，用于区分不同实验的缓存"""
    global _current_config_hash
    content = json.dumps(config, sort_keys=True, ensure_ascii=False)
    _current_config_hash = hashlib.md5(content.encode()).hexdigest()[:8]

# === LLM 缓存 ===
def llm_cache_key(model: str, messages: list, **kwargs) -> str:
    content = json.dumps({"model": model, "messages": messages, **kwargs}, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()


def get_llm_cache(key: str) -> str | None:
    cache_file = LLM_CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["content"]
    return None


def set_llm_cache(key: str, content: str):
    cache_file = LLM_CACHE_DIR / f"{key}.json"
    cache_file.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")


def cached_completion(client, model: str, messages: list, **kwargs):
    """带缓存的 LLM 调用"""
    key = llm_cache_key(model, messages, **kwargs)
    cached = get_llm_cache(key)
    if cached is not None:
        _cost_stats["cache_hits"] += 1
        return cached
    _cost_stats["calls"] += 1
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    _cost_stats["est_prompt_tokens"] += prompt_chars // 4
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    content = response.choices[0].message.content
    _cost_stats["est_completion_tokens"] += len(content) // 4
    set_llm_cache(key, content)
    return content


def get_cost_stats() -> dict:
    return dict(_cost_stats)


# === FAISS 索引缓存 ===
def get_index_cache_key(examples: list) -> str:
    content = json.dumps([e.get("instruction", "")[:100] for e in examples[:10]], ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()


def load_index_cache(examples: list):
    key = get_index_cache_key(examples)
    index_file = INDEX_CACHE_DIR / f"{key}.faiss"
    examples_file = INDEX_CACHE_DIR / f"{key}.pkl"
    if index_file.exists() and examples_file.exists():
        index = faiss.read_index(str(index_file))
        cached_examples = pickle.loads(examples_file.read_bytes())
        return index, cached_examples
    return None, None


def save_index_cache(examples: list, index, indexed_examples: list):
    key = get_index_cache_key(examples)
    faiss.write_index(index, str(INDEX_CACHE_DIR / f"{key}.faiss"))
    (INDEX_CACHE_DIR / f"{key}.pkl").write_bytes(pickle.dumps(indexed_examples))


# === 中间结果缓存 ===
def save_intermediate(name: str, data):
    """保存中间结果，包含配置 hash"""
    suffix = f"_{_current_config_hash}" if _current_config_hash else ""
    (INTERMEDIATE_DIR / f"{name}{suffix}.json").write_text(
        json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8"
    )


def load_intermediate(name: str):
    """加载中间结果，优先加载配置特定的缓存"""
    suffix = f"_{_current_config_hash}" if _current_config_hash else ""
    f = INTERMEDIATE_DIR / f"{name}{suffix}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    # 回退到无后缀版本（兼容旧缓存）
    f = INTERMEDIATE_DIR / f"{name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


# === 检查点 ===
def save_checkpoint(step: str, data: dict = None):
    content = {"step": step, "data": data or {}}
    CHECKPOINT_FILE.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")


def load_checkpoint() -> dict | None:
    return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8")) if CHECKPOINT_FILE.exists() else None


def clear_checkpoint():
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


# === 缓存管理 ===
def clear_all_cache():
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    for d in [LLM_CACHE_DIR, INDEX_CACHE_DIR, INTERMEDIATE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("Cache cleared.")


def cache_stats():
    llm_files = list(LLM_CACHE_DIR.glob("*.json"))
    index_files = list(INDEX_CACHE_DIR.glob("*"))
    llm_size = sum(f.stat().st_size for f in llm_files)
    index_size = sum(f.stat().st_size for f in index_files)
    print(f"LLM cache: {len(llm_files)} entries, {llm_size/1024/1024:.2f} MB")
    print(f"Index cache: {len(index_files)} files, {index_size/1024/1024:.2f} MB")
    print(f"Checkpoint: {'exists' if CHECKPOINT_FILE.exists() else 'none'}")
