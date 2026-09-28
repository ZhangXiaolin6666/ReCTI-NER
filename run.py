import argparse
import json
import numpy as np
from loguru import logger
from tqdm import tqdm
from collections import Counter
from sentence_transformers import SentenceTransformer

from config import CONFIG, TEST_CONFIG, DATASETS
from data import load_data, load_dataset, parse_output, split_data
from baseline import llm_baseline_extract, analyze_errors
from augment import (rewrite_for_error, compute_fix_evidence, compute_features,
                     train_acceptor, calibrate_threshold, accept_candidates,
                     diversity_select, generate_multiview_candidates)
from verify import TriVerifier, build_discriminator_data
from extract import build_index, retrieve_examples, icl_extract, verify_and_filter
from evaluate import (compute_metrics_micro_macro, abstention_rate, regex_verifiable_rate,
                      precision_coverage_curve, find_operating_point)
from cache import save_checkpoint, load_checkpoint, clear_checkpoint, save_intermediate, load_intermediate, clear_all_cache, cache_stats, set_config_hash, get_cost_stats


def augment_data(errors: list, config: dict, baseline_fn) -> tuple[list, dict]:
    embed_model = SentenceTransformer(config["embed_model"])
    candidates, features_list, labels = [], [], []
    fix_rate_by_type = Counter()
    total_by_type = Counter()

    for err in tqdm(errors[:min(30, len(errors))], desc="Generating"):
        rewrites = rewrite_for_error(err["text"], err["error_type"], config.get("augment_n", 3))
        for rewrite in rewrites:
            if not rewrite:
                continue
            fix = compute_fix_evidence(err["text"], rewrite, err, baseline_fn) if config.get("use_fix_evidence", True) else 0
            feat = compute_features(err["text"], rewrite, fix, embed_model)
            candidates.append({
                "instruction": rewrite, "output": err.get("gold_output", ""),
                "original": err["text"], "error_type": err["error_type"]
            })
            features_list.append(feat)
            labels.append(fix)
            total_by_type[err["error_type"]] += 1
            if fix:
                fix_rate_by_type[err["error_type"]] += 1

    stats = {
        "total_candidates": len(candidates),
        "fix_success": sum(labels),
        "fix_rate_by_type": {k: fix_rate_by_type[k] / total_by_type[k] if total_by_type[k] else 0 for k in total_by_type},
    }

    if not candidates:
        return [], stats

    X, y = np.array(features_list), np.array(labels)

    if config.get("use_learned_acceptor", True) and y.sum() >= 5:
        clf = train_acceptor(X, y)
        threshold = calibrate_threshold(clf, X, y, config.get("target_precision", 0.95))
        accepted_mask = accept_candidates(clf, threshold, X)
    else:
        accepted_mask = X[:, 0] > 0.7

    accepted = [c for c, m in zip(candidates, accepted_mask) if m]
    stats["accepted_count"] = len(accepted)

    if len(accepted) > 10:
        embs = embed_model.encode([a["instruction"] for a in accepted])
        selected_idx = diversity_select(embs, min(len(accepted), 50), use_dpp=config.get("use_dpp", True))
        accepted = [accepted[i] for i in selected_idx]

    stats["final_count"] = len(accepted)
    return accepted, stats


def analyze_failure_profile(rejected: list) -> dict:
    profile = {"predicates": Counter(), "reasons": Counter()}
    for r in rejected:
        profile["predicates"][r.get("triple", {}).get("predicate", "unknown")] += 1
        profile["reasons"][r.get("reason", "unknown")] += 1
    return profile


def targeted_augment(profile: dict, train: list, config: dict) -> list:
    weak_predicates = [p for p, _ in profile["predicates"].most_common(3)]
    targeted_samples = [d for d in train if any(pred in d.get("output", "") for pred in weak_predicates)]

    augmented = []
    for d in targeted_samples[:10]:
        for rewrite in rewrite_for_error(d.get("instruction", ""), "missing", 2):
            if rewrite:
                augmented.append({"instruction": rewrite, "output": d.get("output", "")})
    return augmented


def verifier_loop(train: list, val: list, augmented: list, config: dict) -> tuple:
    logger.info("=== Verifier-in-the-Loop ===")
    loop_metrics = []

    for iteration in range(config.get("max_loop_iter", 3)):
        logger.info(f"Loop iteration {iteration + 1}/{config.get('max_loop_iter', 3)}")

        build_index(train + augmented, config["embed_model"])
        rejected_all = []
        iter_results = []

        for d in val[:min(20, len(val))]:
            text = d.get("instruction", "")
            examples = retrieve_examples(text, config["top_k"])
            entities, relations = icl_extract(text, examples)
            verified_ents, verified_rels, rejected = verify_and_filter(
                entities, relations, text,
                use_nli=config.get("use_nli", True),
                use_consistency=config.get("use_consistency", True)
            )
            rejected_all.extend(rejected)
            gold_ents, gold_rels = parse_output(d.get("output", ""))
            iter_results.append({"pred_ents": verified_ents, "gold_ents": gold_ents,
                                 "pred_rels": verified_rels, "gold_rels": gold_rels})

        iter_metrics = compute_metrics_micro_macro(iter_results)
        loop_metrics.append({
            "iteration": iteration + 1,
            "entity_f1": iter_metrics["entity_micro"]["f1"],
            "relation_f1": iter_metrics["relation_micro"]["f1"],
            "entity_precision": iter_metrics["entity_micro"]["precision"],
            "entity_recall": iter_metrics["entity_micro"]["recall"],
            "relation_precision": iter_metrics["relation_micro"]["precision"],
            "relation_recall": iter_metrics["relation_micro"]["recall"],
            "num_rejected": len(rejected_all),
        })
        logger.info(f"Iter {iteration+1}: Ent F1={iter_metrics['entity_micro']['f1']:.3f}, Rel F1={iter_metrics['relation_micro']['f1']:.3f}")

        if not rejected_all:
            logger.info("No rejections, stopping loop")
            break

        profile = analyze_failure_profile(rejected_all)
        logger.info(f"Failure profile: {profile}")

        if profile:
            new_augmented = targeted_augment(profile, train, config)
            augmented.extend(new_augmented)
            logger.info(f"Added {len(new_augmented)} targeted samples")

    return augmented, train, loop_metrics


def run_pipeline(config: dict, use_cache: bool = True) -> dict:
    logger.info("=== EAS-IE Pipeline ===")
    set_config_hash(config)
    checkpoint = load_checkpoint() if use_cache else None
    step = checkpoint["step"] if checkpoint else None

    # 1. Load data
    dataset_name = config.get("dataset", "instruction_output")
    logger.info(f"Loading dataset: {dataset_name}")
    data = load_dataset(dataset_name) if dataset_name in DATASETS else load_data(config["data_path"])
    train, val = split_data(data, config["val_ratio"])
    n = config.get("test_n", 50)  # 测试模式用少量数据
    if config.get("test_mode"):
        train, val = train[:n], val[:n]
        logger.info(f"[TEST MODE] Using {n} samples")
    logger.info(f"Train: {len(train)}, Val: {len(val)}")

    # 2. Baseline + error analysis
    if step in [None, "baseline"]:
        cached = load_intermediate("baseline") if use_cache else None
        if cached:
            baseline_preds, errors, error_dist, error_by_type = cached["preds"], cached["errors"], cached["dist"], cached["by_type"]
            logger.info(f"Loaded cached baseline: {len(errors)} errors")
        else:
            logger.info("Running baseline...")
            baseline_preds = []
            for d in tqdm(train[:min(n, len(train))], desc="Baseline"):
                baseline_preds.append(llm_baseline_extract(d.get("instruction", "")))
            golds = [parse_output(d.get("output", "")) for d in train[:min(n, len(train))]]
            texts = [d.get("instruction", "") for d in train[:min(n, len(train))]]
            errors = analyze_errors(baseline_preds, golds, texts)
            error_dist = dict(Counter(e["error_type"] for e in errors))
            error_by_type = {}
            for e in errors:
                etype = e.get("entity_type") or "unknown"
                if etype not in error_by_type:
                    error_by_type[etype] = {"boundary": 0, "type": 0, "missing": 0}
                error_by_type[etype][e["error_type"]] += 1
            save_intermediate("baseline", {"preds": baseline_preds, "errors": errors, "dist": error_dist, "by_type": error_by_type})
            logger.info(f"Found {len(errors)} errors")
        save_checkpoint("augment")
    else:
        # Load from cache when resuming past this step
        cached = load_intermediate("baseline") if use_cache else None
        if cached:
            baseline_preds, errors, error_dist, error_by_type = cached["preds"], cached["errors"], cached["dist"], cached["by_type"]
            logger.info(f"Loaded cached baseline: {len(errors)} errors")
        else:
            baseline_preds, errors, error_dist, error_by_type = [], [], {}, {}

    # 3. Data augmentation
    augmentation_stats = {}
    if step in [None, "baseline", "augment"]:
        cached = load_intermediate("augmented") if use_cache else None
        if cached:
            augmented = cached
            logger.info(f"Loaded cached augmented: {len(augmented)} samples")
        else:
            if errors:
                augmented, augmentation_stats = augment_data(errors, config, llm_baseline_extract)
            else:
                augmented = []
            logger.info(f"Generated {len(augmented)} error-aware samples")
            if config.get("use_multiview", True):
                for d in tqdm(train[:min(n, len(train))], desc="Multiview"):
                    augmented.extend(generate_multiview_candidates(d.get("instruction", ""), d.get("output", "")))
                logger.info(f"Total augmented: {len(augmented)} samples")
            save_intermediate("augmented", augmented)
        save_checkpoint("verifier")
    else:
        # Load from cache when resuming past this step
        cached = load_intermediate("augmented") if use_cache else None
        augmented = cached if cached else []
        logger.info(f"Loaded cached augmented: {len(augmented)} samples")

    # 4. Train TriVerifier
    tri_verifier = TriVerifier(config["embed_model"])
    if config.get("use_tri_verifier", True):
        logger.info("Training TriVerifier...")
        golds = [parse_output(d.get("output", "")) for d in train[:min(n, len(train))]]
        texts = [d.get("instruction", "") for d in train[:min(n, len(train))]]
        baseline_preds_local = load_intermediate("baseline")["preds"] if use_cache and load_intermediate("baseline") else baseline_preds
        baseline_results = [{"text": t, "pred_ents": p[0], "gold_rels": g[1]} for p, g, t in zip(baseline_preds_local, golds, texts)]
        disc_samples, disc_labels = build_discriminator_data(baseline_results)
        tri_verifier.train(disc_samples, disc_labels)

    # 5. Verifier-in-the-Loop
    loop_metrics = []
    if config.get("use_loop", True):
        augmented, train, loop_metrics = verifier_loop(train, val, augmented, config)

    # 6. Build index
    logger.info("Building index...")
    build_index(train + augmented, config["embed_model"])

    # 7. Extract + verify
    cached_extract = load_intermediate("extract_results") if use_cache else None
    if cached_extract:
        results = cached_extract["results"]
        all_rejected = cached_extract["all_rejected"]
        all_confidences = cached_extract["all_confidences"]
        all_labels = cached_extract["all_labels"]
        total_rels = cached_extract["total_rels"]
        accepted_rels = cached_extract["accepted_rels"]
        logger.info(f"Loaded cached extract results: {len(results)} samples")
    else:
        logger.info("Extracting...")
        results, all_rejected, all_confidences, all_labels = [], [], [], []
        total_rels, accepted_rels = 0, 0

        for d in tqdm(val, desc="Extract"):
            text = d.get("instruction", "")
            examples = retrieve_examples(text, config["top_k"], config.get("retrieval_strategy", "semantic"))
            entities, relations = icl_extract(text, examples)
            verified_ents, verified_rels, rejected = verify_and_filter(
                entities, relations, text, config.get("use_nli", True), config.get("use_consistency", True), config.get("allow_abstention", True))
            if config.get("use_tri_verifier", True) and tri_verifier.is_trained:
                verified_rels = tri_verifier.verify_relations(verified_rels, text, config.get("tri_verifier_threshold", 0.5))
            gold_ents, gold_rels = parse_output(d.get("output", ""))
            gold_rel_set = {(r.get("subject"), r.get("predicate"), r.get("object")) for r in gold_rels}
            for rel in verified_rels:
                all_confidences.append(rel.get("confidence", 1.0))
                all_labels.append(1 if (rel.get("subject"), rel.get("predicate"), rel.get("object")) in gold_rel_set else 0)
            results.append({"text": text, "pred_ents": verified_ents, "gold_ents": gold_ents, "pred_rels": verified_rels, "gold_rels": gold_rels})
            total_rels += len(relations)
            accepted_rels += len(verified_rels)
            all_rejected.extend(rejected)

        save_intermediate("extract_results", {
            "results": results, "all_rejected": all_rejected,
            "all_confidences": all_confidences, "all_labels": all_labels,
            "total_rels": total_rels, "accepted_rels": accepted_rels
        })

    # 8. Evaluate
    metrics = compute_metrics_micro_macro(results)
    metrics["abstention_rate"] = abstention_rate(total_rels, accepted_rels)
    metrics["ioc_verifiable_rate"] = regex_verifiable_rate([e for r in results for e in r["pred_ents"]])
    pc_curve = precision_coverage_curve(all_confidences, all_labels)
    operating_point = find_operating_point(all_confidences, all_labels, config.get("target_precision", 0.95))
    logger.info(f"Operating point: threshold={operating_point['threshold']:.3f}, precision={operating_point['precision']:.3f}, coverage={operating_point['coverage']:.3f}")
    logger.info(f"Results: {json.dumps(metrics, indent=2, ensure_ascii=False)}")

    with open(results_filename, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics, "pc_curve": pc_curve, "operating_point": operating_point,
            "error_distribution": error_dist, "error_by_type": error_by_type,
            "augmentation_stats": augmentation_stats, "loop_metrics": loop_metrics,
            "cost_stats": get_cost_stats(),
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {results_filename}")

    clear_checkpoint()
    return {"metrics": metrics, "results": results, "rejected": all_rejected, "pc_curve": pc_curve, "operating_point": operating_point}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "baselines", "ablation", "visualize", "all"], default="all")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="instruction_output", help="Dataset to use")
    parser.add_argument("--test", action="store_true", help="Quick test with minimal data (3 samples)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cache, run from scratch")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all cache and exit")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics and exit")
    args = parser.parse_args()

    if args.clear_cache:
        clear_all_cache()
        return
    if args.cache_stats:
        cache_stats()
        return

    logger.add("xxxxxx", rotation="1 day")
    use_cache = not args.no_cache
    config = TEST_CONFIG if args.test else CONFIG
    config = {**config, "dataset": args.dataset}

    if args.mode == "train":
        run_pipeline(config, use_cache)
    elif args.mode == "baselines":
        from baselines import run_all_baselines
        run_all_baselines()
    elif args.mode == "ablation":
        from ablation import run_all_ablations
        run_all_ablations()
    elif args.mode == "visualize":
        from visualize import generate_all_figures
        generate_all_figures()
    elif args.mode == "all":
        run_pipeline(config, use_cache)
        if not args.test:
            from baselines import run_all_baselines
            from ablation import run_all_ablations
            from visualize import generate_all_figures
            run_all_baselines()
            run_all_ablations()
            generate_all_figures()


if __name__ == "__main__":
    main()
