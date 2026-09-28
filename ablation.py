import pandas as pd
from tqdm import tqdm
from config import CONFIG, ABLATIONS
from run import run_pipeline


def run_single_ablation(name: str) -> dict:
    config = {**CONFIG, **ABLATIONS[name]}
    result = run_pipeline(config)
    metrics = result["metrics"]
    return {
        "name": name,
        "ent_p": metrics["entity_micro"]["precision"],
        "ent_r": metrics["entity_micro"]["recall"],
        "ent_f1": metrics["entity_micro"]["f1"],
        "rel_p": metrics["relation_micro"]["precision"],
        "rel_r": metrics["relation_micro"]["recall"],
        "rel_f1": metrics["relation_micro"]["f1"],
        "abstention": metrics.get("abstention_rate", 0),
    }


def run_all_ablations():
    results = []
    for name in tqdm(ABLATIONS.keys(), desc="Ablations"):
        print(f"\n=== Running ablation: {name} ===")
        result = run_single_ablation(name)
        results.append(result)
        print(f"{name}: Rel P={result['rel_p']:.3f}, Rel F1={result['rel_f1']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv("xxxxxxx", index=False)
    print("\nAblation results saved to outputs/results/ablation_results.csv")
    print(df.to_string())
    return df
