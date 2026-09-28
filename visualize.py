import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data import ENTITY_TYPES

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_precision_coverage(save_path="outputs/figures/pr_coverage.png"):
    try:
        with open("outputs/results/main_results.json", "r") as f:
            data = json.load(f)
    except:
        print("No main results found, skipping P-C curve")
        return

    pc_curve = data.get("pc_curve", {})
    precisions, coverages = pc_curve.get("precisions", []), pc_curve.get("coverages", [])
    operating_point = data.get("operating_point", {})

    if not precisions or not coverages:
        print("No P-C curve data, skipping")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(coverages, precisions, 'b-', linewidth=2, marker='o', markersize=4)
    plt.axhline(y=0.95, color='r', linestyle='--', label='Target P=0.95')

    if operating_point:
        plt.scatter([operating_point.get("coverage", 0)], [operating_point.get("precision", 0)],
                    color='green', s=100, zorder=5, label=f'Operating Point')

    plt.xlabel("Coverage", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Coverage Curve", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_heatmap(save_path="outputs/figures/error_heatmap.png"):
    try:
        with open("outputs/results/main_results.json", "r") as f:
            error_by_type = json.load(f).get("error_by_type", {})
    except:
        error_by_type = {}

    entity_types = list(ENTITY_TYPES)
    error_types = ["boundary", "type", "missing"]

    matrix = np.zeros((len(entity_types), len(error_types)))
    for i, et in enumerate(entity_types):
        if et in error_by_type:
            for j, err in enumerate(error_types):
                matrix[i, j] = error_by_type[et].get(err, 0)

    plt.figure(figsize=(8, 10))
    sns.heatmap(matrix, xticklabels=error_types, yticklabels=entity_types, annot=True, fmt='.0f', cmap='YlOrRd')
    plt.title("Error Distribution by Entity Type", fontsize=14)
    plt.xlabel("Error Type")
    plt.ylabel("Entity Type")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_ablation_comparison(save_path="outputs/figures/ablation_comparison.png"):
    try:
        df = pd.read_csv("outputs/results/ablation_results.csv")
    except:
        print("No ablation results found, skipping")
        return

    plt.figure(figsize=(12, 6))
    x = range(len(df))
    width = 0.35

    plt.bar([i - width/2 for i in x], df['rel_p'], width, label='Relation Precision', color='steelblue')
    plt.bar([i + width/2 for i in x], df['rel_f1'], width, label='Relation F1', color='coral')

    plt.xlabel('Ablation', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Ablation Study Results', fontsize=14)
    plt.xticks(x, df['name'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_baseline_comparison(save_path="outputs/figures/baseline_comparison.png"):
    try:
        df = pd.read_csv("outputs/results/baselines.csv")
    except:
        print("No baseline results found, skipping")
        return

    plt.figure(figsize=(10, 6))
    x = range(len(df))
    width = 0.25

    plt.bar([i - width for i in x], df['ent_f1'], width, label='Entity F1', color='steelblue')
    plt.bar(x, df['rel_p'], width, label='Relation Precision', color='coral')
    plt.bar([i + width for i in x], df['rel_f1'], width, label='Relation F1', color='seagreen')

    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Baseline Comparison', fontsize=14)
    plt.xticks(x, df['name'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def _load_main_results():
    try:
        with open("outputs/results/main_results.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def generate_latex_table(save_path="outputs/figures/main_comparison_table.tex"):
    try:
        df = pd.read_csv("outputs/results/baselines.csv")
    except Exception:
        print("No baselines.csv, skipping LaTeX table")
        return
    data = _load_main_results()
    rows = []
    for _, row in df.iterrows():
        rows.append(f"  {row['name']} & {row['ent_p']:.3f} & {row['ent_r']:.3f} & {row['ent_f1']:.3f} & {row['rel_p']:.3f} & {row['rel_r']:.3f} & {row['rel_f1']:.3f} \\\\")
    if data:
        m = data["metrics"]
        em, rm = m["entity_micro"], m["relation_micro"]
        rows.append("  \\midrule")
        rows.append(f"  \\textbf{{EAS-IE}} & {em['precision']:.3f} & {em['recall']:.3f} & {em['f1']:.3f} & {rm['precision']:.3f} & {rm['recall']:.3f} & {rm['f1']:.3f} \\\\")
    header = "\\begin{tabular}{lcccccc}\n  \\toprule\n  Method & Ent P & Ent R & Ent F1 & Rel P & Rel R & Rel F1 \\\\\n  \\midrule"
    footer = "  \\bottomrule\n\\end{tabular}"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n" + footer + "\n")
    print(f"Saved: {save_path}")


def generate_ablation_table(save_path="outputs/figures/ablation_table.tex"):
    try:
        df = pd.read_csv("outputs/results/ablation_results.csv")
    except Exception:
        print("No ablation_results.csv, skipping")
        return
    rows = []
    for _, row in df.iterrows():
        rows.append(f"  {row['name']} & {row['ent_f1']:.3f} & {row['rel_p']:.3f} & {row['rel_f1']:.3f} & {row.get('abstention', 0):.3f} \\\\")
    header = "\\begin{tabular}{lcccc}\n  \\toprule\n  Variant & Ent F1 & Rel P & Rel F1 & Abstention \\\\\n  \\midrule"
    footer = "  \\bottomrule\n\\end{tabular}"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n" + footer + "\n")
    print(f"Saved: {save_path}")


def plot_augment_before_after(save_path="outputs/figures/augment_before_after.png"):
    try:
        df = pd.read_csv("outputs/results/ablation_results.csv")
    except Exception:
        print("No ablation results, skipping augment before/after")
        return
    no_fix = df[df["name"] == "no_fix_evidence"]
    full = df[df["name"] == "full"]
    if no_fix.empty or full.empty:
        print("Missing ablation rows for augment comparison, skipping")
        return
    labels = ["Entity F1", "Relation P", "Relation F1"]
    before = [float(no_fix.iloc[0]["ent_f1"]), float(no_fix.iloc[0]["rel_p"]), float(no_fix.iloc[0]["rel_f1"])]
    after = [float(full.iloc[0]["ent_f1"]), float(full.iloc[0]["rel_p"]), float(full.iloc[0]["rel_f1"])]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, before, width, label="w/o Fix Evidence", color="lightcoral")
    plt.bar(x + width/2, after, width, label="Full (w/ Fix Evidence)", color="steelblue")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Effect of Fix Evidence on Augmentation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_predicate_performance(save_path="outputs/figures/predicate_performance.png"):
    data = _load_main_results()
    if not data:
        print("No main results, skipping predicate performance")
        return
    per_pred = data.get("metrics", {}).get("per_predicate", {})
    if not per_pred:
        print("No per_predicate data, skipping")
        return
    preds = sorted(per_pred.keys())
    f1s = [per_pred[p]["f1"] for p in preds]
    plt.figure(figsize=(12, 5))
    plt.barh(preds, f1s, color="steelblue")
    plt.xlabel("F1 Score")
    plt.title("Relation F1 by Predicate Type")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_entity_type_performance(save_path="outputs/figures/entity_type_performance.png"):
    data = _load_main_results()
    if not data:
        print("No main results, skipping entity type performance")
        return
    per_ent = data.get("metrics", {}).get("per_entity_type", {})
    if not per_ent:
        print("No per_entity_type data, skipping")
        return
    types = sorted(per_ent.keys())
    p_vals = [per_ent[t]["precision"] for t in types]
    r_vals = [per_ent[t]["recall"] for t in types]
    f1_vals = [per_ent[t]["f1"] for t in types]
    x = np.arange(len(types))
    width = 0.25
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, p_vals, width, label="Precision", color="steelblue")
    plt.bar(x, r_vals, width, label="Recall", color="coral")
    plt.bar(x + width, f1_vals, width, label="F1", color="seagreen")
    plt.xticks(x, types, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Performance by Entity Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_topk_sensitivity(save_path="outputs/figures/topk_sensitivity.png"):
    try:
        df = pd.read_csv("outputs/results/ablation_results.csv")
    except Exception:
        print("No ablation results, skipping top-k sensitivity")
        return
    topk_rows = df[df["name"].str.startswith("top_k_")]
    if topk_rows.empty:
        full = df[df["name"] == "full"]
        if full.empty:
            print("No top-k data, skipping")
            return
        topk_rows = pd.concat([topk_rows, full])
    ks = []
    for name in topk_rows["name"]:
        if name == "full":
            ks.append(8)
        else:
            ks.append(int(name.split("_")[-1]))
    topk_rows = topk_rows.copy()
    topk_rows["k"] = ks
    topk_rows = topk_rows.sort_values("k")
    plt.figure(figsize=(8, 5))
    plt.plot(topk_rows["k"], topk_rows["ent_f1"], "o-", label="Entity F1", color="steelblue")
    plt.plot(topk_rows["k"], topk_rows["rel_f1"], "s-", label="Relation F1", color="coral")
    plt.xlabel("Top-K")
    plt.ylabel("F1 Score")
    plt.title("Sensitivity to Top-K Retrieval")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_loop_curve(save_path="outputs/figures/loop_iteration_curve.png"):
    data = _load_main_results()
    if not data:
        print("No main results, skipping loop curve")
        return
    loop_metrics = data.get("loop_metrics", [])
    if not loop_metrics:
        print("No loop metrics data, skipping")
        return
    iters = [m["iteration"] for m in loop_metrics]
    ent_f1 = [m["entity_f1"] for m in loop_metrics]
    rel_f1 = [m["relation_f1"] for m in loop_metrics]
    plt.figure(figsize=(8, 5))
    plt.plot(iters, ent_f1, "o-", label="Entity F1", color="steelblue")
    plt.plot(iters, rel_f1, "s-", label="Relation F1", color="coral")
    plt.xlabel("Loop Iteration")
    plt.ylabel("F1 Score")
    plt.title("Verifier-in-the-Loop Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(iters)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_fix_success_rate(save_path="outputs/figures/fix_success_rate.png"):
    data = _load_main_results()
    if not data:
        print("No main results, skipping fix success rate")
        return
    stats = data.get("augmentation_stats", {})
    fix_rate = stats.get("fix_rate_by_type", {})
    if not fix_rate:
        print("No fix rate data, skipping")
        return
    types = list(fix_rate.keys())
    rates = [fix_rate[t] for t in types]
    plt.figure(figsize=(8, 5))
    plt.bar(types, rates, color=["lightcoral", "steelblue", "seagreen"][:len(types)])
    plt.ylabel("Fix Success Rate")
    plt.title("Augmentation Fix Success Rate by Error Type")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_figures():
    os.makedirs("outputs/figures", exist_ok=True)
    plot_precision_coverage()
    plot_error_heatmap()
    plot_ablation_comparison()
    plot_baseline_comparison()
    generate_latex_table()
    generate_ablation_table()
    plot_augment_before_after()
    plot_predicate_performance()
    plot_entity_type_performance()
    plot_topk_sensitivity()
    plot_loop_curve()
    plot_fix_success_rate()
    print("All figures generated.")
