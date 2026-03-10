"""Generate report from experiment results."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_report(results_dir: str):
    """Read results CSV and produce report + plots."""
    csv_path = os.path.join(results_dir, "results.csv")
    diag_path = os.path.join(results_dir, "oracle_diagnostics.csv")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    has_diag = os.path.exists(diag_path)
    df_diag = pd.read_csv(diag_path) if has_diag else None

    # Average over seeds and files
    group_cols = ["network_type", "target_plr", "protection_method"]
    metrics = ["PESQ", "STOI", "ESTOI", "SI-SDR", "post_repair_loss_rate",
               "concealment_rate", "mean_burst_len", "max_burst_len"]
    summary = df.groupby(group_cols)[metrics].mean().reset_index()

    # ─── Plot 1: STOI vs PLR per network type ────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, net in zip(axes, ["random_loss", "burst_loss", "jitter_discard"]):
        sub = summary[summary["network_type"] == net]
        for method in ["none", "random", "heuristic", "importance_aware"]:
            m = sub[sub["protection_method"] == method]
            label = method.replace("_", " ").title()
            marker = {"none": "x", "random": "s", "heuristic": "^", "importance_aware": "o"}[method]
            ax.plot(m["target_plr"] * 100, m["STOI"], marker=marker, label=label, linewidth=2)
        ax.set_xlabel("Target PLR (%)")
        ax.set_title(net.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("STOI")
    fig.suptitle("STOI vs Packet Loss Rate", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "stoi_vs_plr.png"), dpi=150)
    plt.close()

    # ─── Plot 1b: PESQ vs PLR per network type ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, net in zip(axes, ["random_loss", "burst_loss", "jitter_discard"]):
        sub = summary[summary["network_type"] == net]
        for method in ["none", "random", "heuristic", "importance_aware"]:
            m = sub[sub["protection_method"] == method]
            label = method.replace("_", " ").title()
            marker = {"none": "x", "random": "s", "heuristic": "^", "importance_aware": "o"}[method]
            ax.plot(m["target_plr"] * 100, m["PESQ"], marker=marker, label=label, linewidth=2)
        ax.set_xlabel("Target PLR (%)")
        ax.set_title(net.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("PESQ (MOS-LQO)")
    fig.suptitle("PESQ vs Packet Loss Rate", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pesq_vs_plr.png"), dpi=150)
    plt.close()

    # ─── Plot 2: SI-SDR vs PLR per network type ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, net in zip(axes, ["random_loss", "burst_loss", "jitter_discard"]):
        sub = summary[summary["network_type"] == net]
        for method in ["none", "random", "heuristic", "importance_aware"]:
            m = sub[sub["protection_method"] == method]
            label = method.replace("_", " ").title()
            marker = {"none": "x", "random": "s", "heuristic": "^", "importance_aware": "o"}[method]
            ax.plot(m["target_plr"] * 100, m["SI-SDR"], marker=marker, label=label, linewidth=2)
        ax.set_xlabel("Target PLR (%)")
        ax.set_title(net.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("SI-SDR (dB)")
    fig.suptitle("SI-SDR vs Packet Loss Rate", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "si_sdr_vs_plr.png"), dpi=150)
    plt.close()

    # ─── Plot 3: Post-repair loss rate ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, net in zip(axes, ["random_loss", "burst_loss", "jitter_discard"]):
        sub = summary[summary["network_type"] == net]
        for method in ["none", "random", "heuristic", "importance_aware"]:
            m = sub[sub["protection_method"] == method]
            label = method.replace("_", " ").title()
            marker = {"none": "x", "random": "s", "heuristic": "^", "importance_aware": "o"}[method]
            ax.plot(m["target_plr"] * 100, m["post_repair_loss_rate"] * 100,
                    marker=marker, label=label, linewidth=2)
        ax.set_xlabel("Target PLR (%)")
        ax.set_title(net.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Post-Repair Loss Rate (%)")
    fig.suptitle("Post-Repair Loss Rate vs Target PLR", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "post_repair_loss.png"), dpi=150)
    plt.close()

    # ─── Plot 4: Oracle diagnostics (Spearman correlation) ────────
    if df_diag is not None and len(df_diag) > 0:
        diag_summary = df_diag.groupby("method")[["spearman_corr", "precision_at_20pct"]].mean()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        methods = diag_summary.index.tolist()
        x = range(len(methods))

        ax1.bar(x, diag_summary["spearman_corr"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha="right")
        ax1.set_ylabel("Spearman Correlation")
        ax1.set_title("Oracle Ranking Correlation")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(x, diag_summary["precision_at_20pct"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45, ha="right")
        ax2.set_ylabel("Precision @ 20%")
        ax2.set_title("Protected Frame Precision")
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Importance Method Diagnostics vs Oracle", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "oracle_diagnostics.png"), dpi=150)
        plt.close()

    # ─── Generate Markdown Report ─────────────────────────────────
    report_lines = [
        "# Importance-Aware Frame Protection: Experiment Report\n",
        f"**Generated from**: `{csv_path}`\n",
        f"**Total experiments**: {len(df)}\n",
        f"**Audio files**: {df['file'].nunique()}\n",
        f"**Seeds**: {sorted(df['seed'].unique())}\n",
        "",
        "## Experiment Configuration\n",
        f"- **Codec**: EnCodec 24kHz causal, 3 kbps",
        f"- **Protection budget**: {df['budget_frac'].iloc[0]*100:.0f}% extra frames",
        f"- **Network types**: {', '.join(sorted(df['network_type'].unique()))}",
        f"- **PLRs**: {', '.join(str(int(p*100))+'%' for p in sorted(df['target_plr'].unique()))}",
        f"- **Protection methods**: {', '.join(sorted(df['protection_method'].unique()))}",
        "",
        "## Summary Results (averaged over files and seeds)\n",
    ]

    # Table per network type
    NETWORK_TYPES = sorted(df["network_type"].unique())
    for net in NETWORK_TYPES:
        report_lines.append(f"### {net.replace('_', ' ').title()}\n")
        sub = summary[summary["network_type"] == net]
        pivot_pesq = sub.pivot(index="target_plr", columns="protection_method", values="PESQ")
        pivot_stoi = sub.pivot(index="target_plr", columns="protection_method", values="STOI")
        pivot_sdr = sub.pivot(index="target_plr", columns="protection_method", values="SI-SDR")
        pivot_loss = sub.pivot(index="target_plr", columns="protection_method", values="post_repair_loss_rate")

        report_lines.append("**PESQ (MOS-LQO):**\n")
        report_lines.append(pivot_pesq.round(3).to_markdown())
        report_lines.append("")
        report_lines.append("**STOI:**\n")
        report_lines.append(pivot_stoi.round(4).to_markdown())
        report_lines.append("")
        report_lines.append("**SI-SDR (dB):**\n")
        report_lines.append(pivot_sdr.round(2).to_markdown())
        report_lines.append("")
        report_lines.append("**Post-Repair Loss Rate:**\n")
        report_lines.append(pivot_loss.round(4).to_markdown())
        report_lines.append("")

    # Oracle diagnostics section
    if df_diag is not None and len(df_diag) > 0:
        report_lines.append("## Importance Method Diagnostics (vs Oracle)\n")
        diag_summary = df_diag.groupby("method")[["spearman_corr", "precision_at_20pct"]].mean()
        report_lines.append(diag_summary.round(4).to_markdown())
        report_lines.append("")

    # Interpretation
    report_lines.append("## Interpretation\n")

    # Find best method per condition
    best_results = []
    for net in NETWORK_TYPES:
        for plr in sorted(df["target_plr"].unique()):
            sub = summary[(summary["network_type"] == net) & (summary["target_plr"] == plr)]
            if len(sub) == 0:
                continue
            best_stoi = sub.loc[sub["STOI"].idxmax()]
            best_results.append({
                "network": net,
                "plr": plr,
                "best_method": best_stoi["protection_method"],
                "stoi": best_stoi["STOI"],
            })

    imp_wins = sum(1 for r in best_results if r["best_method"] == "importance_aware")
    total = len(best_results)
    report_lines.append(
        f"- **Importance-aware wins {imp_wins}/{total} conditions** "
        f"(highest STOI among all methods)."
    )

    # Check where improvement is largest
    for net in NETWORK_TYPES:
        net_results = summary[summary["network_type"] == net]
        imp = net_results[net_results["protection_method"] == "importance_aware"]
        rand = net_results[net_results["protection_method"] == "random"]
        if len(imp) > 0 and len(rand) > 0:
            imp_stoi = imp["STOI"].mean()
            rand_stoi = rand["STOI"].mean()
            delta = imp_stoi - rand_stoi
            report_lines.append(
                f"- **{net.replace('_',' ').title()}**: importance-aware STOI advantage "
                f"over random = {delta:+.4f}"
            )

    report_lines.extend([
        "",
        "## Plots\n",
        "- [STOI vs PLR](plots/stoi_vs_plr.png)",
        "- [SI-SDR vs PLR](plots/si_sdr_vs_plr.png)",
        "- [Post-Repair Loss](plots/post_repair_loss.png)",
    ])
    if has_diag:
        report_lines.append("- [Oracle Diagnostics](plots/oracle_diagnostics.png)")

    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report saved to {report_path}")
    print(f"Plots saved to {plots_dir}/")


NETWORK_TYPES = ["random_loss", "burst_loss", "jitter_discard"]
PROTECTION_METHODS = ["none", "random", "heuristic", "importance_aware"]


if __name__ == "__main__":
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
    )
    generate_report(results_dir)
