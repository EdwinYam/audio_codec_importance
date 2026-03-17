"""Generate report + plots from streaming experiment results (v3)."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Display order and styling for threshold methods
METHOD_ORDER = [
    "no_duplicate", "fixed_0.3", "fixed_0.5", "fixed_0.7",
    "adaptive_mean_std", "adaptive_quantile_70", "all_duplicate",
]
METHOD_LABELS = {
    "no_duplicate": "No Dup (baseline)",
    "fixed_0.3": "Fixed 0.3",
    "fixed_0.5": "Fixed 0.5",
    "fixed_0.7": "Fixed 0.7",
    "adaptive_mean_std": "Adaptive μ+σ",
    "adaptive_quantile_70": "Adaptive Q70",
    "all_duplicate": "All Dup (upper)",
}
METHOD_MARKERS = {
    "no_duplicate": "x", "fixed_0.3": "s", "fixed_0.5": "D",
    "fixed_0.7": "^", "adaptive_mean_std": "o",
    "adaptive_quantile_70": "P", "all_duplicate": "*",
}
METHOD_COLORS = {
    "no_duplicate": "#999999", "fixed_0.3": "#1f77b4", "fixed_0.5": "#ff7f0e",
    "fixed_0.7": "#2ca02c", "adaptive_mean_std": "#d62728",
    "adaptive_quantile_70": "#9467bd", "all_duplicate": "#333333",
}


def _get_methods_in_data(df):
    """Return methods in display order, filtered to those present in data."""
    present = set(df["threshold_method"].unique())
    return [m for m in METHOD_ORDER if m in present]


def _plot_metric_vs_plr(summary, metric, ylabel, title, save_path, methods):
    """Plot metric vs PLR, one subplot per duplication delay D."""
    delays = sorted(summary["duplication_delay"].unique())
    n = len(delays)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], delays):
        sub = summary[summary["duplication_delay"] == d]
        for method in methods:
            m = sub[sub["threshold_method"] == method]
            if len(m) == 0:
                continue
            ax.plot(m["target_plr"] * 100, m[metric],
                    marker=METHOD_MARKERS.get(method, "o"),
                    color=METHOD_COLORS.get(method),
                    label=METHOD_LABELS.get(method, method),
                    linewidth=2, markersize=6)
        ax.set_xlabel("Target PLR (%)")
        ax.set_title(f"D={d}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_bitrate_vs_plr(summary, save_path, methods):
    """Plot total bitrate vs PLR showing overhead cost."""
    delays = sorted(summary["duplication_delay"].unique())
    n = len(delays)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], delays):
        sub = summary[summary["duplication_delay"] == d]
        for method in methods:
            m = sub[sub["threshold_method"] == method]
            if len(m) == 0:
                continue
            ax.plot(m["target_plr"] * 100, m["total_bitrate_kbps"],
                    marker=METHOD_MARKERS.get(method, "o"),
                    color=METHOD_COLORS.get(method),
                    label=METHOD_LABELS.get(method, method),
                    linewidth=2, markersize=6)
        ax.set_xlabel("Target PLR (%)")
        ax.set_title(f"D={d}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("Total Bitrate (kbps)")
    fig.suptitle("Total Bitrate vs PLR (including duplication overhead)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_duplication_rate_bar(summary, save_path, methods):
    """Bar chart of average duplication rate per method."""
    avg = summary.groupby("threshold_method")["duplication_rate"].mean()
    avg = avg.reindex([m for m in methods if m in avg.index])
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [METHOD_COLORS.get(m, "#666") for m in avg.index]
    labels = [METHOD_LABELS.get(m, m) for m in avg.index]
    ax.bar(labels, avg.values * 100, color=colors)
    ax.set_ylabel("Duplication Rate (%)")
    ax.set_title("Average Duplication Rate by Method")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_quality_vs_overhead(summary, metric, ylabel, title, save_path, methods):
    """Scatter plot of quality vs bandwidth overhead (Pareto front)."""
    # Use only non-zero PLR data for meaningful comparison
    sub = summary[summary["target_plr"] > 0]
    if len(sub) == 0:
        return
    avg = sub.groupby("threshold_method").agg({
        metric: "mean", "bandwidth_overhead_pct": "mean"
    }).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in avg.iterrows():
        m = row["threshold_method"]
        if m not in methods:
            continue
        ax.scatter(row["bandwidth_overhead_pct"], row[metric],
                   marker=METHOD_MARKERS.get(m, "o"),
                   color=METHOD_COLORS.get(m),
                   s=120, zorder=5,
                   label=METHOD_LABELS.get(m, m))
    ax.set_xlabel("Bandwidth Overhead (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_report(results_dir: str):
    """Read results CSV and produce report + plots."""
    csv_path = os.path.join(results_dir, "results.csv")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    methods = _get_methods_in_data(df)

    # Average over seeds and files
    group_cols = ["target_plr", "threshold_method", "duplication_delay"]
    metric_cols = [c for c in [
        "PESQ", "PESQ_WB", "PESQ_NB", "STOI", "ESTOI", "SI-SDR",
        "post_recovery_loss_rate", "duplication_rate", "recovery_rate",
        "total_bitrate_kbps", "bandwidth_overhead_pct",
        "concealment_rate", "n_lost_frames", "mean_burst_len",
        "max_burst_len", "n_bursts", "n_recovered", "n_concealed",
    ] if c in df.columns]
    summary = df.groupby(group_cols)[metric_cols].mean().reset_index()

    # ─── Plots ────────────────────────────────────────────────────
    for metric, ylabel, fname in [
        ("PESQ_WB", "PESQ-WB (MOS-LQO)", "pesq_wb_vs_plr.png"),
        ("PESQ_NB", "PESQ-NB (MOS-LQO)", "pesq_nb_vs_plr.png"),
        ("STOI", "STOI", "stoi_vs_plr.png"),
        ("ESTOI", "ESTOI", "estoi_vs_plr.png"),
        ("SI-SDR", "SI-SDR (dB)", "si_sdr_vs_plr.png"),
        ("recovery_rate", "Recovery Rate", "recovery_rate_vs_plr.png"),
    ]:
        if metric in summary.columns:
            _plot_metric_vs_plr(
                summary, metric, ylabel,
                f"{ylabel} vs Packet Loss Rate",
                os.path.join(plots_dir, fname), methods,
            )

    _plot_bitrate_vs_plr(summary, os.path.join(plots_dir, "bitrate_vs_plr.png"), methods)
    _plot_duplication_rate_bar(summary, os.path.join(plots_dir, "duplication_rate_bar.png"), methods)

    for metric, ylabel in [("PESQ_WB", "PESQ-WB"), ("STOI", "STOI")]:
        if metric in summary.columns:
            _plot_quality_vs_overhead(
                summary, metric, ylabel,
                f"{ylabel} vs Bandwidth Overhead (Pareto)",
                os.path.join(plots_dir, f"{metric.lower().replace('-','_')}_vs_overhead.png"),
                methods,
            )

    # ─── Markdown Report ──────────────────────────────────────────
    delays = sorted(int(d) for d in df["duplication_delay"].unique())
    seeds = sorted(int(s) for s in df["seed"].unique())
    report = [
        "# Streaming-Compatible Importance-Aware Duplication: Experiment Report (v3)\n",
        f"**Generated from**: `{csv_path}`\n",
        f"**Total trials**: {len(df)}\n",
        f"**Audio files**: {df['file'].nunique()}\n",
        f"**Codec**: {', '.join(sorted(df['codec'].unique()))}\n",
        f"**Seeds**: {seeds}\n",
        f"**Duplication delays (D)**: {delays}\n",
        "",
        "## Key Concepts\n",
        "**Duplication delay D**: When a frame is marked as important, its duplicate",
        "is piggybacked onto a future transmission slot. D controls how far ahead:",
        "- **D=1**: duplicate of frame *t* rides with slot *t+1*",
        "- **D=2**: duplicate of frame *t* rides with slot *t+2* (better burst-loss decorrelation, more latency)\n",
        "If slot *t* is lost but slot *t+D* arrives, the receiver recovers frame *t* from the duplicate.",
        "If both are lost, the frame is concealed with `neighbor_copy` (repeat last received frame).\n",
        "",
        "## Experiment Configuration\n",
        f"- **Codec**: HILCodec_3kbps (base rate {df['base_bitrate_kbps'].iloc[0]:.1f} kbps)",
        f"- **PLRs**: {', '.join(str(int(p*100))+'%' for p in sorted(df['target_plr'].unique()))}",
        f"- **Threshold methods**: {', '.join(methods)}",
        f"- **Duplication delays D**: {delays}",
        f"- **Seeds**: {seeds}",
        "- **Concealment**: neighbor_copy (for unrecoverable frames)",
        "",
    ]

    # Tables per D value
    for d in delays:
        report.append(f"## Results for D={d} (averaged over files, seeds)\n")
        dsub = summary[summary["duplication_delay"] == d]

        for metric, label, fmt in [
            ("PESQ_WB", "PESQ-WB (MOS-LQO)", 3),
            ("PESQ_NB", "PESQ-NB (MOS-LQO)", 3),
            ("STOI", "STOI", 4),
            ("ESTOI", "ESTOI", 4),
            ("SI-SDR", "SI-SDR (dB)", 2),
            ("post_recovery_loss_rate", "Post-Recovery Loss Rate", 4),
            ("total_bitrate_kbps", "Total Bitrate (kbps)", 2),
            ("recovery_rate", "Recovery Rate", 4),
        ]:
            if metric not in dsub.columns:
                continue
            pivot = dsub.pivot(
                index="target_plr", columns="threshold_method", values=metric
            )
            # Reorder columns
            ordered_cols = [m for m in methods if m in pivot.columns]
            pivot = pivot[ordered_cols]
            pivot.columns = [METHOD_LABELS.get(c, c) for c in pivot.columns]
            report.append(f"**{label}:**\n")
            report.append(pivot.round(fmt).to_markdown())
            report.append("")

    # Bandwidth overhead summary
    report.append("## Bandwidth Overhead Summary\n")
    bw = df.groupby("threshold_method")["duplication_rate"].mean()
    bw = bw.reindex([m for m in methods if m in bw.index])
    report.append("| Method | Avg Duplication Rate | Overhead (%) | Total Bitrate (kbps) |")
    report.append("|--------|---------------------|-------------|---------------------|")
    for m in bw.index:
        rate = bw[m]
        report.append(
            f"| {METHOD_LABELS.get(m, m)} | {rate:.3f} | {rate*100:.1f}% | "
            f"{3.0*(1+rate):.2f} |"
        )
    report.append("")

    # Interpretation
    report.append("## Interpretation\n")
    # Find best method per PLR (by PESQ_WB) for each D
    for d in delays:
        report.append(f"### D={d}\n")
        dsub = summary[summary["duplication_delay"] == d]
        for plr in sorted(dsub["target_plr"].unique()):
            psub = dsub[dsub["target_plr"] == plr]
            if "PESQ_WB" in psub.columns and len(psub) > 0:
                best = psub.loc[psub["PESQ_WB"].idxmax()]
                m = best["threshold_method"]
                label = METHOD_LABELS.get(m, m)
                report.append(
                    f"- PLR {plr*100:.0f}%: Best PESQ-WB = {best['PESQ_WB']:.3f} "
                    f"({label})"
                )
        report.append("")

    # Plot links
    report.extend([
        "## Plots\n",
        "- [PESQ-WB vs PLR](plots/pesq_wb_vs_plr.png)",
        "- [PESQ-NB vs PLR](plots/pesq_nb_vs_plr.png)",
        "- [STOI vs PLR](plots/stoi_vs_plr.png)",
        "- [ESTOI vs PLR](plots/estoi_vs_plr.png)",
        "- [SI-SDR vs PLR](plots/si_sdr_vs_plr.png)",
        "- [Recovery Rate vs PLR](plots/recovery_rate_vs_plr.png)",
        "- [Total Bitrate vs PLR](plots/bitrate_vs_plr.png)",
        "- [Duplication Rate Bar Chart](plots/duplication_rate_bar.png)",
        "- [PESQ-WB vs Overhead (Pareto)](plots/pesq_wb_vs_overhead.png)",
        "- [STOI vs Overhead (Pareto)](plots/stoi_vs_overhead.png)",
    ])

    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))

    print(f"Report saved to {report_path}")
    print(f"Plots saved to {plots_dir}/")


if __name__ == "__main__":
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results_hilcodec_v3",
    )
    generate_report(results_dir)
