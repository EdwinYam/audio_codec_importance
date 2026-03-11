"""Generate report from experiment results."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METHODS = ["none", "random", "heuristic", "importance_aware", "importance_selective"]
MARKERS = {"none": "x", "random": "s", "heuristic": "^",
           "importance_aware": "o", "importance_selective": "D"}


def _plot_metric_vs_plr(summary, codecs, metric, ylabel, title, save_path,
                        methods=None):
    """Plot a metric vs PLR, one subplot per codec."""
    if methods is None:
        methods = METHODS
    n = len(codecs)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True, squeeze=False)
    for ax, codec in zip(axes[0], codecs):
        sub = summary[summary["codec"] == codec]
        for method in methods:
            m = sub[sub["protection_method"] == method]
            if len(m) == 0:
                continue
            label = method.replace("_", " ").title()
            marker = MARKERS.get(method, "o")
            ax.plot(m["target_plr"] * 100, m[metric],
                    marker=marker, label=label, linewidth=2)
        ax.set_xlabel("Target PLR (%)")
        ax.set_title(codec)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_metric_by_concealment(summary, codecs, metric, ylabel, title, save_path):
    """Plot a metric vs PLR, faceted by codec and concealment."""
    concealments = sorted(summary["concealment"].unique()) if "concealment" in summary.columns else ["zero_fill"]
    n_codecs = len(codecs)
    n_conc = len(concealments)
    fig, axes = plt.subplots(n_conc, n_codecs, figsize=(7 * n_codecs, 4 * n_conc),
                              sharey=True, squeeze=False)
    methods_in_data = [m for m in METHODS if m in summary["protection_method"].values]
    for row, conc in enumerate(concealments):
        for col, codec in enumerate(codecs):
            ax = axes[row][col]
            sub = summary[(summary["codec"] == codec) & (summary["concealment"] == conc)]
            for method in methods_in_data:
                m = sub[sub["protection_method"] == method]
                if len(m) == 0:
                    continue
                label = method.replace("_", " ").title()
                marker = MARKERS.get(method, "o")
                ax.plot(m["target_plr"] * 100, m[metric],
                        marker=marker, label=label, linewidth=2)
            ax.set_xlabel("Target PLR (%)")
            ax.set_title(f"{codec} / {conc}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        axes[row][0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_report(results_dir: str):
    """Read results CSV and produce report + plots."""
    csv_path = os.path.join(results_dir, "results.csv")
    diag_path = os.path.join(results_dir, "oracle_diagnostics.csv")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    has_diag = os.path.exists(diag_path)
    df_diag = pd.read_csv(diag_path) if has_diag else None

    # Detect if multi-codec
    has_codec_col = "codec" in df.columns
    if not has_codec_col:
        df["codec"] = "EnCodec_3kbps"
    codecs = sorted(df["codec"].unique())

    has_concealment = "concealment" in df.columns
    if not has_concealment:
        df["concealment"] = "zero_fill"

    # Average over seeds and files
    group_cols = ["codec", "network_type", "target_plr", "protection_method", "concealment"]
    metrics = ["PESQ", "PESQ_WB", "PESQ_NB", "STOI", "ESTOI", "SI-SDR",
               "post_repair_loss_rate", "concealment_rate", "mean_burst_len",
               "max_burst_len"]
    avail_metrics = [m for m in metrics if m in df.columns]
    summary = df.groupby(group_cols)[avail_metrics].mean().reset_index()

    # For plots that aggregate across concealments (backward compat)
    group_cols_no_conc = ["codec", "network_type", "target_plr", "protection_method"]
    summary_no_conc = df.groupby(group_cols_no_conc)[avail_metrics].mean().reset_index()

    # ─── Plots per metric ──────────────────────────────────────────
    _plot_metric_vs_plr(summary_no_conc, codecs, "PESQ", "PESQ (MOS-LQO)",
                        "PESQ vs Packet Loss Rate",
                        os.path.join(plots_dir, "pesq_vs_plr.png"))
    _plot_metric_vs_plr(summary_no_conc, codecs, "STOI", "STOI",
                        "STOI vs Packet Loss Rate",
                        os.path.join(plots_dir, "stoi_vs_plr.png"))
    _plot_metric_vs_plr(summary_no_conc, codecs, "SI-SDR", "SI-SDR (dB)",
                        "SI-SDR vs Packet Loss Rate",
                        os.path.join(plots_dir, "si_sdr_vs_plr.png"))

    # PESQ_NB plot
    if "PESQ_NB" in df.columns:
        _plot_metric_vs_plr(summary_no_conc, codecs, "PESQ_NB", "PESQ-NB (MOS-LQO)",
                            "PESQ Narrowband vs Packet Loss Rate",
                            os.path.join(plots_dir, "pesq_nb_vs_plr.png"))

    # Post-repair loss rate
    _plot_metric_vs_plr(summary_no_conc, codecs, "post_repair_loss_rate",
                        "Post-Repair Loss Rate",
                        "Post-Repair Loss Rate vs Target PLR",
                        os.path.join(plots_dir, "post_repair_loss.png"))

    # Concealment comparison plots
    if has_concealment and df["concealment"].nunique() > 1:
        for metric_name, ylabel in [("PESQ", "PESQ (MOS-LQO)"), ("STOI", "STOI"),
                                     ("SI-SDR", "SI-SDR (dB)")]:
            if metric_name in avail_metrics:
                _plot_metric_by_concealment(
                    summary, codecs, metric_name, ylabel,
                    f"{metric_name} by Concealment Method",
                    os.path.join(plots_dir, f"{metric_name.lower().replace('-','_')}_by_concealment.png"),
                )

    # ─── Oracle diagnostics plot ───────────────────────────────────
    if df_diag is not None and len(df_diag) > 0:
        has_codec_diag = "codec" in df_diag.columns
        if not has_codec_diag:
            df_diag["codec"] = "EnCodec_3kbps"
        diag_codecs = sorted(df_diag["codec"].unique())
        n = len(diag_codecs)
        fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n), squeeze=False)
        for row, codec in enumerate(diag_codecs):
            dsub = df_diag[df_diag["codec"] == codec]
            diag_summary = dsub.groupby("method")[["spearman_corr", "precision_at_20pct"]].mean()
            methods = diag_summary.index.tolist()
            x = range(len(methods))
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(methods)]

            axes[row][0].bar(x, diag_summary["spearman_corr"], color=colors)
            axes[row][0].set_xticks(list(x))
            axes[row][0].set_xticklabels(methods, rotation=45, ha="right")
            axes[row][0].set_ylabel("Spearman Correlation")
            axes[row][0].set_title(f"{codec} — Oracle Ranking Correlation")
            axes[row][0].grid(True, alpha=0.3, axis="y")

            axes[row][1].bar(x, diag_summary["precision_at_20pct"], color=colors)
            axes[row][1].set_xticks(list(x))
            axes[row][1].set_xticklabels(methods, rotation=45, ha="right")
            axes[row][1].set_ylabel("Precision @ 20%")
            axes[row][1].set_title(f"{codec} — Protected Frame Precision")
            axes[row][1].grid(True, alpha=0.3, axis="y")

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
        f"**Codecs**: {', '.join(codecs)}\n",
        f"**Seeds**: {sorted(df['seed'].unique())}\n",
        f"**Concealments**: {', '.join(sorted(df['concealment'].unique()))}\n",
        "",
        "## Experiment Configuration\n",
        f"- **Codecs**: {', '.join(codecs)}",
        f"- **Protection budget**: {df['budget_frac'].iloc[0]*100:.0f}% extra frames",
        f"- **Network types**: {', '.join(sorted(df['network_type'].unique()))}",
        f"- **PLRs**: {', '.join(str(int(p*100))+'%' for p in sorted(df['target_plr'].unique()))}",
        f"- **Protection methods**: {', '.join(sorted(df['protection_method'].unique()))}",
        f"- **Concealment methods**: {', '.join(sorted(df['concealment'].unique()))}",
        "",
    ]

    # Per-codec results, broken down by concealment method
    concealments = sorted(df["concealment"].unique()) if has_concealment else ["zero_fill"]
    for codec in codecs:
        report_lines.append(f"## {codec} Results (averaged over files, seeds)\n")
        codec_summary = summary[summary["codec"] == codec]

        for net in sorted(df["network_type"].unique()):
            report_lines.append(f"### {net.replace('_', ' ').title()}\n")
            net_sub = codec_summary[codec_summary["network_type"] == net]
            if len(net_sub) == 0:
                report_lines.append("_No data._\n")
                continue

            for conc in concealments:
                sub = net_sub[net_sub["concealment"] == conc]
                if len(sub) == 0:
                    continue
                conc_label = conc.replace("_", " ").title()
                report_lines.append(f"#### Concealment: {conc_label}\n")

                pivot_pesq = sub.pivot(index="target_plr", columns="protection_method", values="PESQ")
                report_lines.append("**PESQ WB (MOS-LQO):**\n")
                report_lines.append(pivot_pesq.round(3).to_markdown())
                report_lines.append("")

                if "PESQ_NB" in sub.columns:
                    pivot_pesq_nb = sub.pivot(index="target_plr", columns="protection_method", values="PESQ_NB")
                    report_lines.append("**PESQ NB (MOS-LQO):**\n")
                    report_lines.append(pivot_pesq_nb.round(3).to_markdown())
                    report_lines.append("")

                pivot_stoi = sub.pivot(index="target_plr", columns="protection_method", values="STOI")
                report_lines.append("**STOI:**\n")
                report_lines.append(pivot_stoi.round(4).to_markdown())
                report_lines.append("")

                pivot_sdr = sub.pivot(index="target_plr", columns="protection_method", values="SI-SDR")
                report_lines.append("**SI-SDR (dB):**\n")
                report_lines.append(pivot_sdr.round(2).to_markdown())
                report_lines.append("")

                pivot_loss = sub.pivot(index="target_plr", columns="protection_method", values="post_repair_loss_rate")
                report_lines.append("**Post-Repair Loss Rate:**\n")
                report_lines.append(pivot_loss.round(4).to_markdown())
                report_lines.append("")

    # Concealment comparison section
    if has_concealment and df["concealment"].nunique() > 1:
        report_lines.append("## Concealment Method Comparison\n")
        conc_summary = df.groupby(["codec", "concealment"])[avail_metrics].mean().reset_index()
        for codec in codecs:
            csub = conc_summary[conc_summary["codec"] == codec]
            report_lines.append(f"### {codec}\n")
            display_cols = ["concealment"] + [m for m in ["PESQ", "PESQ_NB", "STOI", "SI-SDR"] if m in csub.columns]
            report_lines.append(csub[display_cols].round(4).to_markdown(index=False))
            report_lines.append("")

    # Oracle diagnostics section
    if df_diag is not None and len(df_diag) > 0:
        report_lines.append("## Importance Method Diagnostics (vs Oracle)\n")
        for codec in codecs:
            if "codec" in df_diag.columns:
                dsub = df_diag[df_diag["codec"] == codec]
            else:
                dsub = df_diag
            if len(dsub) == 0:
                continue
            report_lines.append(f"### {codec}\n")
            diag_summary = dsub.groupby("method")[["spearman_corr", "precision_at_20pct"]].mean()
            report_lines.append(diag_summary.round(4).to_markdown())
            report_lines.append("")

    # Interpretation
    report_lines.append("## Interpretation\n")
    for codec in codecs:
        codec_summary = summary_no_conc[summary_no_conc["codec"] == codec]
        best_results = []
        for net in sorted(df["network_type"].unique()):
            for plr in sorted(df["target_plr"].unique()):
                sub = codec_summary[(codec_summary["network_type"] == net) &
                                     (codec_summary["target_plr"] == plr)]
                if len(sub) == 0:
                    continue
                best_stoi = sub.loc[sub["STOI"].idxmax()]
                best_results.append(best_stoi["protection_method"])

        avail_methods = sorted(df["protection_method"].unique())
        for method in avail_methods:
            wins = sum(1 for m in best_results if m == method)
            if wins > 0:
                report_lines.append(
                    f"- **{codec}**: {method.replace('_',' ').title()} wins "
                    f"{wins}/{len(best_results)} conditions (highest STOI)."
                )

    report_lines.extend([
        "",
        "## Plots\n",
        "- [PESQ vs PLR](plots/pesq_vs_plr.png)",
        "- [STOI vs PLR](plots/stoi_vs_plr.png)",
        "- [SI-SDR vs PLR](plots/si_sdr_vs_plr.png)",
        "- [Post-Repair Loss](plots/post_repair_loss.png)",
    ])
    if "PESQ_NB" in df.columns:
        report_lines.append("- [PESQ-NB vs PLR](plots/pesq_nb_vs_plr.png)")
    if has_concealment and df["concealment"].nunique() > 1:
        report_lines.append("- [PESQ by Concealment](plots/pesq_by_concealment.png)")
        report_lines.append("- [STOI by Concealment](plots/stoi_by_concealment.png)")
        report_lines.append("- [SI-SDR by Concealment](plots/si_sdr_by_concealment.png)")
    if has_diag:
        report_lines.append("- [Oracle Diagnostics](plots/oracle_diagnostics.png)")

    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report saved to {report_path}")
    print(f"Plots saved to {plots_dir}/")


if __name__ == "__main__":
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
    )
    generate_report(results_dir)
