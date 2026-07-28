"""
uncertainty_viz.py — Uncertainty Communication (5 pts, woven through
Components 1, 4, and the video)

Produces a plain chart showing predicted callback probability WITH its
90% bootstrap confidence interval for the allocated companies, so a
non-specialist can see at a glance which recommendations are confident
vs. which are noisy guesses riding on a thin interval.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_uncertainty(pred_path="reports/predictions_with_uncertainty.csv",
                      alloc_path="reports/hour_allocation.csv",
                      out_path="reports/uncertainty_chart.png"):
    pred = pd.read_csv(pred_path)
    alloc = pd.read_csv(alloc_path)

    merged = pred.merge(
        alloc, left_on=["company_id", "hours"], right_on=["company_id", "allocated_hours"]
    ).sort_values("pred_callback_prob", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = range(len(merged))
    ax.errorbar(
        merged["pred_callback_prob"], y_pos,
        xerr=[
            merged["pred_callback_prob"] - merged["ci_low"],
            merged["ci_high"] - merged["pred_callback_prob"],
        ],
        fmt="o", color="#2c6e9b", ecolor="#a8c6dc", elinewidth=2, capsize=3,
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"Company {cid}" for cid in merged["company_id"]], fontsize=8)
    ax.set_xlabel("Predicted callback probability (point estimate, 90% bootstrap CI)")
    ax.set_title("Recommended allocation: predicted callback probability ± uncertainty")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved uncertainty chart to {out_path}")

    # plain-language summary line
    wide_ci = merged[(merged["ci_high"] - merged["ci_low"]) > 0.4]
    print(
        f"\nPlain-language read: {len(wide_ci)} of {len(merged)} allocated companies "
        f"have a confidence interval wider than 0.4 -- for those, treat the "
        f"recommendation as a rough prioritization signal, not a guarantee. "
        f"Do not read a 55% vs 48% predicted probability as a meaningful "
        f"difference when both intervals span 30+ points."
    )


if __name__ == "__main__":
    plot_uncertainty()
