"""
engine.py — Component 1: The Working Reallocation Tool (12 pts)

RESOURCE BEING REALLOCATED: the student's own limited weekly job-search
time budget (hours), across a shortlist of target companies.

OBJECTIVE (stated in one plain sentence, per the assignment's requirement):
"Allocate a fixed weekly hour budget across companies to maximize the
expected total number of callbacks, estimated from a model fit on
historical (effort_hours, outcome) pairs."

WHAT THAT OBJECTIVE LEAVES OUT (also required — name it):
- It says nothing about a company's TOTAL desirability (pay, growth,
  learning) — only callback probability. A student might rationally
  prefer to spend hours on a lower-probability but much better company.
- It does not model opportunity cost of hours spent elsewhere (classes,
  rest, existing job) — hours are treated as fungible and only scarce
  in the weekly-budget sense.
- It optimizes an OBSERVATIONAL relationship between effort and callback
  (see causal_analysis.py) — a claim this file does not by itself justify
  is causal. That is exactly what Component 5 exists to interrogate.

UNCERTAINTY: every predicted callback probability carries a bootstrap
confidence interval, and the final hour allocation is only as trustworthy
as that interval is narrow. This is surfaced, not hidden.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

RNG = np.random.default_rng(7)

FEATURES = [
    "h1b_history", "wage_level", "funding_recent",
    "github_activity", "fit_score", "effort_hours_historical",
]


def _prep(df):
    d = df.copy()
    d["h1b_history"] = d["h1b_history"].astype(int)
    return d


def fit_model(df):
    d = _prep(df)
    X = d[FEATURES].to_numpy()
    y = d["callback"].to_numpy()
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def bootstrap_predict(df, n_boot=200, budget_scenarios=None):
    """
    For each company, predict callback probability AT SEVERAL hypothetical
    effort levels (since we're deciding how many hours to invest, not just
    scoring the historical amount), with bootstrap confidence intervals.
    """
    d = _prep(df)
    if budget_scenarios is None:
        budget_scenarios = [1, 2, 4, 6, 8]

    boot_models = []
    for _ in range(n_boot):
        boot_df = resample(d, replace=True, n_samples=len(d), random_state=RNG.integers(1e9))
        if boot_df["callback"].nunique() < 2:
            continue
        boot_models.append(fit_model(boot_df))

    base_model = fit_model(d)

    rows = []
    for _, row in d.iterrows():
        for hrs in budget_scenarios:
            x = row[FEATURES].copy()
            x["effort_hours_historical"] = hrs
            x_arr = x.to_numpy().reshape(1, -1).astype(float)
            point = base_model.predict_proba(x_arr)[0, 1]
            boot_preds = [m.predict_proba(x_arr)[0, 1] for m in boot_models]
            lo, hi = np.percentile(boot_preds, [5, 95])
            rows.append({
                "company_id": row["company_id"],
                "company_size": row["company_size"],
                "hours": hrs,
                "pred_callback_prob": point,
                "ci_low": lo,
                "ci_high": hi,
            })
    return pd.DataFrame(rows)


def marginal_value(pred_df):
    """
    Marginal expected-callback gain per additional hour, at each hour level,
    per company — this is what the greedy allocator uses (diminishing
    returns means the 1st hour is worth more than the 8th).
    """
    pred_df = pred_df.sort_values(["company_id", "hours"])
    pred_df["marginal_gain"] = (
        pred_df.groupby("company_id")["pred_callback_prob"].diff().fillna(pred_df["pred_callback_prob"])
    )
    pred_df["marginal_hours"] = (
        pred_df.groupby("company_id")["hours"].diff().fillna(pred_df["hours"])
    )
    pred_df["gain_per_hour"] = pred_df["marginal_gain"] / pred_df["marginal_hours"]
    return pred_df


def greedy_allocate(pred_df, total_budget_hours=20, max_hours_per_company=8):
    """
    Greedy knapsack: repeatedly assign the next hour-block to whichever
    company currently offers the highest marginal gain-per-hour, until the
    weekly budget is exhausted or every company hits its per-company cap.
    """
    mv = marginal_value(pred_df).copy()
    allocated = {cid: 0 for cid in mv["company_id"].unique()}
    remaining_budget = total_budget_hours
    steps = sorted(mv["hours"].unique())
    step_size = steps[0]

    # build a lookup: (company_id, hours) -> gain_per_hour
    mv_idx = mv.set_index(["company_id", "hours"])

    allocation_log = []
    while remaining_budget > 0:
        best = None
        for cid in allocated:
            next_hours = allocated[cid] + step_size
            if next_hours > max_hours_per_company or next_hours not in steps:
                continue
            gph = mv_idx.loc[(cid, next_hours), "gain_per_hour"]
            if best is None or gph > best[1]:
                best = (cid, gph, next_hours)
        if best is None:
            break
        cid, gph, next_hours = best
        allocated[cid] = next_hours
        remaining_budget -= step_size
        allocation_log.append((cid, next_hours, gph))

    result = pd.DataFrame({"company_id": list(allocated.keys()), "allocated_hours": list(allocated.values())})
    result = result[result["allocated_hours"] > 0].sort_values("allocated_hours", ascending=False)
    return result, allocation_log


if __name__ == "__main__":
    df = pd.read_csv("data/companies_gated.csv")
    pred_df = bootstrap_predict(df)
    pred_df.to_csv("reports/predictions_with_uncertainty.csv", index=False)

    allocation, log = greedy_allocate(pred_df, total_budget_hours=20)
    merged = allocation.merge(
        df[["company_id", "company_size", "h1b_history", "fit_score"]], on="company_id"
    )
    merged.to_csv("reports/hour_allocation.csv", index=False)

    print("TOP ALLOCATION (20-hour weekly budget):")
    print(merged.to_string(index=False))

    # attach the CI at the final allocated hour level, for the uncertainty
    # communication requirement
    final_ci = pred_df.merge(
        allocation, left_on=["company_id", "hours"], right_on=["company_id", "allocated_hours"]
    )
    print("\nWith uncertainty (90% bootstrap interval on callback probability):")
    print(final_ci[["company_id", "allocated_hours", "pred_callback_prob", "ci_low", "ci_high"]].to_string(index=False))
