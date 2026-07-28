"""
bias_audit.py — Component 3: Bias Audit, data -> output (10 pts)

QUESTION: Who is systematically advantaged or starved by this engine,
and where does that enter?

TRACED MECHANISM:
The bias enters at DATA COLLECTION, not the model math. h1b_history is
built (both in the real world and in our synthetic DGP) to correlate
with company_size: enterprise and midsize firms have long H-1B filing
histories almost by construction (they've existed longer, filed more
paperwork), while startups mostly haven't, REGARDLESS OF whether the
startup would happily sponsor a visa if asked. So h1b_history is a
proxy for "company age/size," not a direct measure of "willingness to
sponsor" — and the model learns to reward company age/size.

This becomes a FEEDBACK LOOP: the engine allocates more of the
student's hours to already-established companies, which (a) already
receive more attention from every other student using similar
heuristics, and (b) starves newer/smaller companies of applications
that might convert at similar or better rates if actually applied to
-- we cannot know, because they get almost no allocated effort to
generate outcome data from, in this dataset or in reality.

TWO FAIRNESS DEFINITIONS IN TENSION (required):

1. DEMOGRATIC PARITY across company_size groups: the average allocated
   hours should be equal across startup / midsize / enterprise groups,
   regardless of predicted callback probability.

2. EQUAL OPPORTUNITY (a.k.a. equalized true positive rates) across the
   same groups: among companies that WOULD have resulted in a callback
   (in our synthetic ground truth), the allocation should give equal
   average hours regardless of group.

These cannot both be satisfied at once here: if predicted callback
probability is genuinely lower on average for startups (whether from
real signal or from the h1b_history proxy bias), then equalizing
opportunity (rewarding true positives equally) will violate demographic
parity (it will keep allocating more hours to enterprise/midsize, where
more of the true positives happen to sit) -- and equalizing demographic
parity will "waste" hours on some startups with genuinely low fit,
diluting expected callbacks.

CHOSEN TRADEOFF (documented, not hidden): we prioritize demographic
parity by company_size for one specific reason -- the assignment's own
premise is that this tool is dangerous precisely because it can APPEAR
statistically justified while starving a group through a proxy
variable. Since h1b_history is a proxy for age/size and not directly
observed "willingness to sponsor," we do not trust equal-opportunity
optimization here; we deliberately reserve a minimum floor of hours for
startups even where predicted probability is lower, and disclose that
this costs some expected callbacks in exchange for not fully trusting a
biased proxy.

HIGHEST-LEVERAGE INTERVENTION POINT: upstream of the model entirely --
replace h1b_history (company-level, proxy for size) with the weighted
composite the domain text (The Reallocation Engine, Ch. 7, "Who Sponsors")
already specifies: LCA filing rate (3-yr) x 0.40 + H-1B approval rate x
0.30 + funding recency x 0.20 + company-size signal x 0.10. This decouples
the feature from raw company age/size and separates a true "Avoid"
(evidence of non-sponsorship) from an "Unknown" (no data yet, often from
a failed company-name-resolution join) -- a distinction our single
boolean cannot make, and one the book explicitly warns about.
"""

import numpy as np
import pandas as pd

from engine import bootstrap_predict, greedy_allocate


def demographic_parity(df, allocation):
    merged = allocation.merge(df[["company_id", "company_size"]], on="company_id", how="right")
    merged["allocated_hours"] = merged["allocated_hours"].fillna(0)
    return merged.groupby("company_size")["allocated_hours"].mean()


def equal_opportunity(df, allocation):
    merged = allocation.merge(
        df[["company_id", "company_size", "callback"]], on="company_id", how="right"
    )
    merged["allocated_hours"] = merged["allocated_hours"].fillna(0)
    true_positives = merged[merged["callback"] == 1]
    return true_positives.groupby("company_size")["allocated_hours"].mean()


def apply_startup_floor(allocation, df, floor_hours=1, total_budget=20):
    """
    Mitigation: reserve a minimum hour floor for startup-labeled companies
    even when their predicted probability is lower, capped so we don't
    blow the total weekly budget.
    """
    merged = allocation.merge(df[["company_id", "company_size"]], on="company_id", how="right")
    merged["allocated_hours"] = merged["allocated_hours"].fillna(0)
    startups = merged[merged["company_size"] == "startup"].sort_values(
        "allocated_hours"
    )
    n_needing_floor = (startups["allocated_hours"] < floor_hours).sum()
    if n_needing_floor == 0:
        return merged
    # simplistic reallocation: bump lowest-allocated startups to the floor,
    # funding it by trimming 1 hour from the largest current allocations
    idx_to_bump = startups[startups["allocated_hours"] < floor_hours].index[: n_needing_floor]
    merged.loc[idx_to_bump, "allocated_hours"] = floor_hours
    hours_needed = n_needing_floor * floor_hours - startups.loc[idx_to_bump, "allocated_hours"].sum()
    donors = merged.sort_values("allocated_hours", ascending=False).index
    donated = 0
    for d in donors:
        if donated >= hours_needed:
            break
        if merged.loc[d, "allocated_hours"] > floor_hours and d not in idx_to_bump:
            merged.loc[d, "allocated_hours"] -= 1
            donated += 1
    return merged


if __name__ == "__main__":
    df = pd.read_csv("data/companies_gated.csv")
    pred_df = bootstrap_predict(df)
    allocation, _ = greedy_allocate(pred_df, total_budget_hours=20)

    print("=== DEMOGRAPHIC PARITY (mean allocated hours by company_size) ===")
    dp = demographic_parity(df, allocation)
    print(dp)
    print(f"\nDemographic parity gap (max-min): {dp.max() - dp.min():.3f} hours")

    print("\n=== EQUAL OPPORTUNITY (mean hours among TRUE callback cases) ===")
    eo = equal_opportunity(df, allocation)
    print(eo)
    print(f"\nEqual-opportunity gap (max-min): {eo.max() - eo.min():.3f} hours")

    print("\n=== MITIGATION: startup hour floor applied ===")
    mitigated = apply_startup_floor(allocation, df, floor_hours=1, total_budget=20)
    dp_after = mitigated.groupby("company_size")["allocated_hours"].mean()
    print(dp_after)
    print(f"Demographic parity gap AFTER mitigation: {dp_after.max() - dp_after.min():.3f} hours")

    mitigated.to_csv("reports/allocation_bias_mitigated.csv", index=False)
