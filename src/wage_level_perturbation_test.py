"""
wage_level_perturbation_test.py — Frictional Journal test case.

A second, independent adversarial test: instead of flipping h1b_history
(coefficient 0.76, the model's #1 feature), this flips wage_level
(coefficient 0.35, the model's #3 feature) from its lowest value (1) to
its highest (4) for a single company currently receiving 0 allocated
hours -- testing whether the fragility found in adversarial_test.py is
specific to the sponsorship-history feature or a general property of the
greedy allocator.
"""

import pandas as pd

from engine import bootstrap_predict, greedy_allocate


def run_wage_perturbation(path="data/companies_gated.csv"):
    df = pd.read_csv(path)

    pred_baseline = bootstrap_predict(df, n_boot=100)
    alloc_baseline, _ = greedy_allocate(pred_baseline, total_budget_hours=20)

    zero_hour_companies = set(df["company_id"]) - set(alloc_baseline["company_id"])
    candidates = df[
        (df["company_id"].isin(zero_hour_companies)) & (df["wage_level"] == 1)
    ].sort_values("fit_score", ascending=False)
    target_id = candidates.iloc[0]["company_id"]

    df_perturbed = df.copy()
    df_perturbed.loc[df_perturbed["company_id"] == target_id, "wage_level"] = 4

    pred_perturbed = bootstrap_predict(df_perturbed, n_boot=100)
    alloc_perturbed, _ = greedy_allocate(pred_perturbed, total_budget_hours=20)

    before_hours = alloc_baseline.set_index("company_id")["allocated_hours"].get(target_id, 0)
    after_hours = alloc_perturbed.set_index("company_id")["allocated_hours"].get(target_id, 0)

    before_set = set(alloc_baseline["company_id"])
    after_set = set(alloc_perturbed["company_id"])
    displaced = before_set - after_set
    newly_added = after_set - before_set

    return {
        "target_company_id": int(target_id),
        "hours_before": before_hours,
        "hours_after": after_hours,
        "companies_displaced": sorted(displaced),
        "companies_newly_added": sorted(newly_added),
    }


if __name__ == "__main__":
    result = run_wage_perturbation()
    print("=== WAGE_LEVEL PERTURBATION TEST (Frictional Journal test case) ===")
    print(f"Target company_id:          {result['target_company_id']}")
    print(f"Allocated hours BEFORE:      {result['hours_before']}")
    print(f"Allocated hours AFTER:       {result['hours_after']}  "
          f"(wage_level flipped: 1 -> 4)")
    print(f"Companies displaced:         {result['companies_displaced']} "
          f"({len(result['companies_displaced'])} total)")
    print(f"Companies newly added:       {result['companies_newly_added']}")
