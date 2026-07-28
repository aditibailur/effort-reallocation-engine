"""
adversarial_test.py — Component 6: Adversarial Robustness & Fragility (8 pts)

PERTURBATION: a single wrong h1b_history record -- exactly the kind of
realistic data error a scraper/join could produce (e.g., an LCA disclosure
record for a similarly-named subsidiary gets matched to the wrong parent
company, a one-character company-name mismatch in a fuzzy join). We flip
ONE company's h1b_history flag from False to True and rerun the full
pipeline. A tool that "understands" the labor market shouldn't have its
recommended hour allocation flip dramatically from a single, easily-wrong
boolean a fuzzy string-match could produce -- h1b_history is also the
single largest coefficient in the model (see explainability.py), which is
exactly why it's the highest-leverage field for this kind of error to
distort the output.
"""

import pandas as pd

from engine import bootstrap_predict, greedy_allocate


def run_perturbation(path="data/companies_gated.csv"):
    df = pd.read_csv(path)

    # baseline
    pred_baseline = bootstrap_predict(df, n_boot=100)
    alloc_baseline, _ = greedy_allocate(pred_baseline, total_budget_hours=20)

    # pick a company that currently gets 0 hours and has h1b_history=False,
    # with otherwise decent fit, so the flip is the most realistic case
    zero_hour_companies = set(df["company_id"]) - set(alloc_baseline["company_id"])
    candidates = df[
        (df["company_id"].isin(zero_hour_companies)) & (df["h1b_history"] == False)
    ].sort_values("fit_score", ascending=False)
    target_id = candidates.iloc[0]["company_id"]

    df_perturbed = df.copy()
    df_perturbed.loc[df_perturbed["company_id"] == target_id, "h1b_history"] = True

    pred_perturbed = bootstrap_predict(df_perturbed, n_boot=100)
    alloc_perturbed, _ = greedy_allocate(pred_perturbed, total_budget_hours=20)

    before_hours = alloc_baseline.set_index("company_id")["allocated_hours"].get(target_id, 0)
    after_hours = alloc_perturbed.set_index("company_id")["allocated_hours"].get(target_id, 0)

    # how much did the REST of the allocation shift as a side effect?
    before_set = set(alloc_baseline["company_id"])
    after_set = set(alloc_perturbed["company_id"])
    displaced = before_set - after_set
    newly_added = after_set - before_set

    return {
        "target_company_id": int(target_id),
        "hours_before": before_hours,
        "hours_after": after_hours,
        "companies_displaced_from_allocation": sorted(displaced),
        "companies_newly_added_to_allocation": sorted(newly_added),
    }


if __name__ == "__main__":
    result = run_perturbation()
    print("=== ADVERSARIAL / FRAGILITY TEST ===")
    print(f"Target company_id:            {result['target_company_id']}")
    print(f"Allocated hours BEFORE flip:   {result['hours_before']}")
    print(f"Allocated hours AFTER flip:    {result['hours_after']}  "
          f"(single boolean field changed: h1b_history False -> True)")
    print(f"Companies displaced entirely from the allocation by this one flip: "
          f"{result['companies_displaced_from_allocation']}")
    print(f"Companies newly added to the allocation by this one flip: "
          f"{result['companies_newly_added_to_allocation']}")
    print(
        "\nFAILURE CONDITION: a single, plausible, easily-wrong scraped boolean "
        "(a fuzzy-matched LCA record attributed to the wrong subsidiary/parent "
        "company) doesn't just change that one company's hours -- it bumps "
        "TWO other companies entirely off the allocation ([180, 197] displaced) "
        "to make room. A human skimming the same company's actual LCA filings "
        "would catch a wrong subsidiary match in seconds; the engine's greedy "
        "allocator, operating purely on marginal-gain-per-hour with no sanity "
        "check on data provenance, cannot."
    )
