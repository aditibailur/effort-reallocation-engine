"""
generate_data.py — Synthetic company/application dataset for the
Effort Reallocation Engine.

WHY SYNTHETIC (documented per assignment requirement):
Building a clean, labeled dataset of *actual* application outcomes
(hours invested -> callback yes/no) does not exist publicly — no
student publishes a spreadsheet of "hours spent tailoring this resume"
next to "got a callback or not." Scraping live job boards + LCA data
+ actually tracking personal application outcomes over a semester was
not feasible in the time available. So this dataset is synthetic,
built from named, checkable assumptions grounded in real labor-market
structure (H-1B LCA wage levels, funding-stage signals, role-fit
scoring) rather than pulled from thin air.

WHAT REAL-WORLD STRUCTURE THIS DOES CAPTURE:
- Wage level distribution (I-IV) mirrors the real skew in DOL LCA data
  toward Level II/III for software/data roles.
- Funding-recency and company-size correlate with posting volume and
  perceived hiring urgency, consistent with general labor-market
  reporting.
- Effort has DIMINISHING RETURNS (sqrt-shaped), not linear returns —
  this is a deliberate, named assumption, not something "discovered."
- Fit score and effort are correlated by construction (students invest
  more time in companies where they already feel like a strong fit) —
  this is the confound the causal section (Rung 2/3) exists to expose.

WHAT IT DOES NOT CAPTURE (named limitations):
- No real individual outcome data. All "callback" labels are simulated
  from a hand-specified data-generating process (DGP), not observed
  reality. Any causal or fairness finding here is a demonstration of
  METHOD, not a claim about real hiring outcomes.
- No actual company names/postings — company_id is a synthetic key.
- No true unmeasured confounders beyond the ones we deliberately built
  in (real hiring managers have private information — network
  referrals, interviewer mood, headcount freezes — that this DGP does
  not model at all).
- Sample size (300) is small by design (assignment rewards a small,
  well-understood dataset over a large opaque one).
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)
N = 300

def generate():
    company_id = np.arange(1, N + 1)

    # --- company attributes ---
    company_size = RNG.choice(
        ["startup", "midsize", "enterprise"], size=N, p=[0.35, 0.35, 0.30]
    )

    # H-1B sponsorship history: enterprise & midsize skew TRUE more than startups,
    # mirroring real LCA disclosure patterns (large/established firms file more LCAs).
    h1b_prob = np.select(
        [company_size == "enterprise", company_size == "midsize", company_size == "startup"],
        [0.85, 0.55, 0.20],
    )
    h1b_history = RNG.binomial(1, h1b_prob).astype(bool)

    # Wage level (1-4), skewed toward II/III as in real DOL data
    wage_level = RNG.choice([1, 2, 3, 4], size=N, p=[0.15, 0.40, 0.35, 0.10])

    # Funding recency signal: startups/midsize more likely "recent funding",
    # enterprise treated as N/A -> represented as always "stable" (1)
    funding_recent = np.where(
        company_size == "enterprise", 1, RNG.binomial(1, 0.45, size=N)
    )

    # GitHub/ArXiv activity score (0-10), loosely tied to size + funding
    activity_base = np.select(
        [company_size == "enterprise", company_size == "midsize", company_size == "startup"],
        [6.5, 5.0, 4.0],
    )
    github_activity = np.clip(
        RNG.normal(activity_base + funding_recent * 1.2, 1.8, size=N), 0, 10
    )

    # --- applicant-side / fit ---
    fit_score = np.clip(RNG.normal(6.0, 1.8, size=N), 0, 10)  # 0-10, this student's skill/role fit

    # Effort invested (hours), CONFOUNDED with fit: better-fit companies get more
    # voluntary effort even before any "engine" recommends anything.
    effort_hours = np.clip(
        RNG.normal(1.0 + 0.6 * fit_score, 1.5, size=N), 0.25, 12
    )

    # --- data quality issues, deliberately injected for the GIGO gate to catch ---
    # 1) a few missing wage levels (simulating incomplete LCA disclosure joins)
    missing_idx = RNG.choice(N, size=8, replace=False)
    wage_level = wage_level.astype(float)
    wage_level[missing_idx] = np.nan
    # 2) a few duplicate company_id rows (simulating a bad join / scrape re-run)
    dup_idx = RNG.choice(N, size=4, replace=False)
    # 3) one stale funding record: a company flagged "funding_recent" from >18 months
    #    ago that should not still count as recent (measurement-protocol drift)
    stale_idx = RNG.choice(np.where(funding_recent == 1)[0], size=1, replace=False)

    # --- outcome: callback (ground-truth DGP, hidden from the "engine") ---
    # True causal structure (for our own validation / grading of the causal
    # section): callback probability is driven mostly by fit_score and
    # h1b_history and wage_level, with effort_hours having a SMALL true
    # causal bump (diminishing returns) PLUS a confounded correlation via
    # fit_score. The engine, using only observational data, cannot cleanly
    # separate these two paths.
    wl_filled = np.where(np.isnan(wage_level), 2, wage_level)
    true_effort_effect = 0.18 * np.sqrt(effort_hours)  # small true causal bump, diminishing returns
    logit = (
        -2.6
        + 0.50 * (fit_score - 6.0)
        + 0.85 * h1b_history.astype(float)
        + 0.22 * (wl_filled - 2)
        + true_effort_effect
        + 0.10 * (github_activity - 5)
    )
    prob_callback = 1 / (1 + np.exp(-logit))
    callback = RNG.binomial(1, prob_callback)

    df = pd.DataFrame({
        "company_id": company_id,
        "company_size": company_size,
        "h1b_history": h1b_history,
        "wage_level": wage_level,
        "funding_recent": funding_recent,
        "github_activity": np.round(github_activity, 2),
        "fit_score": np.round(fit_score, 2),
        "effort_hours_historical": np.round(effort_hours, 2),
        "callback": callback,
        "data_source": "synthetic_v1",
        "record_timestamp": "2026-07-01",
    })

    # inject duplicates (append copies of dup_idx rows)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    # funding recency freshness: most records are recent (0-6 months old),
    # one record is deliberately stale (19 months) to test the GIGO gate
    months_ago = RNG.integers(0, 6, size=len(df))
    months_ago[stale_idx] = 19
    df["funding_recent_asof_months_ago"] = months_ago

    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/companies_synthetic.csv", index=False)
    print(f"Wrote {len(df)} rows to data/companies_synthetic.csv")
    print(df.head())
    print("\nMissing wage_level:", df["wage_level"].isna().sum())
    print("Duplicate company_id rows:", df["company_id"].duplicated().sum())
