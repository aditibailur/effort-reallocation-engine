"""
gigo_gate.py — Data Validation & the GIGO Gate (Component 2, 10 pts)

Runs a checkable quality standard against the raw dataset BEFORE the
reallocation tool is allowed to run. Named hidden assumptions this gate
is designed to catch:

  A1. Every record has a unique company_id (no duplicate joins/scrapes).
  A2. wage_level is present (a missing value silently defaults to "average"
      in a naive pipeline — that's a hidden assumption we refuse to make).
  A3. funding_recent flag reflects data collected within a defined recency
      window (6 months); anything older is stale and must be re-flagged,
      not trusted as "recent."
  A4. effort_hours_historical and fit_score are within plausible bounds
      (no negative hours, no fit scores outside 0-10).
  A5. The measurement protocol didn't silently change mid-collection —
      here, checked via the record_timestamp / data_source being uniform
      (a proxy for "same collection process for every row").

Any row that fails is NOT silently dropped — it's routed to a
`rejected` table with a reason, and the gate reports what fraction of
the dataset failed and why, before anything downstream runs.
"""

import pandas as pd
import numpy as np

RECENCY_WINDOW_MONTHS = 6


def run_gate(path="data/companies_synthetic.csv"):
    df = pd.read_csv(path)
    n_total = len(df)
    reasons = []

    # A1: duplicate company_id
    dup_mask = df.duplicated(subset="company_id", keep="first")
    reasons.append(("duplicate_company_id", dup_mask))

    # A2: missing wage_level
    missing_wage = df["wage_level"].isna()
    reasons.append(("missing_wage_level", missing_wage))

    # A3: stale funding_recent flag (claims recent but is older than window)
    stale_funding = (df["funding_recent"] == 1) & (
        df["funding_recent_asof_months_ago"] > RECENCY_WINDOW_MONTHS
    )
    reasons.append(("stale_funding_recent_flag", stale_funding))

    # A4: implausible values
    bad_effort = (df["effort_hours_historical"] <= 0) | (df["effort_hours_historical"] > 40)
    bad_fit = (df["fit_score"] < 0) | (df["fit_score"] > 10)
    reasons.append(("implausible_effort_hours", bad_effort))
    reasons.append(("implausible_fit_score", bad_fit))

    # A5: measurement-protocol consistency (single data_source/timestamp expected)
    protocol_drift = (
        (df["data_source"] != df["data_source"].mode()[0])
        | (df["record_timestamp"] != df["record_timestamp"].mode()[0])
    )
    reasons.append(("measurement_protocol_drift", protocol_drift))

    # combine
    fail_mask = np.zeros(n_total, dtype=bool)
    reason_col = pd.Series([""] * n_total)
    for name, mask in reasons:
        mask = mask.fillna(False).to_numpy()
        newly_failed = mask & ~fail_mask
        reason_col.loc[newly_failed] = name  # first reason wins, for clarity
        fail_mask = fail_mask | mask

    df["gigo_fail_reason"] = reason_col
    rejected = df[fail_mask].copy()
    passed = df[~fail_mask].copy()

    print(f"GIGO GATE REPORT — {path}")
    print(f"  Total records:      {n_total}")
    print(f"  Passed gate:        {len(passed)} ({len(passed)/n_total:.1%})")
    print(f"  Rejected:           {len(rejected)} ({len(rejected)/n_total:.1%})")
    print("  Rejection breakdown:")
    for name, _ in reasons:
        cnt = (rejected["gigo_fail_reason"] == name).sum()
        if cnt:
            print(f"    - {name}: {cnt}")

    passed.drop(columns=["gigo_fail_reason"]).to_csv(
        "data/companies_gated.csv", index=False
    )
    rejected.to_csv("data/companies_rejected.csv", index=False)
    return passed, rejected


if __name__ == "__main__":
    run_gate()
