"""
delegation_gate.py — Component 7: Delegation Map + the Hard-Stop Gate (10 pts)

DELEGATION MAP
--------------
| Step                                   | Who decides           | Override point |
|-----------------------------------------|------------------------|-----------------|
| Company shortlist (role filter)         | TOOL                   | Student can add/remove any company manually before Stage 2 |
| Callback probability estimate           | TOOL                   | Student can flag a company where they have PRIVATE information the model can't see (referral, inside knowledge of a hiring freeze) -- this overrides the model's number entirely, no averaging |
| Hour allocation across companies        | TOOL (recommendation)  | Student has final say; recommendation is a default, not a directive |
| COMMITTING the week's hours / submitting applications | HUMAN, always | This is the resource-spending action -- see hard stop below |
| Bias-mitigation floor (startup minimum) | TOOL applies a default | Student can turn it off, but must do so explicitly and the tool logs that this was a deliberate choice, not a silent default |

Why the hard-stop line sits exactly there: everything above "committing
hours" is advisory -- a ranking, a probability, a suggested split. Once
a student actually spends the hours (or worse, submits an application),
that time is GONE -- it cannot be reallocated after the fact. That
irreversibility is exactly the "spends a resource" trigger the assignment
requires a hard stop for.

HARD STOP IMPLEMENTATION
-------------------------
The engine may recommend an allocation, but `commit_allocation()` below
will not execute (i.e., will not write a "committed" plan file) without
an explicit human approval flag. This mirrors what a real deployed tool
must do: recommend, then wait.
"""

import json
from datetime import datetime, timezone

import pandas as pd


class HardStopRequired(Exception):
    pass


def commit_allocation(allocation_df, human_approved: bool, approver_note: str = ""):
    """
    The ONLY function in this codebase allowed to write a 'committed' plan.
    Raises if human_approved is not explicitly True -- there is no default
    that lets this run unattended.
    """
    if not human_approved:
        raise HardStopRequired(
            "This allocation spends real hours (a committed resource). "
            "It cannot be committed without explicit human approval. "
            "Call commit_allocation(df, human_approved=True, approver_note=...) "
            "after you've reviewed the recommendation."
        )

    record = {
        "committed_at": datetime.now(timezone.utc).isoformat(),
        "human_approved": True,
        "approver_note": approver_note,
        "allocation": allocation_df.to_dict(orient="records"),
    }
    with open("reports/committed_allocation.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"Committed {len(allocation_df)} company allocations "
          f"({allocation_df['allocated_hours'].sum()} total hours) "
          f"with human approval note: '{approver_note}'")
    return record


if __name__ == "__main__":
    allocation = pd.read_csv("reports/hour_allocation.csv")

    print("=== Attempting to commit WITHOUT approval (should be blocked) ===")
    try:
        commit_allocation(allocation, human_approved=False)
    except HardStopRequired as e:
        print(f"BLOCKED as expected: {e}")

    print("\n=== Committing WITH explicit human approval ===")
    commit_allocation(
        allocation,
        human_approved=True,
        approver_note=(
            "Reviewed all 20 companies manually; company_id 234 (h1b_history=False) "
            "kept in the plan despite lower predicted probability because I have a "
            "personal referral there that the model has no way to see."
        ),
    )
