# Validation Report — The Reallocation Engine, Audited
### Effort Reallocation Engine: Job-Search Hours Across Companies

**Course:** INFO 7375 — Computational Skepticism for AI
**Domain:** People / Time — reallocating a student's limited weekly
job-search hour budget across a shortlist of target companies.

---

## Anchor to the Domain Text

This tool is anchored to four mechanisms from *The Reallocation Engine*
(Brown, 2026), each doing different work:

**Chapter 2, "The Reallocation Principle"** is the origin of the core
idea. It argues job-search effort should be allocated by *expected
return*, not habit or how legible the feedback feels — cold applications
convert at ~0.2% while 54% of hires flow through personal connections, so
a "disciplined" 40-hour spray-and-apply week is actually misallocated.
The book's default is the **3-3-2 day** (2 hrs targeted applying / 3 hrs
networking / 3 hrs portfolio). This tool operationalizes the same
principle one level down: instead of splitting hours across those three
channels, it splits the *targeted-applying* block across specific
companies by predicted return.

**Chapter 11, "The Bayesian Role Scorer,"** is the closest structural
match. The book's own composite is built from weighted **votes**
(sponsorship at 0.35, fit at 0.30) combined with multiplicative **gates**
(liveness, timeline) that can zero the score regardless of the votes —
and every term is labeled by source: *record*, *model judgment*, or
*your input*. Our engine mirrors this shape directly: the logistic
regression's features (`h1b_history`, `fit_score`, `wage_level`, etc.)
are the votes; the GIGO gate and the hard-stop are this project's
version of the book's gates; and `delegation_gate.py`'s required
`approver_note` on every commit is our implementation of the book's
**Override** mechanism, which explicitly requires "a documented reason
naming the private fact" rather than a silent override.

**Chapter 7, "Who Sponsors: The 80 Days Sponsorship Scorer,"** sharpens
Component 3 (Bias Audit). The book warns that a failed company-name
match can produce an "Unknown" tier indistinguishable from a true
"Avoid," and proposes a weighted composite (LCA filing rate × 0.40 +
approval rate × 0.30 + funding recency × 0.20 + company-size × 0.10)
instead of one boolean. Our bias audit independently found the identical
failure — our single `h1b_history` boolean can't separate "too new to
have filed" from "chose not to sponsor" — and our recommended fix is, in
effect, the book's own remedy. Disclosed as convergent validation, not
independently discovered novelty.

**Chapter 16, "The Build and the Honest Run,"** validates this project's
whole division of labor. Its "Give to the AI / Keep for yourself" table
— scaffolding and formula implementation to the AI, weight calibration
and plausibility audits to the human — is the same shape as our §7
delegation map. The chapter also recounts a real bug: an early draft of
the book's own scorer treated the *timeline* factor (meant to be a
zeroing gate) as a weighted vote, so a role that should have scored zero
instead scored "Consider" — caught only by a human plausibility audit,
not by the code running cleanly ("internally consistent, grounded in
nothing"). This is the same category of failure our adversarial test
(§6) surfaces: a single wrong `h1b_history` flag propagates through the
allocator without any code-level error, because nothing was "broken" in
the sense a test suite would catch — it required a human, not a script,
to notice the result didn't make sense.

---

## 1. The Working Reallocation Tool

**Objective (one sentence):** Allocate a fixed weekly hour budget across
companies to maximize the expected total number of callbacks, estimated
from a model fit on historical (effort_hours, outcome) pairs.

**What that objective leaves out:**
- Company desirability beyond callback odds — pay, growth, learning value.
  A student might rationally prefer fewer expected callbacks from a
  better company.
- Opportunity cost outside the job search (classes, rest, current job).
- Whether the effort→callback relationship is causal at all (see §5).

**Uncertainty:** every predicted probability carries a 90% bootstrap
confidence interval (`src/engine.py::bootstrap_predict`). See the chart
at `reports/uncertainty_chart.png` — **all 20 allocated companies in this
run have a CI wider than 0.4**, meaning the tool's rankings should be read
as rough prioritization, not precise probabilities. This is disclosed
prominently rather than hidden behind a single clean-looking number.

Run: `python3 src/engine.py` → `reports/hour_allocation.csv`,
`reports/predictions_with_uncertainty.csv`

---

## 2. Data Validation & the GIGO Gate

Dataset: `data/companies_synthetic.csv` (304 rows, **documented synthetic**
— see the docstring in `src/generate_data.py` for exactly what real-world
structure it does and doesn't capture; no real applicant outcome data
exists publicly, so this was the only feasible option in the time
available).

Gate checks (5 named hidden assumptions, each independently checkable):
1. No duplicate `company_id` (bad joins/re-scrapes)
2. `wage_level` present (no silent "assume average")
3. `funding_recent` flag not stale (>6 months old)
4. Effort hours / fit score within plausible bounds
5. Measurement-protocol consistency across the collection

**Result:** 291/304 passed (95.7%), 13 rejected — 4 duplicates, 8 missing
wage levels, 1 stale funding flag. Nothing was silently dropped; all
rejections are in `data/companies_rejected.csv` with a reason column.

Run: `python3 src/gigo_gate.py`

---

## 3. Bias Audit (data → output)

**Mechanism traced:** `h1b_history` is a proxy for company *age/size*
(enterprise/midsize firms have simply existed long enough to file LCAs;
startups mostly haven't, regardless of true willingness to sponsor). The
model rewards this proxy directly — it's the single largest coefficient
(0.76, see §4).

**Result — this is not hypothetical, it's what the model actually did:**

| company_size | mean allocated hours (unmitigated) |
|---|---|
| enterprise | 0.157 |
| midsize | 0.069 |
| **startup** | **0.000** |

Startups received **zero hours** in the baseline run.

**Two fairness definitions in tension:**
- *Demographic parity* (equal mean hours per size group) — violated, as above.
- *Equal opportunity* (equal mean hours among companies that WOULD produce
  a true-positive callback) — also violated (enterprise 0.27 vs. startup
  0.00) and would remain violated even after equalizing demographic parity,
  because the underlying predicted-probability distribution genuinely
  differs by group given the biased proxy feature.

**Chosen tradeoff:** we do not trust `h1b_history` as a clean causal
signal of "willingness to sponsor" (it's a proxy for age/size), so we
prioritize demographic parity via an explicit startup hour floor
(`bias_audit.py::apply_startup_floor`) even though this "spends" some
expected callbacks on companies the model rates lower. That cost is
disclosed, not hidden.

**Highest-leverage intervention point:** upstream of the model —
replace `h1b_history` (ever-sponsored, company-level boolean) with the
weighted composite Chapter 7 already specifies: LCA filing rate (3-yr) ×
0.40 + H-1B approval rate × 0.30 + funding recency × 0.20 + company-size
signal × 0.10. This decouples the feature from raw company age (a
company can score "Likely" on recent filings without needing decades of
history) and — critically — separates a true "Avoid" (evidence of
non-sponsorship) from an "Unknown" (no data yet, often from a failed
name-resolution join), which our binary flag cannot do.

Run: `python3 src/bias_audit.py`

---

## 4. Explainability & Its Critique

Global importance (logistic coefficients + permutation importance —
`shap`/`fairlearn` were unavailable, no internet access in the build
environment; permutation importance answers the equivalent question
model-agnostically):

| feature | coefficient | perm. importance |
|---|---|---|
| h1b_history | 0.760 | 0.052 |
| fit_score | 0.387 | 0.127 |
| wage_level | 0.350 | 0.026 |
| funding_recent | 0.194 | 0.001 |
| github_activity | 0.056 | 0.004 |
| effort_hours_historical | 0.023 | 0.001 |

**The critique (the actual point of this component):** for a well-funded,
high-GitHub-activity startup with `h1b_history=0` (predicted callback
probability 0.325), the explanation will flag "no H-1B history" as the
top negative driver — technically accurate about the *model*, but
practically misleading about the *world*: this company hasn't existed
long enough to file any LCA, not chosen not to. A student reading the
explanation at face value could wrongly conclude the company won't
sponsor, when the correct read is "no data yet, not a negative signal."

Run: `python3 src/explainability.py`

---

## 5. Causal & Counterfactual Reasoning — Pearl's Three Rungs

**Rung 1 (Observation):** raw correlation(effort_hours, callback) = 0.158.
Callback rate rises from 12% in the lowest effort quartile to 31% in the
highest.

**Rung 2 (Intervention):** adjusting for `fit_score` — the confounder
that plausibly drives BOTH voluntary effort and callback likelihood
directly — shrinks the effort coefficient by **79.7%** (0.228 → 0.046).
Most of the raw correlation is confounded, not a direct effect of hours.
Other named confounders: company_size/h1b_history (bigger companies get
more effort from every applicant AND more open headcount, independent of
any one student's hours), and an unmeasured one — referral/network
access, which this dataset does not capture at all.

**Rung 3 (Counterfactual):** for company #57 (actual: 0.25 hours invested,
real callback), the fitted model estimates only a 0.069 → 0.081 probability
shift for a hypothetical 8-hour investment — a small, uncertain effect,
resting on assumptions that do NOT hold in reality (e.g., that 8 hours of
research wouldn't also raise the student's own effective `fit_score`,
which it almost certainly would, and which this counterfactual doesn't
model).

**Honest verdict:** this engine reallocates hours based substantially on
a correlation that is mostly confounded by pre-existing fit. In THIS
synthetic dataset a small true causal effect of effort exists *by
construction* (we built it into the DGP), but in a real deployment, using
real historical application data, there would be no such guarantee — the
learned correlation would very likely be dominated by the fit_score
confound. **This tool should be read as a prioritization heuristic under
acknowledged confounding, not a validated causal estimate of what extra
hours will buy a student.**

Run: `python3 src/causal_analysis.py`

---

## 6. Adversarial Robustness & Fragility

**Perturbation:** flip a single company's `h1b_history` from False to
True — modeling a realistic error (a fuzzy-matched LCA record attributed
to the wrong subsidiary/parent company).

**Result:** the target company jumps from 0 → 1 allocated hour, and this
single flip **displaces two other companies entirely** from the
allocation to make room (companies #180 and #197 drop out). A human
skimming the same company's actual LCA filing would catch a wrong
subsidiary match in seconds. The engine's greedy allocator — optimizing
purely on marginal-gain-per-hour with no sanity check on data provenance
— cannot.

Run: `python3 src/adversarial_test.py`

---

## 7. Delegation Map + the Hard-Stop Gate

Following Chapter 16's "Give to the AI / Keep for yourself" framing, and
Chapter 11's requirement that every term be labeled by source:

| Step | Who decides | Source label | Override point |
|---|---|---|---|
| Company shortlist (role filter) | Tool | *record* (job posting text) | Student can add/remove any company |
| Callback probability estimate | Tool | *model judgment* | Student can override with private info (referral, inside knowledge) the model can't see — logged, per Chapter 11's Override rule |
| Hour allocation | Tool (recommendation) | *model judgment* | Student has final say; it's a default, not a directive |
| **Committing hours / submitting applications** | **Human, always** | *your input* | — this is the hard stop |
| Bias-mitigation floor | Tool applies by default | *model judgment* | Student can disable, but must do so explicitly and it's logged |

**Hard stop:** `src/delegation_gate.py::commit_allocation()` raises
`HardStopRequired` and refuses to write a committed plan unless called
with `human_approved=True` and an `approver_note` — this is this
project's implementation of Chapter 11's Override rule ("an override
without a documented reason is just ignoring the math because you like
the company"). Demonstrated: the script shows both the blocked attempt
and the successful, explicitly-approved commit.

Run: `python3 src/delegation_gate.py`

---

## Uncertainty Communication (summary)

- All 20 allocated companies in this run carry a 90% CI wider than 0.4 —
  see `reports/uncertainty_chart.png`.
- **Plain-language line for a non-specialist:** *"This tool tells you
  roughly which companies are worth more of your time, not a precise
  probability. Don't treat a 55% vs. 48% predicted callback chance as a
  meaningful difference — both numbers are riding on a wide, uncertain
  interval."*
- **Where I would not trust this tool:** any single company decision
  based on a probability near the 0.5 line with a wide interval, and any
  case where the student has private information (a referral, insider
  knowledge of a freeze) the model structurally cannot see.

---

## Known Limitations (stated plainly, not buried)

1. Fully synthetic outcome data — no real hiring outcomes were observed.
   Every causal/bias finding here demonstrates *method*, not a claim
   about real hiring.
2. Small sample (n≈300) by design — wide bootstrap intervals reflect this
   honestly rather than a bug to fix.
3. `effort_hours_historical` in the synthetic DGP has a real causal bump
   built in for demonstration purposes; a real dataset would need its own
   Rung 2/3 analysis, and might well show zero true causal effect.
