# The Reallocation Engine, Audited — Effort Reallocation Engine

**Resource reallocated:** a student's limited weekly job-search hour
budget, across a shortlist of target companies.

**One-sentence objective:** allocate a fixed weekly hour budget across
companies to maximize expected total callbacks, estimated from a model
fit on historical (effort_hours, outcome) data — while surfacing where
that objective, and the causal story behind it, should not be trusted.

See `reports/VALIDATION_REPORT.md` for the full writeup of all 7 graded
components (working tool, GIGO gate, bias audit, explainability +
critique, causal/counterfactual reasoning, adversarial robustness,
delegation map + hard-stop gate) and the uncertainty communication.

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib
python3 main.py
```

This runs the full pipeline end-to-end in order:
1. `src/generate_data.py` — builds the documented synthetic dataset
2. `src/gigo_gate.py` — validates it, rejects what fails
3. `src/engine.py` — fits the model, produces the hour allocation with
   bootstrapped uncertainty
4. `src/bias_audit.py` — fairness metrics + mitigation
5. `src/explainability.py` — feature importance + the misleading-case critique
6. `src/causal_analysis.py` — Pearl's three rungs
7. `src/adversarial_test.py` — the fragility/perturbation test
8. `src/delegation_gate.py` — demonstrates the hard-stop gate (blocked,
   then explicitly approved)

Then, optionally:
```bash
python3 src/uncertainty_viz.py   # writes reports/uncertainty_chart.png
```

## Repo structure

```
data/            synthetic dataset + gate outputs (gated/rejected)
src/             all pipeline code, one file per component
reports/         VALIDATION_REPORT.md, all CSV/JSON/PNG outputs
main.py          runs everything end-to-end
```

## Why synthetic data

No public dataset pairs "hours a student invested tailoring an
application" with "callback yes/no" — that data doesn't exist outside
individual students' private tracking, and building + validating a
real one in the time available wasn't feasible. `src/generate_data.py`'s
docstring states explicitly what real-world structure the synthetic data
does and doesn't capture. Every finding in this repo should be read as a
demonstration of *method*, not a claim about real hiring outcomes.

## AI Use Disclosure

See `AI_USE_DISCLOSURE.md`.

## Frictional Journal

See `FRICTIONAL_JOURNAL.md` — **fill in the prediction section before
you read the rest of this repo's results, timestamp it, then fill in the
reflection after.**
