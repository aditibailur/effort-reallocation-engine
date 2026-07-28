# Frictional Journal

**Important — this section is graded on genuine calibration, not on
being "right." Fill it in yourself, honestly, before you dig into
`reports/VALIDATION_REPORT.md` in detail. If you fill this in after
already knowing the results, it defeats the point (and a grader who
looks at your timestamp / phrasing will likely be able to tell).**

## Prediction (fill in BEFORE reading the validation report in detail)

**Timestamp:** _____________________ (fill in the real date/time)

1. What do you expect the hardest failure to be — the GIGO gate, the
   bias audit, the causal analysis, the adversarial test, or something
   else? Why?

   _[Your answer here]_

2. How causally valid do you expect the "more effort → more callbacks"
   relationship to turn out to be, once a confounder is controlled for?
   Give a rough number (e.g., "I expect the effect to shrink by about
   X% once fit_score is controlled for") and your confidence in that
   guess (as a number, e.g., "60% confident").

   _[Your answer here]_

3. Where do you expect this tool to be biased, and against whom?

   _[Your answer here]_

## Reflection (fill in AFTER reading the full validation report)

1. What actually happened? Where was your prediction right, and where
   was it wrong?

   _[Your answer here]_

2. If you were wrong about the causal shrinkage number or the bias
   finding, what does that say about your calibration going in? Were
   you overconfident, underconfident, or wrong about the mechanism
   entirely (e.g., predicted bias against a different group than the
   one actually found)?

   _[Your answer here]_

3. What is the one finding in this repo that most changed how much you'd
   trust a tool like this in real life?

   _[Your answer here]_
