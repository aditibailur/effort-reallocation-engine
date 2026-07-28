# AI Use Disclosure

**Tool(s) used:** Claude (Anthropic), via claude.ai chat + code execution
environment.

**Portions assisted:** Nearly all of the initial code (data generation,
GIGO gate, engine, bias audit, explainability, causal analysis,
adversarial test, delegation gate) was drafted by Claude, delegated as
Tier 1 build work per the assignment's own framing. The validation
report was also drafted by Claude, synthesizing the code's output.

**How used:** I described the domain (reallocating job-search time
across companies) and asked Claude to design and build a working
Reallocation Engine covering all 7 required components. Claude proposed
the synthetic-data approach (no internet access was available to pull
real LCA/job-posting data at the volume needed, and no real
effort-hours-to-callback dataset exists publicly), wrote the
data-generating process, the model, the fairness/causal/adversarial
analyses, and the report.

**What I changed:**
_[Fill this in yourself — this field needs to be genuine. Actually run
`main.py`, actually read `reports/VALIDATION_REPORT.md` end to end, and
note anything you adjusted: a synthetic-data assumption you thought was
unrealistic and fixed, a bias-mitigation choice you disagreed with, a
causal-analysis interpretation you'd phrase differently, a company-size
threshold you changed. If you make no changes at all, say that
explicitly rather than leaving this blank — but for a 100-point
assignment where the validation work IS the grade, you should expect to
find and fix at least one real thing.]_

**What the AI could not do:**
_[This field must name one SPECIFIC, concrete Tier-4/5 judgment call —
not a category claim like "AI can't understand context." Below is a
draft candidate; replace it with your own if you find a better one, or
add a second if you have one:]_

> The synthetic data-generating process encodes `h1b_history` as a proxy
> for company age/size and reports that this produces a "zero hours to
> startups" bias — but Claude cannot know whether that specific
> assumption (that H-1B history mostly reflects age rather than genuine
> unwillingness to sponsor) is actually true for the real companies I'm
> targeting. I know, from having actually read individual companies'
> career pages and Glassdoor/Blind threads, that some small AI startups
> explicitly refuse to sponsor as a stated policy regardless of age —
> which means the "fix" of a blanket startup floor (spend hours on
> startups regardless of predicted probability) could waste real hours
> on companies that would never sponsor me specifically, for reasons the
> model and the mitigation both miss. Knowing which specific startups on
> my real shortlist fall into that category is domain knowledge Claude
> has no access to and I have to supply by hand before trusting the
> mitigated allocation.
