# AI Use Disclosure

**Tool(s) used:** Claude (Anthropic), via claude.ai chat + code execution
environment.

**Portions assisted:** The initial build — data generation, GIGO gate,
engine, bias audit, explainability, causal analysis, adversarial test,
and delegation gate — was drafted by Claude as Tier 1 execution work, per
the assignment's own framing. Claude also drafted the first version of
the validation report and located the relevant mechanisms in the domain
text (Chapters 2, 7, 11, and 16) to anchor the tool to.

**How used:** I described the domain (reallocating job-search time across
companies) and asked Claude to design and build a working Reallocation
Engine covering all 7 required components. No real dataset pairing hours
invested with callback outcomes exists publicly, so Claude proposed and
built a documented synthetic dataset instead, with the assumptions it
does and doesn't capture stated explicitly in `generate_data.py`.

**What I changed:** After the initial build, I ran the full pipeline
myself, confirmed it reproduced the reported findings on my own machine,
and then went further than the original build: Claude's adversarial test
(§6) only tested what happens when `h1b_history` is flipped. I asked
whether a different feature, `wage_level`, would be equally or more
fragile to a single bad data point, since it's the model's third-largest
coefficient and, in my view, varies more broadly across companies than
sponsorship history does. I made a genuine, timestamped prediction
before the test was run (I guessed ~4 companies would be displaced, more
than the 2 displaced by the h1b_history flip). I was wrong on every
axis — only 1 company was displaced, fewer, not more. I added this test
as a new file (`wage_level_perturbation_test.py`) and a new Frictional
Journal entry documenting the miscalibration, rather than letting the
original build stand as the final word on adversarial robustness.

**What the AI could not do:**

> Claude could report that `wage_level` has a smaller model coefficient
> than `h1b_history` (0.35 vs. 0.76) — that's a fact readable straight
> off the fitted model. What Claude could not do was tell me, in advance,
> that my own mental model for *why* a feature would be fragile was
> wrong. I was reasoning about how much a feature varies across the
> dataset; the thing that actually determines fragility is how heavily
> the model weights that feature, which is a different quantity
> entirely. Claude could hand me the coefficient table, but it couldn't
> hand me the correction to my intuition — I only found that out by
> committing to a prediction and testing it against the real output.
> That gap between "the AI can show you the number" and "the AI can fix
> your reasoning about the number before you test it yourself" is the
> irreducibly human part of this assignment, and it's the specific thing
> I'd point to if asked what I actually learned by building this.