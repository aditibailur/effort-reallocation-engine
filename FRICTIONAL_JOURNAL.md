# Frictional Journal

## Prediction (made BEFORE running the wage_level perturbation test)

**Timestamp:** 2026-07-28, 8:30pm (recorded live, before the test was run)

**Test being predicted:** what happens if a single company's `wage_level`
is flipped from its lowest value (1) to its highest (4), instead of
flipping `h1b_history` as in the main adversarial test (§6 of the
validation report, which displaced 2 companies)?

1. **Do you expect this to displace more, fewer, or about the same number
   of companies as the h1b_history flip (2 companies)? Why?**

   "I guess more. I expect the `wage_level` flip to displace more than
   two companies because wage level directly affects how competitive or
   eligible a company appears. More companies are likely to differ in
   wage levels than in H-1B history, so changing this factor should have
   a broader effect on the results."

2. **Rough guess + confidence:**

   "My rough guess is that around 4 companies will be displaced. I'm
   about 65% confident, since `wage_level` will likely affect more
   companies than the `h1b_history` flip did."

3. **Where do you expect the effect to show up — a small ripple, or
   something as dramatic as the h1b_history flip?**

   "I expect a moderate ripple in the allocation, affecting a few
   companies, but nothing as dramatic as the `h1b_history` flip."

---

## Reflection (written AFTER seeing the actual result)
**Timestamp:** 2026-07-28, 9:00pm (recorded live, after the test was run)

**Actual result:** only **1 company displaced** ([180]) — fewer than the
h1b_history flip's 2, not more.

1. **What actually happened — where were you right/wrong?**

   "Only one company was displaced, fewer than the two displaced by the
   `h1b_history` flip. I incorrectly predicted about four companies and
   a moderate effect. I was wrong about both the direction and
   magnitude of the impact."

2. **What does that say about your calibration — overconfident,
   underconfident, or wrong about the mechanism entirely?**

   "I was overconfident and wrong about the underlying mechanism. I
   focused on how much `wage_level` varies across companies instead of
   its coefficient. Since its coefficient was only 0.35, compared with
   0.76 for `h1b_history`, the model gave it much less influence."

3. **What single finding here most changed how much you'd trust a tool
   like this?**

   "The most important finding was that a large change in a feature's
   raw value does not necessarily produce a large allocation change. I
   would trust the tool more when it provides explainability
   information, such as feature coefficients, but I would avoid relying
   on intuition alone without testing the actual effect."