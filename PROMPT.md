# Ralph Autonomous Loop: NFL Big Data Bowl 2027 Continuous Improvement

## Objective

Continuously discover, evaluate, and refine kinematic insights, empirical translation models, visualizations, and front-office scouting assets for the **NFL Big Data Bowl 2027** competition.

---

## Operating Principles & Guardrails

1. **Football Context First:**
   - Every metric must solve a real football problem for NFL coaches, scouts, and front office executives.
   - Ground insights in modern NFL schemes (split-safety coverage, match-quarters, pass protection vs pass rush).

2. **Data Science Rigor & Anti-Leakage:**
   - Always validate predictive models using `GroupKFold(groups=nfl_id)` across all folds.
   - Always control for physical confounders (body weight, drill type, draft class year).
   - Compute all kinematic derivatives strictly at 10 Hz ($\Delta t = 0.1$s) using numerical gradients ($\frac{ds}{dt}$).

3. **Strict Submission Constraints:**
   - Submission writeup (`docs/SUBMISSION_WRITEUP.md`) must strictly remain $\le 2,000$ words.
   - Maximum 10 publication figures.
   - Starter notebook (`notebooks/bdb_2027_starter.ipynb`) must be 100% standalone, error-free, and runnable from start to finish.

4. **Executable Verification:**
   - Run `uv run pytest tests/test_bdb_pipeline.py` after every iteration.
   - Verify notebook execution before committing changes.
   - Keep `references/literature_and_resources.md` and `references/bibliography.json` updated with any newly introduced scientific literature.

---

## Iterative Task Backlog

- [x] **Task 1: The Velocity Decomposition Principle** (Tangential acceleration vs eccentric braking vs normal curvature).
- [x] **Task 2: The Stopwatch Blind Spot** (Proving 40-yd dash fails on $>75\%$ of route targets).
- [x] **Task 3: 4-Quadrant Scouting Archetype Matrix** (Speed Z vs Maneuverability Z).
- [x] **Task 4: GroupKFold Cross-Validation & LightGBM** ($r = 0.521, R^2 = 0.253$).
- [x] **Task 5: Front-Office Prospect Scouting Cards** (Head-to-head percentile comparisons).
- [x] **Task 6: Defensive Line Trench Mass Confounder** (Proving weight explains $84\%$ of get-off variance).
- [x] **Task 7: Bootstrap 95% Confidence Intervals** (Generalization floor verification).
- [x] **Task 8: Interactive Generative UI Prospect Explorer** (`artifacts/prospect_explorer.html`).
- [x] **Task 9: Defensive Back & Cornerback Hip Transition Latency** (Evaluating CB break on ball from W-Drill / DB Backpedal tracking to NFL EPA and completion rate allowed: $r = +0.204, p = 0.0429$).
- [x] **Task 10: Offensive Line Anchor & Pass Protection Kinetics** (Evaluating OL pass set stability vs pass-rushers encountered: lateral mirror acceleration $r = -0.254, p = 0.0096$; OLS $R^2 = 0.115$).

---

## When Done

When all tasks pass verification and tests are green, emit `LOOP_COMPLETE`.
