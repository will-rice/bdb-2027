# External Literature & Research Resource Registry

**NFL Big Data Bowl 2027: Prospect Tracking & Translation Analysis**
_Maintained for competition methodology, model validation, and final executive writeup citations._

---

## 1. Peer-Reviewed Academic Literature

### [Harper-2022-Decel]

- **Title:** Biomechanical and Neuromuscular Performance Requirements of Horizontal Deceleration: A Review with Implications for Random Intermittent Multi-Directional Sports
- **Authors:** Damian J. Harper, Christopher Carling, John Kiely
- **Journal:** _Sports Medicine_, 52(5), 1021–1054 (2022)
- **DOI / URL:** [10.1007/s40279-022-01693-0](https://doi.org/10.1007/s40279-022-01693-0)
- **OpenAlex ID:** `https://openalex.org/W4281704315`
- **Core Findings:**
  - Horizontal deceleration is a mechanically distinct physical quality from acceleration, requiring rapid dissipation of kinetic energy via high eccentric ground reaction forces (often 2–3× body weight higher than propulsive forces).
  - Braking impulse ($J_{brake} = \int F_x \, dt = m \cdot \Delta v_x$) governs the transition into change-of-direction (COD) maneuvers.
  - Athletes with superior deceleration can sprint into cuts at higher velocities without losing balance or inflating turning radius.
- **Application to BDB 2027:**
  - Explains why 40-yard dash stopwatch times fail to predict route separation on breaking routes: sprint velocity measures propulsive impulse, whereas route-running sharpness depends on braking impulse and deceleration rate entering the cut.

---

### [Harper-2024-BrakingFramework]

- **Title:** The Braking Performance Framework: Practical Recommendations and Guidelines to Enhance Horizontal Deceleration Ability in Multi-Directional Sports
- **Authors:** Damian J. Harper, Daniel D. Cohen, Christopher Carling, John Kiely
- **Journal:** _International Journal of Sports Science and Coaching_ / IUSCA, 4(1), 351 (2024)
- **DOI / URL:** [10.47206/ijsc.v4i1.351](https://doi.org/10.47206/ijsc.v4i1.351)
- **OpenAlex ID:** `https://openalex.org/W4402509752`
- **Core Findings:**
  - Formulates multi-phase braking metrics: initial braking phase, penultimate foot contact (PFC), and final foot contact (FFC) plant phase.
  - Proves that peak deceleration ($a_{min}$) and braking duration differentiate elite multi-directional athletes from linear track sprinters.
- **Application to BDB 2027:**
  - Guides feature engineering for Wide Receiver combine route drills (`SKILL_DRILLS_WR`) and defensive back change of direction drills, extracting entry deceleration gradient ($\frac{dv}{dt}$) and transition latency.

---

### [Nimphius-2016-CODD]

- **Title:** Change of Direction Deficit: A More Isolated Measure of Change of Direction Performance Than Total 505 Time
- **Authors:** Sophia Nimphius, S. J. Callaghan, T. Spiteri, R. G. Lockie
- **Journal:** _Journal of Strength and Conditioning Research_, 30(11), 3024–3032 (2016)
- **DOI / URL:** [10.1519/JSC.0000000000001421](https://doi.org/10.1519/JSC.0000000000001421)
- **OpenAlex ID:** `https://openalex.org/W2301686871`
- **Core Formula:**
  $$\text{COD Deficit} = \text{COD Test Time} - \text{Linear Sprint Time over Same Distance}$$
- **Core Findings:**
  - Total times in agility drills (like 3-cone and pro agility) correlate strongly with straight-line sprint speed ($r > 0.70$), masking an athlete's true maneuverability and cutting efficiency.
  - COD Deficit mathematically removes linear speed variance, isolating true direction-change mechanics, plant mechanics, and hip mobility.
- **Application to BDB 2027:**
  - We isolate Combine drill agility metrics by residualizing drill transit times and cut transitions against straight-line 40-yard dash and 10-yard split times, creating a pure _Kinematic Maneuverability Index (KMI)_.

---

### [Teramoto-2015-CombineValidity]

- **Title:** Predictive Value of National Football League Scouting Combine on Future Performance of Running Backs and Wide Receivers
- **Authors:** Masaru Teramoto, Chad L. Cross, Melissa Willick
- **Journal:** _Journal of Strength and Conditioning Research_, 29(6), 1579–1587 (2015)
- **DOI / URL:** [10.1519/JSC.0000000000001202](https://doi.org/10.1519/JSC.0000000000001202)
- **OpenAlex ID:** `https://openalex.org/W2438621791`
- **Core Findings:**
  - Empirical study across historical Combine cohorts found that traditional stopwatch tests (40-yard dash, vertical jump, broad jump) have little to no statistically significant correlation with on-field receiving metrics (yards per route run, receptions, touchdowns).
  - Identified the "Stopwatch Paradox": NFL teams over-index on raw linear speed during draft day, creating draft inefficiencies and missed evaluations on prospects with elite non-linear movement traits.
- **Application to BDB 2027:**
  - Provides the peer-reviewed justification for our core thesis: 10 Hz optical sensor tracking bridges the gap that stopwatches miss, uncovering genuine in-game route translation.

---

### [Deshpande-2023-STRAIN]

- **Title:** Here Comes the STRAIN: A Mechanical Framework for Quantifying Pass-Rush Pressure
- **Authors:** Sameer K. Deshpande, Katherine Evans, et al.
- **Journal:** _Journal of Quantitative Analysis in Sports_, Big Data Bowl 2023 Grand Prize Research
- **ArXiv / URL:** [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) / [Amazon Science](https://www.amazon.science)
- **Core Concept:**
  - Models pass-rush pressure as continuous physical strain on the quarterback pocket based on rusher-to-QB distance, vector approach velocity, and block resistance over time, rather than coarse binary sack/hit outcomes.
- **Application to BDB 2027:**
  - Validates using 10 Hz continuous kinematics to evaluate trench play (pass rusher get-off and offensive line pass protection) rather than relying strictly on binary sack counts.

---

### [Deshpande-Evans-2019-EHCP]

- **Title:** Expected Hypothetical Completion Probability (EHCP)
- **Authors:** Sameer K. Deshpande, Katherine Evans
- **Journal:** _Journal of Quantitative Analysis in Sports_ (2019)
- **ArXiv ID:** [arXiv:1910.12337](https://arxiv.org/abs/1910.12337)
- **Core Findings:**
  - Bayesian non-parametric framework modeling receiver separation, speed, defender closure, and ball trajectories across all routes on a play, accounting for counterfactuals of un-targeted receivers.
- **Application to BDB 2027:**
  - Establishes route separation at throw/catch point as the foundational currency of NFL receiver evaluation.

---

## 2. Industry Analytics, Next Gen Stats & Tracking Frameworks

### [AWS-NGS-Separation]

- **Organization:** NFL Next Gen Stats / Amazon Web Services
- **Metric Definitions:**
  - **Separation at Pass Forward:** Distance in yards between the targeted receiver and nearest defensive player when the quarterback releases the pass.
  - **Separation at Catch:** Distance in yards when the pass arrives or is contested.
  - **Pass Rush Get-Off:** Elapsed time (seconds) from the official snap of the ball until a defensive pass rusher's body crosses the line of scrimmage (or initial acceleration burst exceeding 3.0 yd/s²).
- **Application to BDB 2027:**
  - Standardizes our target variables (`separation`, `player_get_off`, `quick_pressure`).

### [SumerSports-Riske-2023-Agility]

- **Author:** Timo Riske / Eric Eager (SumerSports Analytics)
- **Article:** "Evaluating NFL Combine Drills: The Signal in the Noise" (2023)
- **Key Insight:**
  - Combine drills without defenders or physical contact are cleaner measures of isolated motor control and physiological capacity, but only when controlled for athlete anthropometrics (BMI, height, weight).
  - Linear speed does not correlate with short-area change of direction; prospects who test well in both are statistical outliers.

---

## 3. Official Competition Specifications & Organizer Rulings

### [NFL-BDB-2027-Rules]

- **Organizer:** NFL Operations & Kaggle
- **Document:** Official Competition Rules, Overview & Data Dictionary
- **URL:** [Kaggle NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027)
- **Evaluation Criteria:**
  1. Mandatory: Combine-to-NFL Linkage (Pass/Fail)
  2. Mandatory: Public Kaggle Notebook Complete & Runnable (Pass/Fail)
  3. Football Context (30%)
  4. Data Science Rigor (30%)
  5. Writeup Quality & Narrative (20%)
  6. Data Visualization (20%)
- **Constraints:** Writeup limited to 2,000 words and 10 figures.

### [Bliss-2026-Rulings]

- **Authority:** Tom Bliss (Director of Football Data & Analytics, NFL)
- **Official Clarifications (Kaggle Discussion Forum):**
  1. **Pass-Rush Definition:** When `blitz` is true, player is rushing the passer. For non-blitzing defensive players on passing plays, filter for `pass_result != ""` to isolate true dropbacks.
  2. **Pressure Encoding:** In `player_play.csv`, when `quick_pressure` or `time_to_pressure` is null on a blitz play, it denotes **no pressure occurred** (not missing data). When `quick_pressure == 0`, pressure occurred but exceeded 2.50 seconds.
  3. **Get-Off Nulls:** `player_get_off` is null if a player never crosses the line of scrimmage (rare event).
  4. **Ball Frames in Combine Tracking:** In `combine_tracking.csv`, `nfl_id` on rows with `entity_type == "BALL"` is attributed to the athlete performing that rep (occurs in 551 reps across 54 players, primarily WR/TE).
  5. **Unrostered Prospects:** 13 prospects in the combine tracking never logged an NFL regular-season play (did not make active 53-man game roster).

---

## 4. Methodological Traps & Guardrails

| Trap Identified                | Root Cause                                                                                                                                    | Analytical Solution / Guardrail                                                                                                          |
| :----------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **Draft Class Sensor Drift**   | 2023/2024 coordinates sampled on 1/60 yd grid; 2025 coordinates continuous floats. 2024 DB drills truncated at 6.0s vs 11.6s in 2023.         | **Class-Centering:** Standardize all drill kinematics by `(drill_type, draft_year)` before pooling.                                      |
| **Leakage via Group Overlap**  | A single player has dozens of reps across combine and regular-season games. Standard KFold leaks player identity across train and test folds. | **GroupKFold by `nfl_id`:** Prospect-level stratification ensuring strict out-of-sample validation.                                      |
| **Anthropometric Confounding** | Heavier players naturally have slower get-off and lower peak speed, but higher power. Raw correlations reflect size, not skill.               | **Weight & Alignment Residualization:** Orthogonalize kinetic features against athlete mass ($m$) and snap alignment (interior vs edge). |
| **Stopwatch Collinearity**     | Agility drills (3-cone, shuttle) are collinear with 40-yard dash speed ($r = 0.70$).                                                          | **COD Deficit Calculation:** Compute difference between transit times and sprint baseline to isolate pure turning mechanics.             |
