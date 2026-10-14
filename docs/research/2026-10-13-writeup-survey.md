# Write-up survey and direction, 13 October 2026

Scope: every public notebook attached to NFL Big Data Bowl 2027 one week after launch, the 31 discussion
topics, the last three Big Data Bowl winners, and ten probes of candidate directions on the competition data.
Every number in the probe tables is reproduced by a script in [`direction-probes/`](direction-probes/).

## Bottom line

1. **The field is saturated and has already converged.** 185 public notebooks exist after seven days. The
   rigorous ones agree: tracked top speed transfers to games, body size confounds almost everything else, and
   cut, brake and change-of-direction metrics from single drill reps neither repeat nor transfer.
2. **Our current write-up is a weaker entry in the most crowded lane.** It is a four-position careful null. At
   least 26 notebooks report nulls, and the most-upvoted analysis in the competition is a tracking audit that
   does reliability, power and confound control better than we do.
3. **No hidden signal turned up.** Six candidate directions were probed. Five failed. The one large, validated
   effect is unglamorous: a receiver's weight and height predict a third of the variance in his separation in a
   draft class the model never saw, and adding route-drill tracking makes that prediction worse.
4. **Recommended direction: narrow to wide receivers and lead with "frame first".** Use separation on every
   route as the outcome, a frame baseline as the bar, and judge each tracking feature by what it adds out of
   class. Details and risks are in the last section.

## What the hosts have clarified

| Topic                           | Ruling                                                                                               | Source                                                                                 |
| :------------------------------ | :--------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- |
| Word and figure limits          | Apply to the write-up only; an appendix inside the write-up is excluded                              | [746425](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/discussion/746425) |
| AI assistants and private cloud | Allowed under rules section 2.6                                                                      | [747086](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/discussion/747086) |
| Preseason and postseason plays  | May be used as supporting outcomes                                                                   | [746425](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/discussion/746425) |
| Ball rows in Combine tracking   | Carry the player's `nfl_id`; the data dictionary was wrong and has been corrected                    | [747108](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/discussion/747108) |
| Receivers and tight ends        | Only plays where they ran a route are included, by design                                            | [748491](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/discussion/748491) |
| Row counts                      | `player_play.csv` has 314,197 rows; the CSVs are the full dataset                                    | [747584](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/discussion/747584) |
| Pass-rush nulls                 | Null pressure with `blitzing` true means no pressure; filter dropbacks with a non-null `pass_result` | [746678](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/discussion/746678) |
| `player_get_off`                | Measured from the ball's location at the snap, which is not provided                                 | [748114](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/discussion/748114) |
| External data                   | Public sources such as nflreadr are acceptable                                                       | [747108](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/discussion/747108) |

An earlier note in this session said `player_play.csv` has 327,019 rows. That was a line count; play
descriptions contain line breaks. The parsed count is 314,197.

## The field in numbers

Keyword census over the 178 notebooks readable as notebooks (markdown plus the first 12,000 characters of
code, so counts are lower bounds). Script: `survey_census.py`.

| Share | What the notebook does                                 |
| ----: | :----------------------------------------------------- |
|   47% | Wide receiver or tight end routes, cuts, separation    |
|   35% | Defensive line or edge pass rush                       |
|   32% | Offensive line pass protection                         |
|   12% | Defensive back coverage                                |
|   46% | Reads game tracking, not only the play table           |
|   53% | Controls for weight                                    |
|   42% | Controls for draft slot                                |
|   52% | Checks repeatability or reliability                    |
|   43% | Holds out a draft class or runs a permutation test     |
|   15% | States a null or negative result                       |
|   17% | Uses "beyond the stopwatch" or "beyond the 40" framing |
|   12% | Uses a quadrant or archetype matrix                    |
|   11% | Promises hidden gems, sleepers or steals               |
|    5% | Uses external data                                     |
|    2% | Animates anything                                      |
|    1% | Pairs two tracked players on the same play             |
|    1% | Tests on the live 2026 season                          |

Our original write-up used a "4-Quadrant Scouting Archetype Matrix" of "Hidden Movers".
[Another public notebook](https://www.kaggle.com/code/oriolpi/notebook7b60ad12bf) uses the same two labels.

## What the rigorous notebooks agree on

- **Tracked speed is the one trait that repeats and transfers.** The tracking reproduces the official 10 to 40
  yard split closely but not the first 10 yards
  ([audit](https://www.kaggle.com/code/dariushafshar/combine-tracking-audit-speed-r-0-95-start-r-0-28)).
  Speed carries from Combine to games; the cut does not
  ([cut study](https://www.kaggle.com/code/candiceevawilliams/the-combine-can-t-see-the-cut),
  [transfer map](https://www.kaggle.com/code/kassasana/bdb2027-speed-transfer)).
- **Draft slot dominates playing time.** Out of class, draft position explains about half of rookie playing
  time and the Combine adds nothing detectable
  ([who beat their Combine](https://www.kaggle.com/code/dariushafshar/who-beat-their-combine-2023-25-rookies)).
- **Pre-specified receiver tests are null.** Braking, cut recovery and curl-drill features show no dependable
  gain on held-out classes ([one](https://www.kaggle.com/code/antoniorotundo2/nfl-combine-movement-transfer-2027),
  [two](https://www.kaggle.com/code/gautiermarti/nfl-2027-beyond-the-stopwatch-cpu-baseline),
  [three](https://www.kaggle.com/code/fireaideveloper/beyond-the-stopwatch-curl-to-nfl)).
- **Offensive line drill features are null on frozen tests**
  ([mirror](https://www.kaggle.com/code/abhishekd2512/relative-mirror-recovery-and-rookie-protection),
  [pass sets](https://www.kaggle.com/code/wangh2o/nfl-2027-phase-aligned-pass-sets)).
- **Defensive line is where the positive claims cluster.** Tracked sprint speed is linked to pass-rush
  production, in sample, at roughly 1.3 to 1.6 times the odds per standard deviation
  ([matchup layer](https://www.kaggle.com/code/rolandobouloy/who-faced-whom-rookie-vs-rookie-matchup-bdb-2027)),
  and to interior linemen's rush movement
  ([bend](https://www.kaggle.com/code/agileteam/bend-before-sacks-dl-combine-tracking)). The most careful
  pricing of it finds Combine burst adds nothing once the draft profile is known
  ([44 snaps](https://www.kaggle.com/code/jasonrobinson1/worth-44-snaps-when-film-beats-the-combine)).

## The entries to beat

| Notebook                                                                                                             | Idea                                                                               | Why it is strong                                                                      |
| :------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| [Worth 44 Snaps](https://www.kaggle.com/code/jasonrobinson1/worth-44-snaps-when-film-beats-the-combine)              | Prices a pass rusher's pre-draft profile in NFL snaps of evidence                  | One estimator on both feeds, held-out classes, an in-season forecast, polished layout |
| [Indy or Tape?](https://www.kaggle.com/code/olsols4/indy-or-tape-pricing-a-combine-workout)                          | Exchange rate between a Combine prior and per-game evidence, every position        | Decision framing a front office already uses                                          |
| [Combine Tracking Audit](https://www.kaggle.com/code/dariushafshar/combine-tracking-audit-speed-r-0-95-start-r-0-28) | What the tracking can and cannot measure, with a metric scorecard others can reuse | Most-upvoted analysis; owns the audit lane                                            |
| [Bend Before Sacks](https://www.kaggle.com/code/agileteam/bend-before-sacks-dl-combine-tracking)                     | In-game rush signature, pressure over expected, edge versus interior               | Negative control on blockers, references to public win-rate work                      |
| [The Speed Tax](https://www.kaggle.com/code/ethanwchen/the-speed-tax-reproducible-analysis)                          | How much of a receiver's 40 he reaches by the throw                                | Locked yardstick, notebook fails if a quoted number drifts                            |
| [Expected Movement](https://www.kaggle.com/code/wonofone/bdb2027-expected-movement)                                  | Movement over expected for position, weight and height, on both feeds              | Size-adjusted scores done with sprint physics                                         |
| [Who Faced Whom?](https://www.kaggle.com/code/rolandobouloy/who-faced-whom-rookie-vs-rookie-matchup-bdb-2027)        | Validated rookie-versus-rookie matchups and a pre-registered 2026 test             | The only two-sided design; one hypothesis frozen in public                            |

## What past winners look like

| Year | Winner                                  | What it was                                                                                                                                                                                                      |
| :--- | :-------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2024 | Uncovering Missed Tackle Opportunities  | One new tackling metric ([NFL](https://operations.nfl.com/gameday/analytics/big-data-bowl/2023-big-data-bowl-finalists))                                                                                         |
| 2025 | Exposing Coverage Tells in the Pre-Snap | A model turned into a usable tell ([NYU](https://www.sps.nyu.edu/about/news-and-ideas/articles/preston-robert-tisch-institute-for-global-sport/2025/nyu-duo-crowned-winners-of-the-2025-nfl-big-data-bowl.html)) |
| 2026 | Ghostbusters                            | Defender movement in the air against hypothetical ghost defenders ([NFL](https://operations.nfl.com/gameday/analytics/big-data-bowl/))                                                                           |

Each is one named idea about one football question. None is an audit or a multi-position survey. Only titles
and summaries were read for this table, not the full write-ups. A
[review of the 2025 field](https://statthinksportsanalytics.substack.com/p/thoughts-on-big-data-bowl-2025)
makes the same points the rubric does: report base rates and uncertainty, match scope to the data, and put the
novelty in the downstream use.

## Probes of candidate directions

Regular season only unless stated. Reliability is a split-half correlation of player means.

|   # | Candidate                                                                               | Result                                                                                                                                                                                               | Verdict                                        |
| --: | :-------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------- |
|   1 | Rookie-versus-rookie matchups: match the recorded separation to a second tracked player | 6,354 reps where a tracked defender is exactly the recorded nearest defender (5,160 regular season, 2,151 pairs). Only 1 pair contradicts the match                                                  | Real data asset, already built by one notebook |
|   2 | Does a Combine speed edge decide those reps?                                            | +0.26 yards per yd/s (p = 0.007), gone once the weight edge is added (p = 0.42). In receiver-versus-corner man reps, player identity explains 1 to 3% of rep separation                              | Too small to carry a write-up                  |
|   3 | A defender-side outcome for corners                                                     | Route-adjusted separation allowed has reliability 0.13 to 0.19                                                                                                                                       | Not a player trait in this sample              |
|   4 | Closing on a tracked receiver's catch point during ball flight                          | 2,833 reps, 84 defenders. Context explains 91% of closing speed; closing over expected has reliability 0.03                                                                                          | Dead                                           |
|   5 | Year-2 headroom: does Combine speed add to year-1 game speed?                           | Year 1 to year 2 r = 0.65. Combine adds R² 0.006 (p = 0.29, n = 114)                                                                                                                                 | Dead                                           |
|   6 | Deployment instead of performance                                                       | Tracked speed predicts a receiver's vertical-route share (r = 0.30, n = 56) but not cushion; nothing for corners                                                                                     | Supporting detail only                         |
|   7 | Offensive line drills against pressure probability allowed                              | Outcome reliability 0.84. Mirror drill alone r = 0.00. Mirror plus wave acceleration is borderline (p = 0.05) among five metrics                                                                     | Our earlier "lead" is noise-level              |
|   8 | Offensive line chain: Combine pass set to game pass set to pressure                     | Game set depth has within-position reliability 0.96 and predicts pressure on disjoint plays (r = 0.31). Combine first-second burst has rep-to-rep r = -0.20 and no link to either                    | The Combine cannot see it                      |
|   9 | Receiver separation against frame                                                       | Separation on all routes has reliability 0.81 (receivers with at least 100 routes). Weight r = -0.57, height r = -0.56 (n = 62), unchanged after adjusting for route, alignment and coverage (-0.61) | **Large and robust**                           |
|  10 | Does it hold in an unseen draft class?                                                  | Leave one draft class out, 56 receivers. Weight plus height: out-of-class R² 0.33. Adding tracked top speed: 0.35. Adding route-drill braking and speed: 0.16. Official 40 alone: -0.05              | **Validated; tracking drills subtract**        |

One caution from probe 2. A pooled mixed model first attributed 30% of rep separation to the receiver and
0.1% to the corner. Fitted separately for man and zone it gives 1.2% and 3.4% in man, 1.3% and 0.3% in zone.
The pooled figure was a degenerate fit and is not used anywhere.

A caution on 2026 tests. Two notebooks freeze a hypothesis and score it on the live season. That guards
against noise in the outcome, not against the small number of players: the players are the same, and stable
player traits will "replicate" whatever they correlated with by chance. The only new-player test available is
leave-one-draft-class-out.

## Recommended direction: frame first

**Thesis.** For receiver separation the Combine's useful signal is the tape measure and the scale, with a
little speed. The 10 Hz route drills add nothing yet, because one rep per route does not repeat. Separation
should be read against a frame expectation, and tracked speed should be used to project role.

**Why this one.**

- It is the largest validated Combine-to-NFL effect in the data: out-of-class R² 0.33 on 56 receivers. The
  six-feature model in our current notebook reaches 0.09 on its own, narrower outcome.
- It explains the field's most common result. Cut metrics fail because they must beat a strong frame baseline,
  and most notebooks treat size as a nuisance control instead of the finding.
- It fixes a real weakness in our notebook, which scores only targeted breaking routes. The play table records
  separation on 46,737 route reps; 9,842 are targets.
- It stays inside the hosts' advice to study one position group.

**What changes.**

1. Outcome: mean separation at the throw over every route, with a version adjusted for route, alignment and
   man or zone.
2. Baseline: weight and height, scored leave-one-draft-class-out.
3. A ladder figure: what each addition does to out-of-class R². Tracked top speed, then route-drill features.
4. Product: separation over frame expectation for every receiver, and role projection from tracked speed.
5. Cut the quadrant matrix, the Dell and Mingo cards, the draft-capital score and the cornerback section.
   Keep the defensive line weight result as a one-paragraph echo. Move the offensive line result to a short
   "what the Combine cannot see" note.
6. Animate one comparison. Only 2% of notebooks animate, and visualization is 20% of the score.

**Risks.**

- The headline is anthropometric. A judge may say it is not a tracking result. Tracking has to be integral:
  speed comes from tracking, the ladder tests the drills, and usage comes from game data.
- Receivers are the most crowded position, and about one notebook in seven mentions size and separation.
- The cohort is 62 receivers with at least 100 routes.
- Odds are long in any direction. The Open Track has six prizes and the field already contains several
  polished entries.

**Ranked alternatives.**

1. Cross-position "the scale beats the sensors", using the receiver, defensive line and offensive line results
   together. Broader, but closer to
   [Expected Movement](https://www.kaggle.com/code/wonofone/bdb2027-expected-movement) and against the hosts'
   advice to focus.
2. The matchup layer as the centrepiece. Distinctive and visual, but player effects are 1 to 3% of rep
   variance.
3. Keep the four-position null. Not recommended.

## Reproduce

From the repository root, with the competition CSVs in `data/raw/`:

```bash
uv run python docs/research/direction-probes/build_cache.py
```

```bash
uv run python docs/research/direction-probes/probe_matchups.py
```

Then run any other `probe_*.py`. `probe_matchups.py` must run first because it writes the play table and the
confirmed matchups that later probes read. The census needs Kaggle credentials and two passes because the API
rate-limits after about 50 notebook reads:

```bash
uv run python docs/research/direction-probes/survey_fetch_notebooks.py
```

```bash
uv run python docs/research/direction-probes/survey_census.py
```

Caches are written under `data/`, which is git-ignored. Notebook text is other people's work and is not
committed.

## Outcome of the build

The frame-first version was built the same day. It is in
[`notebooks/frame_first.ipynb`](../../notebooks/frame_first.ipynb) and
[`docs/SUBMISSION_WRITEUP.md`](../SUBMISSION_WRITEUP.md). Three things differ from the plan above.

- **Dropped: a separation-over-frame grade.** What is left after frame does not persist from season to season,
  so a leaderboard of it would rank noise. The write-up reports that as a finding instead.
- **Dropped: role projection from tracked speed.** It holds in sample and fails in a held-out draft class.
- **Added: play speed from game tracking.** Lighter receivers reach a higher peak speed before the throw on
  the same route with the same time to throw, and that play speed carries part of the frame effect. It is
  also the one place where tracked Combine speed beats the official 40 time out of class.

The build also replaced the in-season product. Frame is tested directly against a receiver's own first routes
of NFL tape as a forecast of his later separation.
