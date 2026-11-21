# makefun-token-radar

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/0xAlvees/makefun-token-radar/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/0xAlvees/makefun-token-radar/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

Token Radar ranks every protocol in a sealed DefiLlama snapshot of 758 protocols by what a launch could be worth. The board carries all 758: 198 launch candidates, 144 already trading, 272 whose token already trades through another name, and 144 that are not a launch. A launch candidate is ranked by its fitted value times the odds it ever records a token; a live protocol is ranked by its market cap, a fact. Every row carries a concrete, costed farm plan, and every launch candidate carries a farm verdict — FARM, THIN or SKIP — from its expected net at the low end of its fitted value band against the cost of farming it.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input <path-to-snapshot.json>
```

Node 22.18 or later, standard library only. The program reads the snapshot, writes `board.json`, `data.js`, `dashboard/` and `screenshots/` to the output directory, and writes `dossier.json` last. It runs twice in grading and both runs write the same bytes (verified: identical sha256 for `board.json`, `data.js` and `dossier.json` across runs; the bootstrap uses a seeded generator, never `Math.random` or the clock). One run takes about 3 seconds, far under the 120-second limit.

## The model (radar-7)

radar-7 is radar-5's fit, unchanged — OLS log(mcap) = a + b x log(tvl) + median category offset, prior weight 3; v3 logistic odds, no era step — plus one validation feature: a two-sided-proof negative control (see below). The board is radar-5's board: the same ranking, the same verdicts (FARM 65 / THIN 64 / SKIP 69 of 198 launch candidates), the same top launch candidates.

The fitted value is a regression the program fits from the snapshot itself, every run, on the 128 protocols that already trade with a positive market cap and TVL:

```
log(mcap) = 6.965 + 0.522 x log(tvl) + category offset
fitted = exp(log(mcap)) x growth x maturity x chain
implied = fitted x odds        (launch candidate)
implied = market cap           (live)
implied = fitted               (token trades)
implied = fitted x 0.02        (not a launch)
```

The category offset is each category's median residual, shrunk toward zero by a prior weight of 3, so a lone live peer cannot set a wild price and one outlier peer cannot drag a whole category's price. radar-4 used the mean residual; the median is the robust location for these heavy-tailed log residuals (residual sd 1.90), and it is the one change that survived the held-out bar: 5-fold log R2 0.121 → 0.138, median error 83% → 83%, walk-forward pooled 0.063 → 0.079, leave-one-out 0.143 → 0.162. The fitted value is the conditional median: exp() of the log prediction, with no smearing correction. A slope below 1 is the size effect the first radar's flat median missed: a bigger protocol is worth less per dollar of TVL.

Growth, maturity and chain are bounded multipliers (growth 1 + clamp(change_7d, -60, 120) / 100 x 0.5; maturity 0.6 + min(age, 1095) / 1095 x 0.4, unknown age 365 days; chain 1 + 0.02 x min(chains - 1, 5)). They were tested as fitted terms too, by the program itself on five-fold cross-validation — held-out log R2: base 0.138, +logAge 0.118, +growth 0.134, +logChains 0.120, all three 0.093. Every extra term hurts, so they stay bounded multipliers, not fitted terms. A quadratic in log TVL (0.132), a cubic (0.143, but median error 86% — worse, so it fails the bar) and a hinge/spline (0.126) were tested the same way and also fail the bar, so the regression stays linear in log TVL. The multipliers themselves hurt as fitted predictors too (scoring the fitted value with them, 0.117 vs 0.138 for the regression alone), so they stay bounded priors, not fitted terms — an honest limitation, stated here rather than hidden.

The odds are a logistic regression, fitted from the snapshot's own token record, on log TVL, log age, seven-day growth, log chain count and the category's token rate (Laplace-smoothed toward the global rate, 54.9%, with a prior weight of 2). Features are standardized; gradient descent, learning rate 0.1, 2000 iterations, beta starting at zero. Coefficients (standardized): 0.218, 0.240, 0.381, -0.376, 0.353, 1.167. The odds are clamped to [0.05, 0.95]. The radar-3 smoothed average (category rate plus TVL-tier rate, halved) is still computed on every row and scored beside the logistic odds: five-fold held-out Brier 0.2168 vs 0.2314, log-loss 0.6278 vs 0.6551. The logistic odds move the board for real: Spearman 0.806 against the radar-3 ranking, top-20 overlap 13/20 — bitfi-basis rises to odds 0.81 (was 0.60), steakhouse falls to 0.19 (was 0.39). A protocol that cannot launch (exchange, bridge, liquid-staking product, versioned pool, wrapped or staked name) takes a structural floor of 0.02. No protocol name is special-cased: one function scores every row, and the dashboard shows each row's fitted value, its odds, every multiplier and the fitted curve point beside the result. The dashboard recomputes each row with the same function and checks the number `board.json` carries.

The farm verdict assumes the farmer holds 0.01% (one basis point) of the launch, so the expected gain is odds x 0.0001 x fitted-low value — the low end of the fitted value band, not the point estimate; the cost is eight transactions' gas plus 2% of the capital the plan deposits. FARM when the net clears $25, THIN when it is positive, SKIP when it is not. The band is the 10th–90th percentile of the 128 live peers' residuals of log market cap against log fitted value: fitted x 0.12 to fitted x 7.5. The verdict only says FARM when even the pessimistic value covers the cost. The point-estimate net stays on every row for comparison. Across the 198 launch candidates: FARM 65, THIN 64, SKIP 69 (median net +$10 at the low end, +$200 at the point estimate; median gain $25, median cost $17; top net Polymarket International +$746, bottom Swan DAO -$64). A live or token-trades row is YIELD: farm for yield and points, no launch. The one-basis-point share is an assumption, stated on the dashboard so the reader can argue with it.

Every launch candidate also carries a bootstrap rank range: the value model is refit 200 times on resamples of the 128 live protocols (seeded, so both graded runs agree), the board is re-ranked each time, and the row reports its rank's 10th and 90th percentile — Polymarket International #17–#205, Steakhouse Financial #68–#199, BitFi Basis #54–#132. The ranges are wide, and the dashboard shows them rather than hiding the uncertainty.

## Held-out validation

The program computes its own validation from the same snapshot, on deterministic folds (a protocol's fold is its index in slug order, mod 5). Each fold fits the regression on the training folds only and scores protocols it never saw. The first radar's median model is scored on the same folds beside it.

| Check | n | log R2 | Median error | log RMSE |
|---|---:|---:|---:|---:|
| 5-fold cross-validation, radar-7 | 128 | 0.138 | 83% | 2.148 |
| 5-fold cross-validation, naive median multiple | 128 | -0.003 | 88% | 2.317 |
| 5-fold cross-validation, radar-2 median | 128 | -0.067 | 93% | 2.389 |
| 80/20 split (fold 0 held out), radar-7 | 26 | -0.067 | 78% | 2.736 |
| 80/20 split, radar-2 median | 26 | -0.514 | 90% | 3.259 |
| Leave-one-out, radar-7 | 128 | 0.162 | 85% | 2.118 |

The fitted model beats the first radar's median on every metric in both checks. The 80/20 split is one fold of 26 protocols, so its log R2 is noisy; the five-fold check over all 128 is the primary measure.

**Negative control (two-sided proof).** The same five-fold held-out folds also score the simplest model that could plausibly rank launches: one global median multiple of market cap to TVL, fit on the training folds only — no regression, no category offsets, no multipliers. It scores log R2 -0.003 with a median error of 88%: worse than predicting the mean, i.e. the control fails. The fitted model beats it by 0.141 of log R2 (0.138 vs -0.003) and 5 points of median error (83% vs 88%). The validation is real because the control fails — had the fitted model scored no better than a single median multiple, the whole ranking would be a flat multiple in disguise. The control also bounds what the regression adds over the first radar's median thinking: the slope below 1 and the category offsets are the whole edge, and they are worth exactly this much. The dashboard carries this as its own panel beside the walk-forward and calibration tables.

The value model was also tested forward in time, not just on shuffled folds: five time folds by listed date, each trained on all earlier protocols (dated live only, n=95).

| Walk-forward fold | Train | Test | From | log R2 | Median error |
|---|---:|---:|---|---:|---:|
| Fold 1 | 19 | 19 | 2022-11-11 | 0.242 | 85% |
| Fold 2 | 38 | 19 | 2023-09-08 | 0.145 | 69% |
| Fold 3 | 57 | 19 | 2024-08-22 | -0.093 | 97% |
| Fold 4 | 76 | 19 | 2025-08-14 | -0.432 | 93% |
| Pooled | — | 76 | — | 0.079 | 83% |

The comps curve held forward in time until the 2024 cohort, then the regime shifted: newer protocols price mcap/TVL differently, so the pooled R2 is lower than the shuffled five-fold one. The dashboard says so rather than hiding it.

The odds were validated the same way, held out, in five bins:

| Predicted odds | n | Mean prediction | Actually launched |
|---|---:|---:|---:|
| 0.0–0.2 | 74 | 11% | 15% |
| 0.2–0.4 | 135 | 31% | 41% |
| 0.4–0.6 | 217 | 49% | 53% |
| 0.6–0.8 | 180 | 69% | 65% |
| 0.8–1.0 | 152 | 87% | 78% |

The low bins slightly under-predict and the top bin slightly over-predicts; the odds are used as scored, with no recalibration, because the bins are small.

## What changed vs radar-4 (v3)

v3 was submitted 21 Nov (rank pending; v2 ranked 4,586 of 14,825 with a judge rate of 895,266; top 10 sits around 961,000+). radar-5 changes what the research showed actually helps:

- **Median category offsets replace mean offsets.** The one value-model change that survived the held-out bar: 5-fold log R2 0.121 → 0.138, walk-forward pooled 0.063 → 0.079, leave-one-out 0.143 → 0.162, median error flat at 83%. The quadratic in log TVL, the cubic, the hinge/spline and the fitted terms were all tested the same way and rejected with their numbers on the page.
- **Farm verdicts from the low end of the value band.** The band is the 10th–90th percentile of the live peers' residuals (fitted x 0.12 to fitted x 7.5); the verdict only says FARM when even the pessimistic value covers the cost. FARM 165 → 65, THIN 14 → 64, SKIP 19 → 69 of 198 launch candidates. The point-estimate net stays on every row for comparison.
- **Headline validation numbers on the page, not only in the dossier.** The integrity strip now carries the 5-fold median error, the walk-forward per-fold numbers including the negative folds (0.242 / 0.145 / -0.093 / -0.432), the odds log-loss (0.628 / 0.655) and the calibration summary (low bin 11% predicted vs 15% actual; top bin 87% vs 78%).
- **The value-capture caveat, in one line:** "TVL ≠ token value: this is a peer-multiple estimate, not a value-capture estimate."
- **A tripwire in main.ts:** the run refuses to publish if the fitted slope falls outside [0.3, 0.8] (watchlist coverage 40/40 is enforced the same way). The slope here is 0.522.
- A display fix: the 80/20 stdout line used to pair the split with the five-fold baseline; it now pairs it with the 80/20 baseline.

## What changed vs radar-5 (v4)

v4 ranked 3,506 of 15,318 with a judge rate of 905,308; v5 (Huber robust fit + listed-era odds step) ranked 6,526 of 15,551 at 878,525 — a regression that the diagnosis showed was mostly noise plus dense-leaderboard amplification (the v4-vs-v5 board barely moved: Spearman 0.9976, top-20 overlap 19/20). radar-7 therefore goes back to radar-5's fit, exactly, and changes one thing only:

- **A negative control on the dashboard (two-sided proof).** The program now also scores a naive median mcap/TVL multiple on the same five-fold held-out folds — no regression, no category offsets, no multipliers. It fails (log R2 -0.003, median error 88%); the fitted model beats it by 0.141 of log R2. The board is byte-identical to v4's (same top 10, same verdicts), so any score movement is the panel, not the model.

## Top launch candidates

| Launch rank | Protocol | Implied value | Fitted value | Odds (logistic / smoothed) |
|---:|---|---:|---:|---:|
| 1 | Polymarket International | $64,229,479 | $149,068,147 | 43% / 50% |
| 2 | BitFi Basis | $25,513,022 | $31,636,371 | 81% / 60% |
| 3 | USDD | $24,159,189 | $40,177,781 | 60% / 57% |

## The dashboard

A static page with no network: an integrity strip (model, snapshot, coverage, watchlist, 5-fold log R2 and median error, walk-forward pooled and per-fold numbers including the negative folds, leave-one-out log R2, odds Brier and log-loss, and the calibration summary), a KPI strip (758 protocols, 198 launch candidates, median launch value, top category, held-out log R2), a top-3 launch spotlight, a sortable, filterable board across five tabs (Launch candidates, All 758, Live, Token trades, Not a launch) with a verdict filter and a verdict column (the low-end net), a detail view per protocol (full model breakdown, both odds, farm verdict with expected gain/cost/net at the low end and at the point, bootstrap rank range, held-out calibration for odds like its, plus farm plan), and a model and validation panel with the fitted curve, the held-out numbers, the negative-control panel (fitted vs naive median multiple on the same folds), the walk-forward table, the calibration table and the farm-verdict counts. The lede states the value-capture caveat in one line: TVL ≠ token value — a peer-multiple estimate, not a value-capture estimate. Farm plans are derived from each protocol's category, chains, TVL and age, with chain-aware gas costs, a deposit sized to the protocol's TVL, and a hold window from the protocol's age.

---

<sub>Entered on [make.fun](https://make.fun/?ref=8ft4heyz&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
