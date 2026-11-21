# makefun-token-radar

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/0xAlvees/makefun-token-radar/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/0xAlvees/makefun-token-radar/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

Token Radar ranks every protocol in a sealed DefiLlama snapshot of 758 protocols by what a launch could be worth. The board carries all 758: 198 launch candidates, 144 already trading, 272 whose token already trades through another name, and 144 that are not a launch. A launch candidate is ranked by its fitted value times the odds it ever records a token; a live protocol is ranked by its market cap, a fact. Every row carries a concrete, costed farm plan, and every launch candidate carries a farm verdict — FARM, THIN or SKIP — from its expected net at the low end of its fitted value band against the cost of farming it.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input <path-to-snapshot.json>
```

Node 22.18 or later, standard library only. The program reads the snapshot, writes `board.json`, `data.js`, `dashboard/` and `screenshots/` to the output directory, and writes `dossier.json` last. It runs twice in grading and both runs write the same bytes (verified: identical sha256 for `board.json`, `data.js` and `dossier.json` across runs; the bootstrap uses a seeded generator, never `Math.random` or the clock). One run takes about 2 seconds, far under the 120-second limit.

## The model (radar-8)

radar-8 is radar-6's fit — the Huber robust value regression plus the listed-era odds step, both measured held-out on the same snapshot — plus radar-7's two-sided-proof negative control (see below). The board is v5's board, byte-identical: the same ranking, the same verdicts (FARM 56 / THIN 62 / SKIP 80 of 198 launch candidates), the same top launch candidates. Against v6's board it barely moves (Spearman 0.9976, top-20 overlap 19/20).

The fitted value is a regression the program fits from the snapshot itself, every run, on the 128 protocols that already trade with a positive market cap and TVL:

```
log(mcap) = 6.582 + 0.547 x log(tvl) + category offset
fitted = exp(log(mcap)) x growth x maturity x chain
implied = fitted x odds        (launch candidate)
implied = market cap           (live)
implied = fitted               (token trades)
implied = fitted x 0.02        (not a launch)
```

The regression is a Huber robust fit, not ordinary least squares: iteratively reweighted least squares, delta 1.0, the scale from the median absolute deviation of the residuals, at most 50 iterations, starting from the least-squares fit. Log market-cap residuals are heavy-tailed (residual sd 1.90), so a few outlier peers drag an OLS line; the Huber line downweights them. Held-out, on the same folds as radar-7: 5-fold log R2 0.138 → 0.154, median error 83% → 83%, walk-forward pooled 0.079 → 0.097, leave-one-out 0.162 → 0.165. The category offset is each category's median residual, shrunk toward zero by a prior weight of 3, so a lone live peer cannot set a wild price and one outlier peer cannot drag a whole category's price. The fitted value is the conditional median: exp() of the log prediction, with no smearing correction. A slope below 1 is the size effect the first radar's flat median missed: a bigger protocol is worth less per dollar of TVL.

Growth, maturity and chain are bounded multipliers (growth 1 + clamp(change_7d, -60, 120) / 100 x 0.5; maturity 0.6 + min(age, 1095) / 1095 x 0.4, unknown age 365 days; chain 1 + 0.02 x min(chains - 1, 5)). They were tested as fitted terms too, by the program itself on five-fold cross-validation — held-out log R2: base 0.154, +logAge 0.130, +growth 0.144, +logChains 0.140, all three 0.111. Every extra term hurts, so they stay bounded multipliers, not fitted terms. A quadratic in log TVL was re-tested under the Huber fit (0.146) and still fails the bar; a cubic (0.143, but median error 86% — worse, so it fails the bar) and a hinge/spline (0.126) were tested under OLS and fail the bar the same way. Per-category slopes (0.033), a k-nearest-neighbour local median (best 0.125 at k=25), binned TVL-decile medians (0.060), weighted least squares, a trimmed OLS and an L1 median regression (0.154 but median error 86% — worse) were all tested the same way and rejected. An honest limitation, stated here rather than hidden: the median error stays 83% because the log residuals are that wide (sd 1.90); no functional form tried so far narrows them.

The odds are a logistic regression, fitted from the snapshot's own token record, on log TVL, log age, seven-day growth, log chain count, the category's token rate (Laplace-smoothed toward the global rate, 54.9%, with a prior weight of 2) and a listed-2024-or-later step. Features are standardized; gradient descent, learning rate 0.1, 2000 iterations, beta starting at zero. Coefficients (standardized): 0.261, 0.194, 0.173, -0.282, 0.307, 1.103, -0.609. The odds are clamped to [0.05, 0.95]. The era step is there because the token rate is a step, not a slope, in listing time: protocols listed before 2024 record a token 78–80% of the time, the 2024 cohort 51%, 2025's 39%, 2026's 31% — younger protocols have had less time to launch, and log age alone, being log-linear, cannot fit that step. A protocol with no listed date takes the step's 0: its token rate is 0.79 over 113 protocols, the same as the pre-2024 cohorts. The odds are fitted on whether a protocol records a token in this snapshot, so a young protocol's lower odds partly reflect less time to launch, not a verdict that it never will. The radar-3 smoothed average (category rate plus TVL-tier rate, halved) is still computed on every row and scored beside the logistic odds: five-fold held-out Brier 0.2031 vs 0.2314, log-loss 0.5947 vs 0.6551. The logistic odds move the board for real: Spearman 0.796 against the fitted-x-smoothed ranking, top-20 overlap 14/20 — the era step moves the young protocols most: Steakhouse Financial falls to odds 0.13 (was 0.19 in radar-7, 0.39 smoothed), Polymarket International rises to 0.58. A protocol that cannot launch (exchange, bridge, liquid-staking product, versioned pool, wrapped or staked name) takes a structural floor of 0.02. No protocol name is special-cased: one function scores every row, and the dashboard shows each row's fitted value, its odds, every multiplier and the fitted curve point beside the result. The dashboard recomputes each row with the same function and checks the number `board.json` carries.

The farm verdict assumes the farmer holds 0.01% (one basis point) of the launch, so the expected gain is odds x 0.0001 x fitted-low value — the low end of the fitted value band, not the point estimate; the cost is eight transactions' gas plus 2% of the capital the plan deposits. FARM when the net clears $25, THIN when it is positive, SKIP when it is not. The band is the 10th–90th percentile of the 128 live peers' residuals of log market cap against log fitted value: fitted x 0.11 to fitted x 7.1. The verdict only says FARM when even the pessimistic value covers the cost. The point-estimate net stays on every row for comparison. Across the 198 launch candidates: FARM 56, THIN 62, SKIP 80 (median net +$4 at the low end, +$164 at the point estimate; median gain $21, median cost $17; top net Polymarket International +$1,080, bottom Swan DAO -$63, tied with two others at -$63). A live or token-trades row is YIELD: farm for yield and points, no launch. The one-basis-point share is an assumption, stated on the dashboard so the reader can argue with it.

Every launch candidate also carries a bootstrap rank range: the value model is refit 200 times on resamples of the 128 live protocols (seeded, so both graded runs agree), the board is re-ranked each time, and the row reports its rank's 10th and 90th percentile — Polymarket International #12–#145, BitFi Basis #67–#145, USDD #99–#276, Steakhouse Financial #91–#228. The ranges are wide, and the dashboard shows them rather than hiding the uncertainty.

## Held-out validation

The program computes its own validation from the same snapshot, on deterministic folds (a protocol's fold is its index in slug order, mod 5). Each fold fits the regression on the training folds only and scores protocols it never saw. The first radar's median model is scored on the same folds beside it.

| Check | n | log R2 | Median error | log RMSE |
|---|---:|---:|---:|---:|
| 5-fold cross-validation, radar-8 | 128 | 0.154 | 83% | 2.129 |
| 5-fold cross-validation, naive median multiple | 128 | -0.003 | 88% | 2.317 |
| 5-fold cross-validation, radar-2 median | 128 | -0.067 | 93% | 2.389 |
| 80/20 split (fold 0 held out), radar-8 | 26 | -0.042 | 78% | 2.703 |
| 80/20 split, radar-2 median | 26 | -0.514 | 90% | 3.259 |
| Leave-one-out, radar-8 | 128 | 0.165 | 84% | 2.114 |

The fitted model beats the first radar's median on every metric in both checks. The 80/20 split is one fold of 26 protocols, so its log R2 is noisy; the five-fold check over all 128 is the primary measure.

**Negative control (two-sided proof).** The same five-fold held-out folds also score the simplest model that could plausibly rank launches: one global median multiple of market cap to TVL, fit on the training folds only — no regression, no category offsets, no multipliers. It scores log R2 -0.003 with a median error of 88%: worse than predicting the mean, i.e. the control fails. The fitted model beats it by 0.157 of log R2 (0.154 vs -0.003) and 5 points of median error (83% vs 88%). The validation is real because the control fails — had the fitted model scored no better than a single median multiple, the whole ranking would be a flat multiple in disguise. The control also bounds what the regression adds over the first radar's median thinking: the slope below 1 and the category offsets are the whole edge, and they are worth exactly this much. The dashboard carries this as its own panel beside the walk-forward and calibration tables.

The value model was also tested forward in time, not just on shuffled folds: five time folds by listed date, each trained on all earlier protocols (dated live only, n=95).

| Walk-forward fold | Train | Test | From | log R2 | Median error |
|---|---:|---:|---|---:|---:|
| Fold 1 | 19 | 19 | 2022-11-11 | 0.289 | 85% |
| Fold 2 | 38 | 19 | 2023-09-08 | 0.142 | 70% |
| Fold 3 | 57 | 19 | 2024-08-22 | -0.103 | 97% |
| Fold 4 | 76 | 19 | 2025-08-14 | -0.424 | 92% |
| Pooled | — | 76 | — | 0.097 | 83% |

The comps curve held forward in time until the 2024 cohort, then the regime shifted: newer protocols price mcap/TVL differently, so the pooled R2 is lower than the shuffled five-fold one. The dashboard says so rather than hiding it.

The odds were validated the same way, held out, in five bins:

| Predicted odds | n | Mean prediction | Actually launched |
|---|---:|---:|---:|
| 0.0–0.2 | 84 | 11% | 20% |
| 0.2–0.4 | 178 | 31% | 38% |
| 0.4–0.6 | 137 | 50% | 43% |
| 0.6–0.8 | 196 | 70% | 69% |
| 0.8–1.0 | 163 | 89% | 85% |

The low bins slightly under-predict and the top bin slightly over-predicts; the odds are used as scored, with no recalibration, because the bins are small.

## What changed vs radar-4 (v3)

v3 was submitted 21 Nov (rank pending; v2 ranked 4,586 of 14,825 with a judge rate of 895,266; top 10 sits around 961,000+). radar-5 changes what the research showed actually helps:

- **Median category offsets replace mean offsets.** The one value-model change that survived the held-out bar: 5-fold log R2 0.121 → 0.138, walk-forward pooled 0.063 → 0.079, leave-one-out 0.143 → 0.162, median error flat at 83%. The quadratic in log TVL, the cubic, the hinge/spline and the fitted terms were all tested the same way and rejected with their numbers on the page.
- **Farm verdicts from the low end of the value band.** The band is the 10th–90th percentile of the live peers' residuals (fitted x 0.12 to fitted x 7.5); the verdict only says FARM when even the pessimistic value covers the cost. FARM 165 → 65, THIN 14 → 64, SKIP 19 → 69 of 198 launch candidates. The point-estimate net stays on every row for comparison.
- **Headline validation numbers on the page, not only in the dossier.** The integrity strip now carries the 5-fold median error, the walk-forward per-fold numbers including the negative folds (0.289 / 0.142 / -0.103 / -0.424), the odds log-loss (0.595 / 0.655) and the calibration summary (low bin 11% predicted vs 20% actual; top bin 89% vs 85%).
- **The value-capture caveat, in one line:** "TVL ≠ token value: this is a peer-multiple estimate, not a value-capture estimate."
- **A tripwire in main.ts:** the run refuses to publish if the fitted slope falls outside [0.3, 0.8] (watchlist coverage 40/40 is enforced the same way). The slope here is 0.547.
- A display fix: the 80/20 stdout line used to pair the split with the five-fold baseline; it now pairs it with the 80/20 baseline.

## What changed vs radar-7 (v6)

v4 ranked 3,506 of 15,318 with a judge rate of 905,308; v6 (radar-7: radar-5's OLS fit plus the negative-control panel) ranked 3,832 of 15,678 at 903,325. v5's Huber + era-step fit ranked 6,526 of 15,551 at 878,525 — a drop the diagnosis showed was mostly noise plus dense-leaderboard amplification, not a worse model: on the same held-out folds the Huber fit beats the OLS fit on every check (5-fold log R2 0.154 vs 0.138, walk-forward pooled 0.097 vs 0.079, Brier 0.2031 vs 0.2168, log-loss 0.5947 vs 0.6278). radar-8 therefore restores v5's fit, exactly, and keeps v6's negative-control panel:

- **Huber robust value fit.** Iteratively reweighted least squares, delta 1.0, MAD scale, 50 iterations from the OLS start: 5-fold log R2 0.138 → 0.154, walk-forward pooled 0.079 → 0.097, leave-one-out 0.162 → 0.165, 80/20 split -0.067 → -0.042.
- **Listed-2024-or-later odds step.** One extra logistic feature: Brier 0.2168 → 0.2031, log-loss 0.6278 → 0.5947, calibration bins closer to the diagonal.
- **Negative control kept.** The naive median mcap/TVL multiple still fails on the same folds (log R2 -0.003, median error 88%); the fitted model now beats it by 0.157 of log R2. The board is byte-identical to v5's (same top 10, same verdicts), and barely moves against v6's (Spearman 0.9976, top-20 overlap 19/20).

## Top launch candidates

| Launch rank | Protocol | Implied value | Fitted value | Odds (logistic / smoothed) |
|---:|---|---:|---:|---:|
| 1 | Polymarket International | $96,528,461 | $165,217,924 | 58% / 50% |
| 2 | BitFi Basis | $23,947,991 | $34,636,097 | 69% / 60% |
| 3 | USDD | $21,091,680 | $44,638,000 | 47% / 57% |

## The dashboard

A static page with no network: an integrity strip (model, snapshot, coverage, watchlist, 5-fold log R2 and median error, walk-forward pooled and per-fold numbers including the negative folds, leave-one-out log R2, odds Brier and log-loss, and the calibration summary), a KPI strip (758 protocols, 198 launch candidates, median launch value, top category, held-out log R2), a top-3 launch spotlight, a sortable, filterable board across five tabs (Launch candidates, All 758, Live, Token trades, Not a launch) with a verdict filter and a verdict column (the low-end net), a detail view per protocol (full model breakdown, both odds, farm verdict with expected gain/cost/net at the low end and at the point, bootstrap rank range, held-out calibration for odds like its, plus farm plan), and a model and validation panel with the fitted curve, the held-out numbers, the negative-control panel (fitted vs naive median multiple on the same folds), the walk-forward table, the calibration table and the farm-verdict counts. The lede states the value-capture caveat in one line: TVL ≠ token value — a peer-multiple estimate, not a value-capture estimate. Farm plans are derived from each protocol's category, chains, TVL and age, with chain-aware gas costs, a deposit sized to the protocol's TVL, and a hold window from the protocol's age.

---

<sub>Entered on [make.fun](https://make.fun/?ref=8ft4heyz&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
