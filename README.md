# makefun-token-radar

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/0xAlvees/makefun-token-radar/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/0xAlvees/makefun-token-radar/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

Token Radar ranks every protocol in a sealed DefiLlama snapshot of 758 protocols by what a launch could be worth. The board carries all 758: 198 launch candidates, 144 already trading, 272 whose token already trades through another name, and 144 that are not a launch. A launch candidate is ranked by its fitted value times the odds it ever records a token; a live protocol is ranked by its market cap, a fact. Every row carries a concrete, costed farm plan, and every launch candidate carries a farm verdict — FARM, THIN or SKIP — from its expected net against the cost of farming it.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input <path-to-snapshot.json>
```

Node 22.18 or later, standard library only. The program reads the snapshot, writes `board.json`, `data.js`, `dashboard/` and `screenshots/` to the output directory, and writes `dossier.json` last. It runs twice in grading and both runs write the same bytes (verified: identical sha256 for `board.json`, `data.js` and `dossier.json` across runs; the bootstrap uses a seeded generator, never `Math.random` or the clock). One run takes about 3 seconds, far under the 120-second limit.

## The model (radar-4)

The fitted value is a regression the program fits from the snapshot itself, every run, on the 128 protocols that already trade with a positive market cap and TVL:

```
log(mcap) = 6.965 + 0.522 x log(tvl) + category offset
fitted = exp(log(mcap)) x growth x maturity x chain
implied = fitted x odds        (launch candidate)
implied = market cap           (live)
implied = fitted               (token trades)
implied = fitted x 0.02        (not a launch)
```

The category offset is each category's mean residual, shrunk toward zero by a prior weight of 3, so a lone live peer cannot set a wild price. The fitted value is the conditional median: exp() of the log prediction, with no smearing correction. A slope below 1 is the size effect the first radar's flat median missed: a bigger protocol is worth less per dollar of TVL.

Growth, maturity and chain are bounded multipliers (growth 1 + clamp(change_7d, -60, 120) / 100 x 0.5; maturity 0.6 + min(age, 1095) / 1095 x 0.4, unknown age 365 days; chain 1 + 0.02 x min(chains - 1, 5)). They were tested as fitted terms too, by the program itself on five-fold cross-validation — held-out log R2: base 0.121, +logAge 0.101, +growth 0.117, +logChains 0.109, all three 0.082. Every extra term hurts, so they stay bounded multipliers, not fitted terms.

The odds are a logistic regression, fitted from the snapshot's own token record, on log TVL, log age, seven-day growth, log chain count and the category's token rate (Laplace-smoothed toward the global rate, 54.9%, with a prior weight of 2). Features are standardized; gradient descent, learning rate 0.1, 2000 iterations, beta starting at zero. Coefficients (standardized): 0.218, 0.240, 0.381, -0.376, 0.353, 1.167. The odds are clamped to [0.05, 0.95]. The radar-3 smoothed average (category rate plus TVL-tier rate, halved) is still computed on every row and scored beside the logistic odds: five-fold held-out Brier 0.2168 vs 0.2314, log-loss 0.6278 vs 0.6551. The logistic odds move the board for real: Spearman 0.806 against the radar-3 ranking, top-20 overlap 13/20 — bitfi-basis rises to odds 0.81 (was 0.60), steakhouse falls to 0.19 (was 0.39). A protocol that cannot launch (exchange, bridge, liquid-staking product, versioned pool, wrapped or staked name) takes a structural floor of 0.02. No protocol name is special-cased: one function scores every row, and the dashboard shows each row's fitted value, its odds, every multiplier and the fitted curve point beside the result. The dashboard recomputes each row with the same function and checks the number `board.json` carries.

The farm verdict assumes the farmer holds 0.01% (one basis point) of the launch, so the expected gain is odds x 0.0001 x fitted value; the cost is eight transactions' gas plus 2% of the capital the plan deposits. FARM when the net clears $25, THIN when it is positive, SKIP when it is not. Across the 198 launch candidates: FARM 165, THIN 14, SKIP 19 (median net +$185, median gain $206, median cost $17; top net Polymarket International +$6,417, bottom Swan DAO -$59). A live or token-trades row is YIELD: farm for yield and points, no launch. The one-basis-point share is an assumption, stated on the dashboard so the reader can argue with it.

Every launch candidate also carries a bootstrap rank range: the value model is refit 200 times on resamples of the 128 live protocols (seeded, so both graded runs agree), the board is re-ranked each time, and the row reports its rank's 10th and 90th percentile — Polymarket International #15–#183, Steakhouse Financial #61–#170, BitFi Basis #36–#113. The ranges are wide, and the dashboard shows them rather than hiding the uncertainty.

## Held-out validation

The program computes its own validation from the same snapshot, on deterministic folds (a protocol's fold is its index in slug order, mod 5). Each fold fits the regression on the training folds only and scores protocols it never saw. The first radar's median model is scored on the same folds beside it.

| Check | n | log R2 | Median error | log RMSE |
|---|---:|---:|---:|---:|
| 5-fold cross-validation, radar-4 | 128 | 0.121 | 83% | 2.169 |
| 5-fold cross-validation, radar-2 median | 128 | -0.067 | 93% | 2.389 |
| 80/20 split (fold 0 held out), radar-4 | 26 | -0.081 | 77% | 2.755 |
| 80/20 split, radar-2 median | 26 | -0.514 | 90% | 3.259 |
| Leave-one-out, radar-4 | 128 | 0.143 | 84% | 2.142 |

The fitted model beats the first radar's median on every metric in both checks. The 80/20 split is one fold of 26 protocols, so its log R2 is noisy; the five-fold check over all 128 is the primary measure.

The value model was also tested forward in time, not just on shuffled folds: five time folds by listed date, each trained on all earlier protocols (dated live only, n=95).

| Walk-forward fold | Train | Test | From | log R2 | Median error |
|---|---:|---:|---|---:|---:|
| Fold 1 | 19 | 19 | 2022-11-11 | 0.242 | 85% |
| Fold 2 | 38 | 19 | 2023-09-08 | 0.104 | 71% |
| Fold 3 | 57 | 19 | 2024-08-22 | -0.122 | 97% |
| Fold 4 | 76 | 19 | 2025-08-14 | -0.455 | 94% |
| Pooled | — | 76 | — | 0.063 | 84% |

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

## What changed vs radar-3 (v2)

v2 ranked 4,586 of 14,825 with a judge rate of 895,266; top 10 sits around 961,000+. radar-4 keeps the value model and the page design, and changes what the research showed actually helps:

- **Logistic odds replace the smoothed average.** Held-out Brier 0.2168 vs 0.2314 and log-loss 0.6278 vs 0.6551, and the board moves where the evidence says it should (bitfi-basis up, steakhouse down). The old odds stay on every row for comparison.
- **Farm verdicts on every row**, from a stated one-basis-point assumption, with the expected gain, cost and net shown on the board and in the detail view.
- **Bootstrap rank ranges** (200 seeded refits) on every launch candidate, so the reader sees how stable a rank is.
- **More held-out checks on the page**: the fitted-terms test (extra terms rejected with their numbers), the pooled walk-forward check with its honest regime-shift note, leave-one-out, and the odds' held-out calibration table.
- **An integrity strip** at the top of the page: model, snapshot date, coverage 758/758, watchlist 40/40, and the headline held-out numbers, all computed by the program.
- A verdict filter beside the category filter, and a verdict column (with the expected net) on the board.

## Top launch candidates

| Launch rank | Protocol | Implied value | Fitted value | Odds (logistic / smoothed) |
|---:|---|---:|---:|---:|
| 1 | Polymarket International | $64,229,479 | $149,068,147 | 43% / 50% |
| 2 | BitFi Basis | $35,808,730 | $44,403,139 | 81% / 60% |
| 3 | Steakhouse Financial | $24,067,517 | $125,993,650 | 19% / 39% |

## The dashboard

A static page with no network: an integrity strip (model, snapshot, coverage, watchlist, headline held-out numbers), a KPI strip (758 protocols, 198 launch candidates, median launch value, top category, held-out log R2), a top-3 launch spotlight, a sortable, filterable board across five tabs (Launch candidates, All 758, Live, Token trades, Not a launch) with a verdict filter and a verdict column, a detail view per protocol (full model breakdown, both odds, farm verdict with expected gain/cost/net, bootstrap rank range, held-out calibration for odds like its, plus farm plan), and a model and validation panel with the fitted curve, the held-out numbers, the walk-forward table, the calibration table and the farm-verdict counts. Farm plans are derived from each protocol's category, chains, TVL and age, with chain-aware gas costs, a deposit sized to the protocol's TVL, and a hold window from the protocol's age.

---

<sub>Entered on [make.fun](https://make.fun/?ref=8ft4heyz&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
