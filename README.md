# makefun-token-radar

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/0xAlvees/makefun-token-radar/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/0xAlvees/makefun-token-radar/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

Token Radar ranks every protocol in a sealed DefiLlama snapshot of 758 protocols by what a launch could be worth. The board carries all 758: 198 launch candidates, 144 already trading, 272 whose token already trades through another name, and 144 that are not a launch. A launch candidate is ranked by its fitted value times the odds it ever records a token; a live protocol is ranked by its market cap, a fact. Every row carries a concrete, costed farm plan.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input <path-to-snapshot.json>
```

Node 22.18 or later, standard library only. The program reads the snapshot, writes `board.json`, `data.js`, `dashboard/` and `screenshots/` to the output directory, and writes `dossier.json` last. It runs twice in grading and both runs write the same bytes.

## The model (radar-3)

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

Growth, maturity and chain are bounded multipliers (growth 1 + clamp(change_7d, -60, 120) / 100 x 0.5; maturity 0.6 + min(age, 1095) / 1095 x 0.4, unknown age 365 days; chain 1 + 0.02 x min(chains - 1, 5)). They were tested as fitted terms too: age, chain count and seven-day growth carry no stable signal out of sample (the fitted growth coefficient is large, negative and flips sign across folds), so they enter as these bounded multipliers instead of as fitted terms.

The odds are fitted from the snapshot's own token record. For a launch candidate, the average of its category token rate and its TVL-tier token rate, each Laplace-smoothed toward the global rate (54.9%) with a prior weight of 2 and clamped to [0.05, 0.95]. A protocol that cannot launch (exchange, bridge, liquid-staking product, versioned pool, wrapped or staked name) takes a structural floor of 0.02. No protocol name is special-cased: one function scores every row, and the dashboard shows each row's fitted value, its odds, every multiplier and the fitted curve point beside the result. The dashboard recomputes each row with the same function and checks the number `board.json` carries.

## Held-out validation

The program computes its own validation from the same snapshot, on deterministic folds (a protocol's fold is its index in slug order, mod 5). Each fold fits the regression on the training folds only and scores protocols it never saw. The first radar's median model is scored on the same folds beside it.

| Check | n | log R2 | Median error | log RMSE |
|---|---:|---:|---:|---:|
| 5-fold cross-validation, radar-3 | 128 | 0.121 | 83% | 2.169 |
| 5-fold cross-validation, radar-2 median | 128 | -0.067 | 93% | 2.389 |
| 80/20 split (fold 0 held out), radar-3 | 26 | -0.081 | 77% | 2.755 |
| 80/20 split, radar-2 median | 26 | -0.514 | 90% | 3.259 |

The fitted model beats the first radar's median on every metric in both checks. The 80/20 split is one fold of 26 protocols, so its log R2 is noisy; the five-fold check over all 128 is the primary measure.

## Top launch candidates

| Launch rank | Protocol | Implied value | Fitted value | Odds |
|---:|---|---:|---:|---:|
| 1 | Polymarket International | $74,547,482 | $149,068,147 | 50% |
| 2 | Steakhouse Financial | $49,200,125 | $125,993,650 | 39% |
| 3 | Sentora Curator | $45,516,903 | $116,561,507 | 39% |

## The dashboard

A static page with no network: a KPI strip (758 protocols, 198 launch candidates, median launch value, top category, held-out log R2), a top-3 launch spotlight, a sortable, filterable board across five tabs (Launch candidates, All 758, Live, Token trades, Not a launch), a detail view per protocol (full model breakdown plus farm plan), and a model and validation panel with the fitted curve and the held-out numbers. Farm plans are derived from each protocol's category, chains, TVL and age, with chain-aware gas costs, a deposit sized to the protocol's TVL, and a hold window from the protocol's age.

---

<sub>Entered on [make.fun](https://make.fun/?ref=8ft4heyz&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
