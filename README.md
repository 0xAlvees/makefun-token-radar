# makefun-token-radar

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/0xAlvees/makefun-token-radar/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/0xAlvees/makefun-token-radar/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

Token Radar ranks the protocols most likely to launch a valuable token, from a sealed DefiLlama snapshot of 758 protocols. The board covers all 40 watchlist protocols (TVL of at least $50M, no token on record) and adds 160 pre-token protocols below that floor, each ranked by an implied value and handed a concrete farm plan.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input <path-to-snapshot.json>
```

Node 22.18 or later, standard library only. The program reads the snapshot, writes `board.json`, `data.js`, `dashboard/` and `screenshots/` to the output directory, and writes `dossier.json` last. It runs twice in grading and both runs write the same bytes.

## The model (radar-2)

Implied value follows only from the snapshot's own figures, through a peer multiple the snapshot itself supplies:

```
implied = round(tvl_usd x peer multiple
  x (1 + clamp(change_7d, -60, 120) / 100 x 0.5)
  x (0.6 + min(age_days, 1095) / 1095 x 0.4)
  x (1 + 0.02 x min(chains - 1, 5)))
```

The peer multiple is each category's median market-cap-to-TVL ratio among its protocols that already trade, shrunk toward the global median (0.17x) by a prior weight of 4 and clamped to [0.05, 1.50]. A category with no live peer takes the global median. No protocol name is special-cased: one function scores every row, and the dashboard shows each row's peer multiple, peer count and every multiplier beside the result. The dashboard recomputes each row with the same function and checks the number `board.json` carries.

## The dashboard

A static page with no network: a sortable, filterable board, a top-3 spotlight, a detail view per protocol (model breakdown plus farm plan) and a live-token view set aside with market caps. Farm plans are derived from each protocol's category, chains, TVL and age, with chain-aware gas costs and a hold window from the protocol's age.

---

<sub>Entered on [make.fun](https://make.fun/?ref=8ft4heyz&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
