// Token Radar's model, radar-3: one snapshot of the protocol universe in, one ranked board out.
//
// The board ranks every protocol in the snapshot (758 rows), strongest implied value first, and
// separates the four kinds of protocol the snapshot actually holds:
//
//   pre-token    a launch candidate: it records no token and passes the watchlist's own type rule
//                (not an exchange, bridge, liquid-staking product or versioned pool, no wrapped or
//                staked word in its name, not named for an exchange). Implied value is the fitted
//                launch value times the odds it ever records a token.
//   live         its token already trades, so the implied value is the market cap itself, a fact,
//                except when the snapshot records that cap as zero or negative, when the fitted
//                value is used instead so no row ever carries an implied value of zero or less.
//   token-trades it records a token (a symbol or a CoinGecko id) but no market cap, almost always
//                a version or a pool of a protocol whose token already trades. Implied value is
//                the fitted value, with no odds: the token is already out there.
//   not-a-launch it records no token but fails the type rule (an exchange, a bridge, a liquid
//                staking product, a versioned pool, a wrapped or staked name). It is almost never
//                a launch, so its odds are a structural floor of 0.02 and it ranks at the bottom.
//
// The fitted launch value is a regression the program fits from the snapshot itself, every run:
//
//   log(mcap) = a + b x log(tvl_usd) + category offset
//
// fit by ordinary least squares on the protocols that already trade (128 of them, market cap and
// TVL both positive). The category offset is each category's mean residual, shrunk toward zero by
// a prior weight of 3, so a lone live peer cannot set a wild price. The fitted value is then:
//
//   fitted = exp(a + b x log(tvl_usd) + offset) x growth x maturity x chain
//
// where growth, maturity and chain are bounded multipliers, the same shape the first radar used:
//
//   growth   = 1 + clamp(change_7d, -60, 120) / 100 x 0.5
//   maturity = 0.6 + min(age_days, 1095) / 1095 x 0.4   (age unknown: 365 days)
//   chain    = 1 + 0.02 x min(chains - 1, 5)
//
// The regression was tested with age, chain count and seven-day growth as fitted terms as well:
// none of them carries a stable signal out of sample (the fitted growth coefficient is large,
// negative and flips sign across folds), so they enter as these bounded multipliers instead of
// as fitted terms, and the dashboard says so. The fitted value is the conditional median of the
// market cap given TVL and category: exp() of the log prediction, with no smearing correction,
// because the residual spread is wide and a mean correction would be set by the largest outliers.
//
// The odds are fitted from the snapshot's own token record: for a launch candidate, the average
// of its category's token rate and its TVL tier's token rate, each Laplace-smoothed toward the
// global rate (0.549) with a prior weight of 2 and clamped to [0.05, 0.95]. A protocol's token
// rate is the fraction of comparable protocols that record a token. The odds are shown beside
// the fitted value on every row, so the reader sees both halves of the claim.
//
// Validation is computed by the program itself, from the same snapshot, on deterministic folds
// (a protocol's fold is its index in slug order, mod 5): a five-fold cross-validation and a
// single 80/20 split, each fitting the regression on the training folds only and scoring the
// held-out fold. The dashboard and the dossier carry those numbers, and the same folds score the
// first radar's median model, so the improvement is measured, not asserted.
//
// No protocol name is ever special-cased: the same functions score every row, and the dashboard
// shows each row's fitted value, its odds, every multiplier and the fitted curve point beside
// the result.

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** The model's constants, as the dashboard states them. */
export interface ModelConstants {
  growth_weight: number;
  growth_floor: number;
  growth_cap: number;
  maturity_floor: number;
  maturity_range: number;
  maturity_days: number;
  unknown_age_days: number;
  peer_prior_weight: number;
  chain_step: number;
  chain_max_bonus: number;
  odds_prior_weight: number;
  odds_floor: number;
  odds_cap: number;
  odds_non_launch: number;
  cv_folds: number;
}

export const MODEL: ModelConstants = {
  growth_weight: 0.5,
  growth_floor: -60,
  growth_cap: 120,
  maturity_floor: 0.6,
  maturity_range: 0.4,
  maturity_days: 1095,
  unknown_age_days: 365,
  peer_prior_weight: 3,
  chain_step: 0.02,
  chain_max_bonus: 5,
  odds_prior_weight: 2,
  odds_floor: 0.05,
  odds_cap: 0.95,
  odds_non_launch: 0.02,
  cv_folds: 5,
};

/** The model's version label: the dashboard names it. */
export const MODEL_VERSION = 'radar-3';

/** How many live-token protocols the Live view carries, by market cap. */
export const LIVE_ROWS = 24;

/** A farm step: the action, what it costs, and how long it takes. */
export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

/** The four kinds of protocol the board separates. */
export type TokenStatus = 'pre-token' | 'live' | 'token-trades' | 'not-a-launch';

/** One row of board.json: the ranked board's row. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: TokenStatus;
  age_days: number | null;
  implied_value_usd: number;
  model: {
    fitted_value_usd: number;
    token_odds: number | null;
    intercept: number;
    slope: number;
    category_offset: number;
    category_live_peers: number;
    growth_pct: number;
    growth_multiplier: number;
    maturity_multiplier: number;
    chain_multiplier: number;
    age_days: number | null;
  };
  farm_steps: string[];
}

/** The dashboard's compact view of one row, with farm plans generated in the page. */
export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  age_days: number | null;
  implied_value_usd: number;
  token_status: TokenStatus;
  fitted_value_usd: number;
  token_odds: number | null;
  intercept: number;
  slope: number;
  category_offset: number;
  category_live_peers: number;
  growth_multiplier: number;
  maturity_multiplier: number;
  chain_multiplier: number;
}

/** A live-token protocol: it already trades, so the board shows the market's own value. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

/** One held-out validation: a fitted model scored on protocols it never saw. */
export interface ValidationFold {
  name: string;
  test_protocols: number;
  log_r2: number;
  median_ape: number;
  mape: number;
  log_rmse: number;
}

/** The validation the program computed, on deterministic folds, with the first radar beside it. */
export interface Validation {
  model: string;
  train_protocols: number;
  folds: ValidationFold[];
  split_80_20: ValidationFold;
  baseline: ValidationFold;
  baseline_split_80_20: ValidationFold;
}

/** The fitted curve the dashboard draws: the regression's line and the live peers behind it. */
export interface CurvePoint {
  log_tvl: number;
  log_mcap: number;
  slug: string;
  category: string | null;
}

/** The odds table the dashboard recomputes a row's odds from: counts per category and TVL tier. */
export interface OddsData {
  global: number;
  by_category: Record<string, { tok: number; n: number }>;
  by_tier: Record<string, { tok: number; n: number }>;
}

/** Everything the dashboard needs: the board, the live set, the model and its validation. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model: ModelConstants;
  model_version: string;
  intercept: number;
  slope: number;
  global_token_rate: number;
  watchlist_size: number;
  board_size: number;
  pre_token_size: number;
  live_size: number;
  token_trades_size: number;
  not_a_launch_size: number;
  validation: Validation;
  curve: CurvePoint[];
  odds: OddsData;
  rows: RadarRow[];
  live: LiveRow[];
}

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;

/** A finite number, else `fallback`. */
function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** The median of a sorted array, or null when it is empty. */
function medianOf(sorted: number[]): number | null {
  if (sorted.length === 0) return null;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

/** A whole day count from `from` to `to`, clamped at zero; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** Whether a protocol records no token at all: no market cap, no CoinGecko id, no symbol. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** DefiLlama's version-marker rule: a name ending in a release is a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

/** A whole word `wrapped`, `staked` or `staking` in a name: a wrapped or staked product. */
const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;

/** The exchange names the watchlist rule treats as an exchange-named protocol. */
const EXCHANGE_NAMES = ['binance', 'coinbase', 'bybit', 'kraken', 'bitget', 'okx', 'bitfinex', 'gemini'];

/** Whether a name begins with an exchange's name. */
function isExchangeNamed(name: string): boolean {
  const lower = name.toLowerCase();
  return EXCHANGE_NAMES.some((e) => lower === e || lower.startsWith(`${e} `) || lower.startsWith(`${e}-`) || lower.startsWith(`${e}:`));
}

/** The categories the watchlist rule treats as not a launch: exchanges, bridges, liquid staking. */
const NON_LAUNCH_CATEGORIES = new Set(['CEX', 'Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC']);

/**
 * Whether a protocol is a launch candidate: it records no token and passes the watchlist's own
 * type rule. This is the same rule the house used to draw the watchlist, minus the TVL floor.
 */
export function isLaunchCandidate(p: SnapshotProtocol): boolean {
  if (!isPreToken(p)) return false;
  if (p.category !== null && NON_LAUNCH_CATEGORIES.has(p.category)) return false;
  if (VERSIONED.test(p.name)) return false;
  if (WRAPPED_WORD.test(p.name)) return false;
  if (isExchangeNamed(p.name)) return false;
  return true;
}

/** A protocol's kind on the board: launch, live, token already trades, or not a launch. */
export function tokenStatus(p: SnapshotProtocol): TokenStatus {
  if (p.mcap_usd !== null) return 'live';
  if (p.gecko_id !== null || p.symbol !== null) return 'token-trades';
  return isLaunchCandidate(p) ? 'pre-token' : 'not-a-launch';
}

/** The snapshot's protocol by slug. */
function bySlug(protocols: SnapshotProtocol[]): Map<string, SnapshotProtocol> {
  return new Map(protocols.map((p) => [p.slug, p]));
}

/** The protocols the regression trains on: a market cap and a TVL, both positive. */
export function trainSet(protocols: SnapshotProtocol[]): SnapshotProtocol[] {
  return protocols
    .filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0)
    .sort((a, b) => (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0));
}

/** The ordinary-least-squares fit of log(mcap) on log(TVL): intercept and slope, closed form. */
export function olsLogLog(train: SnapshotProtocol[]): { intercept: number; slope: number } {
  const n = train.length;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (const p of train) {
    const x = Math.log(p.tvl_usd);
    const y = Math.log(p.mcap_usd as number);
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const denom = n * sxx - sx * sx;
  const slope = denom === 0 ? 0 : (n * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / n;
  return { intercept, slope };
}

/**
 * Each category's offset: its mean residual from the global fit, shrunk toward zero by the prior
 * weight, so a lone live peer cannot set a wild price. A category with no live peer gets zero.
 */
export function categoryOffsets(
  train: SnapshotProtocol[],
  fit: { intercept: number; slope: number },
  model: ModelConstants = MODEL,
): Map<string, { offset: number; peers: number }> {
  const sums = new Map<string, { sum: number; n: number }>();
  for (const p of train) {
    const resid = Math.log(p.mcap_usd as number) - (fit.intercept + fit.slope * Math.log(p.tvl_usd));
    const key = p.category ?? 'Other';
    const e = sums.get(key) ?? { sum: 0, n: 0 };
    e.sum += resid;
    e.n += 1;
    sums.set(key, e);
  }
  const out = new Map<string, { offset: number; peers: number }>();
  for (const [key, e] of sums) {
    out.set(key, { offset: (e.n * (e.sum / e.n)) / (e.n + model.peer_prior_weight), peers: e.n });
  }
  return out;
}

/** The fitted log market cap of one protocol: the regression plus its category offset. */
export function fittedLog(
  protocol: SnapshotProtocol,
  fit: { intercept: number; slope: number },
  offsets: Map<string, { offset: number; peers: number }>,
): { log: number; offset: number; peers: number } {
  const entry = offsets.get(protocol.category ?? 'Other');
  const offset = entry === undefined ? 0 : entry.offset;
  const peers = entry === undefined ? 0 : entry.peers;
  return { log: fit.intercept + fit.slope * Math.log(Math.max(1, protocol.tvl_usd)) + offset, offset, peers };
}

/** The three bounded multipliers: growth, maturity from age, and chain count. */
export function multipliers(
  protocol: SnapshotProtocol,
  asOf: string,
  model: ModelConstants = MODEL,
): { growthPct: number; growth: number; maturity: number; chain: number; age: number | null } {
  const age = ageDays(asOf, protocol.listed_at);
  const rawGrowth = finite(protocol.change_7d, 0);
  const growthPct = Math.min(model.growth_cap, Math.max(model.growth_floor, rawGrowth));
  const growth = 1 + (growthPct / 100) * model.growth_weight;
  const ageForModel = age ?? model.unknown_age_days;
  const maturity = model.maturity_floor + (Math.min(Math.max(ageForModel, 0), model.maturity_days) / model.maturity_days) * model.maturity_range;
  const extraChains = Math.min(Math.max(protocol.chains.length - 1, 0), model.chain_max_bonus);
  const chain = 1 + model.chain_step * extraChains;
  return { growthPct, growth, maturity, chain, age };
}

/** The fitted value of one protocol: the regression's conditional median, times the multipliers. */
export function fittedValue(
  protocol: SnapshotProtocol,
  asOf: string,
  fit: { intercept: number; slope: number },
  offsets: Map<string, { offset: number; peers: number }>,
  model: ModelConstants = MODEL,
): { fitted: number; log: number; offset: number; peers: number; growthPct: number; growth: number; maturity: number; chain: number; age: number | null } {
  const f = fittedLog(protocol, fit, offsets);
  const m = multipliers(protocol, asOf, model);
  const fitted = Math.max(1, Math.round(Math.exp(f.log) * m.growth * m.maturity * m.chain));
  return { fitted, log: f.log, offset: f.offset, peers: f.peers, growthPct: m.growthPct, growth: m.growth, maturity: m.maturity, chain: m.chain, age: m.age };
}

/** A TVL tier for the odds: the same five bands the dashboard names. */
export function tvlTier(tvlUsd: number): string {
  if (tvlUsd < 10_000_000) return '<10M';
  if (tvlUsd < 50_000_000) return '10-50M';
  if (tvlUsd < 200_000_000) return '50-200M';
  if (tvlUsd < 1_000_000_000) return '200M-1B';
  return '>1B';
}

/** Whether a protocol records any token: a market cap, a CoinGecko id or a symbol. */
export function hasTokenRecord(p: SnapshotProtocol): boolean {
  return p.mcap_usd !== null || p.gecko_id !== null || p.symbol !== null;
}

/**
 * The token rates the snapshot supplies: the global rate, and each category's and each TVL
 * tier's rate, Laplace-smoothed toward the global one. A rate is the fraction of comparable
 * protocols that record a token. The counts ride along so the dashboard can recompute a row's
 * odds from the same numbers, and so its check of board.json is a real recompute.
 */
export interface OddsTable {
  global: number;
  byCategory: Map<string, number>;
  byTier: Map<string, number>;
  catCounts: Map<string, { tok: number; n: number }>;
  tierCounts: Map<string, { tok: number; n: number }>;
}

export function oddsTable(protocols: SnapshotProtocol[], model: ModelConstants = MODEL): OddsTable {
  const total = protocols.length;
  const withToken = protocols.filter(hasTokenRecord).length;
  const global = total === 0 ? 0.5 : withToken / total;
  const smooth = (tok: number, n: number): number => (tok + model.odds_prior_weight * global) / (n + model.odds_prior_weight);
  const catTok = new Map<string, number>();
  const catTot = new Map<string, number>();
  const tierTok = new Map<string, number>();
  const tierTot = new Map<string, number>();
  for (const p of protocols) {
    const c = p.category ?? 'Other';
    const t = tvlTier(p.tvl_usd);
    catTot.set(c, (catTot.get(c) ?? 0) + 1);
    tierTot.set(t, (tierTot.get(t) ?? 0) + 1);
    if (hasTokenRecord(p)) {
      catTok.set(c, (catTok.get(c) ?? 0) + 1);
      tierTok.set(t, (tierTok.get(t) ?? 0) + 1);
    }
  }
  const byCategory = new Map<string, number>();
  const catCounts = new Map<string, { tok: number; n: number }>();
  for (const [c, n] of catTot) {
    byCategory.set(c, smooth(catTok.get(c) ?? 0, n));
    catCounts.set(c, { tok: catTok.get(c) ?? 0, n });
  }
  const byTier = new Map<string, number>();
  const tierCounts = new Map<string, { tok: number; n: number }>();
  for (const [t, n] of tierTot) {
    byTier.set(t, smooth(tierTok.get(t) ?? 0, n));
    tierCounts.set(t, { tok: tierTok.get(t) ?? 0, n });
  }
  return { global, byCategory, byTier, catCounts, tierCounts };
}

/** A launch candidate's odds: the average of its category rate and its TVL-tier rate, clamped. */
export function tokenOdds(protocol: SnapshotProtocol, table: OddsTable, model: ModelConstants = MODEL): number {
  const cat = table.byCategory.get(protocol.category ?? 'Other') ?? table.global;
  const tier = table.byTier.get(tvlTier(protocol.tvl_usd)) ?? table.global;
  const odds = (cat + tier) / 2;
  return Math.min(model.odds_cap, Math.max(model.odds_floor, odds));
}

const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;

/** One validation metric set: log R-squared, median and mean absolute percentage error, log RMSE. */
function validationMetrics(actualLog: number[], predLog: number[]): { logR2: number; medianApe: number; mape: number; logRmse: number } {
  const n = actualLog.length;
  const mean = actualLog.reduce((a, b) => a + b, 0) / n;
  let ssRes = 0;
  let ssTot = 0;
  let sqErr = 0;
  const apes: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = actualLog[i]!;
    const p = predLog[i]!;
    ssRes += (a - p) * (a - p);
    ssTot += (a - mean) * (a - mean);
    sqErr += (a - p) * (a - p);
    apes.push(Math.abs(Math.exp(a) - Math.exp(p)) / Math.exp(a));
  }
  apes.sort((a, b) => a - b);
  const medianApe = medianOf(apes) ?? 0;
  const mape = apes.reduce((a, b) => a + b, 0) / n;
  return {
    logR2: ssTot === 0 ? 0 : 1 - ssRes / ssTot,
    medianApe: round4(medianApe),
    mape: round4(mape),
    logRmse: round4(Math.sqrt(sqErr / n)),
  };
}

/**
 * The first radar's median model (radar-2) as a held-out predictor, on the same folds, for
 * comparison: the category median mcap/TVL shrunk toward the global median with a prior weight
 * of 4 and clamped to [0.05, 1.5], times the same three bounded multipliers the first radar
 * used. Its peer table includes the protocols with a zero or negative market cap, exactly as
 * the first radar's did, because those are in its training set and never in the test set.
 */
function baselinePredictions(trainAll: SnapshotProtocol[], test: SnapshotProtocol[], asOf: string): number[] {
  const live = trainAll.filter((p) => p.mcap_usd !== null && p.tvl_usd > 0);
  const ratios = live.map((p) => (p.mcap_usd as number) / p.tvl_usd).sort((a, b) => a - b);
  const globalMedian = medianOf(ratios) ?? 0.17;
  const byCat = new Map<string, number[]>();
  for (const p of live) {
    const key = p.category ?? 'Other';
    const list = byCat.get(key) ?? [];
    list.push((p.mcap_usd as number) / p.tvl_usd);
    byCat.set(key, list);
  }
  const multiples = new Map<string, number>();
  for (const [key, rs] of byCat) {
    const sorted = [...rs].sort((a, b) => a - b);
    const catMedian = medianOf(sorted);
    const n = sorted.length;
    const shrunk = catMedian === null ? globalMedian : (n * catMedian + 4 * globalMedian) / (n + 4);
    multiples.set(key, Math.min(1.5, Math.max(0.05, shrunk)));
  }
  return test.map((p) => {
    const peer = multiples.get(p.category ?? 'Other') ?? globalMedian;
    const g = Math.min(120, Math.max(-60, finite(p.change_7d, 0)));
    const gm = 1 + (g / 100) * 0.5;
    const age = ageDays(asOf, p.listed_at);
    const ageForModel = age ?? 365;
    const mm = 0.6 + (Math.min(Math.max(ageForModel, 0), 1095) / 1095) * 0.4;
    const extraChains = Math.min(Math.max(p.chains.length - 1, 0), 5);
    const cm = 1 + 0.02 * extraChains;
    return Math.log(Math.max(1, p.tvl_usd * peer * gm * mm * cm));
  });
}

/**
 * The validation: five-fold cross-validation (a protocol's fold is its index in slug order,
 * mod 5) and a single 80/20 split (fold 0 held out), each fitting the regression on the
 * training folds only, with the first radar's median model scored on the same folds beside it.
 * The baseline's training set for a fold is the same training folds plus the protocols with a
 * zero or negative market cap, which the first radar's peer table counted and which are never
 * in the test set.
 */
export function validate(train: SnapshotProtocol[], snapshot: Snapshot, model: ModelConstants = MODEL): Validation {
  const folds = model.cv_folds;
  const foldOf = (i: number): number => i % folds;
  const zeroMcap = snapshot.protocols.filter((p) => p.mcap_usd !== null && (p.mcap_usd as number) <= 0 && p.tvl_usd > 0);
  const cvPred: number[] = new Array(train.length).fill(0);
  const cvBase: number[] = new Array(train.length).fill(0);
  for (let f = 0; f < folds; f++) {
    const tr = train.filter((_, i) => foldOf(i) !== f);
    const te = train.filter((_, i) => foldOf(i) === f);
    const fit = olsLogLog(tr);
    const offs = categoryOffsets(tr, fit, model);
    te.forEach((p) => {
      const i = train.indexOf(p);
      cvPred[i] = fittedLog(p, fit, offs).log;
    });
    const base = baselinePredictions([...tr, ...zeroMcap], te, snapshot.as_of);
    te.forEach((p, j) => {
      cvBase[train.indexOf(p)] = base[j]!;
    });
  }
  const actualLog = train.map((p) => Math.log(p.mcap_usd as number));
  const cv = validationMetrics(actualLog, cvPred);
  const cvB = validationMetrics(actualLog, cvBase);

  const testIdx = train.map((_, i) => i).filter((i) => foldOf(i) === 0);
  const trainIdx = train.map((_, i) => i).filter((i) => foldOf(i) !== 0);
  const tr = trainIdx.map((i) => train[i]!);
  const te = testIdx.map((i) => train[i]!);
  const fit = olsLogLog(tr);
  const offs = categoryOffsets(tr, fit, model);
  const splitPred = te.map((p) => fittedLog(p, fit, offs).log);
  const splitBase = baselinePredictions([...tr, ...zeroMcap], te, snapshot.as_of);
  const split = validationMetrics(te.map((p) => Math.log(p.mcap_usd as number)), splitPred);
  const splitB = validationMetrics(te.map((p) => Math.log(p.mcap_usd as number)), splitBase);

  return {
    model: MODEL_VERSION,
    train_protocols: train.length,
    folds: [{ name: `${folds}-fold cross-validation`, test_protocols: train.length, log_r2: round4(cv.logR2), median_ape: cv.medianApe, mape: cv.mape, log_rmse: cv.logRmse }],
    split_80_20: { name: '80/20 split (fold 0 held out)', test_protocols: te.length, log_r2: round4(split.logR2), median_ape: split.medianApe, mape: split.mape, log_rmse: split.logRmse },
    baseline: { name: `${folds}-fold cross-validation (radar-2 median)`, test_protocols: train.length, log_r2: round4(cvB.logR2), median_ape: cvB.medianApe, mape: cvB.mape, log_rmse: cvB.logRmse },
    baseline_split_80_20: { name: '80/20 split (radar-2 median)', test_protocols: te.length, log_r2: round4(splitB.logR2), median_ape: splitB.medianApe, mape: splitB.mape, log_rmse: splitB.logRmse },
  };
}

const round2 = (n: number): number => Math.round(n * 100) / 100;

/** The protocols the board ranks, strongest first, with ranks 1..n: every protocol in the snapshot. */
export function buildBoard(snapshot: Snapshot, model: ModelConstants = MODEL): { board: BoardRow[]; fit: { intercept: number; slope: number }; offsets: Map<string, { offset: number; peers: number }>; odds: OddsTable; validation: Validation } {
  const train = trainSet(snapshot.protocols);
  const fit = olsLogLog(train);
  const offsets = categoryOffsets(train, fit, model);
  const odds = oddsTable(snapshot.protocols, model);
  const validation = validate(train, snapshot, model);

  const scored = snapshot.protocols.map((p) => {
    const status = tokenStatus(p);
    const v = fittedValue(p, snapshot.as_of, fit, offsets, model);
    let implied: number;
    let oddsValue: number | null;
    if (status === 'live') {
      // The market cap is the fact, except when the snapshot records it as zero or negative:
      // a token that trades but whose market cap is not a positive number takes the fitted
      // value instead, so no row ever carries an implied value of zero or less.
      implied = p.mcap_usd !== null && p.mcap_usd > 0 ? p.mcap_usd : v.fitted;
      oddsValue = null;
    } else if (status === 'token-trades') {
      implied = v.fitted;
      oddsValue = null;
    } else if (status === 'not-a-launch') {
      implied = Math.max(1, Math.round(v.fitted * model.odds_non_launch));
      oddsValue = model.odds_non_launch;
    } else {
      oddsValue = tokenOdds(p, odds, model);
      implied = Math.max(1, Math.round(v.fitted * oddsValue));
    }
    return { p, status, v, implied, tokenOdds: oddsValue };
  });
  scored.sort((a, b) => b.implied - a.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  const board: BoardRow[] = scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    mcap_usd: s.p.mcap_usd,
    listed_at: s.p.listed_at,
    gecko_id: s.p.gecko_id,
    symbol: s.p.symbol,
    token_status: s.status,
    age_days: s.v.age,
    implied_value_usd: s.implied,
    model: {
      fitted_value_usd: s.v.fitted,
      token_odds: s.tokenOdds === null ? null : round4(s.tokenOdds),
      // Full precision here, not rounded: the dashboard recomputes each row from these
      // numbers and its check of board.json must come out exact.
      intercept: fit.intercept,
      slope: fit.slope,
      category_offset: s.v.offset,
      category_live_peers: s.v.peers,
      growth_pct: s.v.growthPct,
      growth_multiplier: round4(s.v.growth),
      maturity_multiplier: round4(s.v.maturity),
      chain_multiplier: round4(s.v.chain),
      age_days: s.v.age,
    },
    farm_steps: farmSteps(s.p, s.v.age, s.status).map(stepText),
  }));
  return { board, fit, offsets, odds, validation };
}

/** A step as one string: its action, then its cost and time, within the gate's 20..600 characters. */
export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

/** The first chain a protocol names, or a neutral phrase. */
function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

/** What one transaction costs on a chain, in dollars: L1s are dear, L2s and Solana are cheap. */
function gasFor(chain: string): string {
  const c = chain.toLowerCase();
  if (c === 'ethereum') return 'about $8 of gas';
  if (c === 'solana') return 'about $0.05 of gas';
  if (c === 'base' || c === 'arbitrum' || c === 'optimism' || c === 'polygon' || c === 'monad' || c === 'hyperliquid l1' || c === 'plasma' || c === 'arc' || c === 'avalanche' || c === 'bsc' || c === 'binance') return 'about $0.50 of gas';
  return 'about $2 of gas';
}

/** How much capital one farm position needs, from the protocol's own TVL. */
function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  return 50;
}

/** How long to hold, from the protocol's age: younger protocols get a longer qualifying window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** The category's primary action, in the words a farmer can follow. */
function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `Supply $${capital} of USDC to ${name} on ${chain}, then borrow a small amount against it and repay it once.`;
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm')) return `Provide $${capital} of liquidity to ${name}'s main pool on ${chain} and keep the position open.`;
  if (c.includes('bridge') || c.includes('cross chain')) return `Bridge $${capital} of USDC to ${chain} through ${name} and use the bridged funds in its app.`;
  if (c.includes('staking') || c.includes('restak')) return `Stake $${capital} of the chain's staking asset through ${name} and keep the receipt token.`;
  if (c.includes('perpetual') || c.includes('derivative') || c.includes('basis trad')) return `Deposit $${capital} of USDC as margin on ${name} and open one small hedged trade.`;
  if (c.includes('prediction')) return `Fund $${capital} and place one market position on ${name}.`;
  if (c.includes('curator') || c.includes('allocator') || c.includes('risk')) return `Deposit $${capital} into a ${name} vault on ${chain} and keep the shares.`;
  if (c.includes('stable')) return `Mint or deposit $${capital} of the stable asset through ${name} on ${chain}.`;
  if (c.includes('rwa')) return `Mint $${capital} of the protocol's yield-bearing asset on ${chain} and hold it.`;
  if (c.includes('insurance')) return `Buy $${capital} of cover from ${name} on ${chain} and keep the policy active.`;
  if (c.includes('payment')) return `Send $${capital} through ${name} on ${chain} and keep the receipt.`;
  if (c.includes('chain')) return `Use $${capital} in ${name}'s app on ${chain} and keep the position open.`;
  return `Deposit $${capital} of USDC into ${name} on ${chain} and keep the position open.`;
}

/** The category's weekly qualifying action, so the plan is not the same sentence twice. */
function weeklyFor(p: SnapshotProtocol): string {
  const c = (p.category ?? '').toLowerCase();
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return 'Make one qualifying action each week, such as a supply, a borrow or a repayment.';
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm')) return 'Make one qualifying action each week, such as a swap, a deposit or a claim of pool fees.';
  if (c.includes('bridge') || c.includes('cross chain')) return 'Make one qualifying action each week, such as a bridge, a swap or a deposit on the destination chain.';
  if (c.includes('perpetual') || c.includes('derivative') || c.includes('basis trad')) return 'Make one qualifying action each week, such as a trade, a deposit or a funding payment.';
  if (c.includes('staking') || c.includes('restak')) return 'Make one qualifying action each week, such as a stake, a restake or a vote.';
  if (c.includes('prediction')) return 'Make one qualifying action each week, such as a market position or a settlement.';
  return 'Make one qualifying action each week, such as a deposit, a swap, a vote or a claim.';
}

/**
 * A protocol's farm plan: six steps, each an action with a cost and a time. The plan is derived
 * from the protocol's own record (category, chains, TVL, age) and its status on the board: a
 * launch candidate's plan is written for an airdrop, a live protocol's for yield and points, and
 * a token-trades or not-a-launch row's for the position itself, never for a launch that is not
 * coming. No name is special-cased.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null, status: TokenStatus): FarmStep[] {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const gas = gasFor(chain);
  const wallet = chain.toLowerCase() === 'solana'
    ? 'Use a fresh Solana wallet that holds only the funds you plan to farm, and keep its seed phrase offline.'
    : 'Use a fresh EVM wallet that holds only the funds you plan to farm, and keep its seed phrase offline.';
  const c = (p.category ?? '').toLowerCase();
  const claimWord = c.includes('curator') || c.includes('allocator') || c.includes('risk') || c.includes('yield') ? 'points' : 'rewards';
  const claimText = status === 'pre-token'
    ? `Before you withdraw, open the protocol's own ${claimWord} page and claim only from the link it publishes; a real airdrop never asks you to pay to claim.`
    : status === 'live'
      ? `Before you withdraw, open the protocol's own ${claimWord} page and claim only from the link it publishes; a real distribution never asks you to pay to claim.`
      : `Before you withdraw, open the protocol's own ${claimWord} page and claim only from the link it publishes; never pay a fee to claim a reward.`;
  return [
    { text: wallet, cost: 'nothing but the wallet', time: '5 minutes' },
    { text: actionFor(p, capital, chain), cost: `$${capital} of capital and ${gas}`, time: '15 minutes to set up' },
    { text: weeklyFor(p), cost: `${gas} a week`, time: '10 minutes a week' },
    { text: `Hold the position for at least ${hold} days so the activity is counted in more than one snapshot.`, cost: `the $${capital} stays in the position`, time: `${hold} days` },
    { text: claimText, cost: 'no fee to check', time: '10 minutes' },
    { text: `Write the wallet, the dates and every transaction in a farm log so you can repeat the position or unwind it.`, cost: 'nothing but the log', time: '5 minutes' },
  ];
}

/** The live-token protocols, by market cap, for the dashboard's Live view. */
export function liveRows(snapshot: Snapshot): LiveRow[] {
  return snapshot.protocols
    .filter((p) => p.mcap_usd !== null)
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, LIVE_ROWS)
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      change_7d: p.change_7d,
    }));
}

/** A board row's compact form for data.js, with the farm plan generated in the page. */
export function radarRow(row: BoardRow): RadarRow {
  return {
    rank: row.rank,
    slug: row.slug,
    name: row.name,
    category: row.category,
    chains: row.chains,
    tvl_usd: row.tvl_usd,
    change_7d: row.change_7d,
    mcap_usd: row.mcap_usd,
    listed_at: row.listed_at,
    age_days: row.age_days,
    implied_value_usd: row.implied_value_usd,
    token_status: row.token_status,
    fitted_value_usd: row.model.fitted_value_usd,
    token_odds: row.model.token_odds,
    intercept: row.model.intercept,
    slope: row.model.slope,
    category_offset: row.model.category_offset,
    category_live_peers: row.model.category_live_peers,
    growth_multiplier: row.model.growth_multiplier,
    maturity_multiplier: row.model.maturity_multiplier,
    chain_multiplier: row.model.chain_multiplier,
  };
}

/** Everything the dashboard renders, from the snapshot. */
export function buildRadar(snapshot: Snapshot, model: ModelConstants = MODEL): { board: BoardRow[]; radar: RadarData } {
  const { board, fit, offsets, odds, validation } = buildBoard(snapshot, model);
  const live = liveRows(snapshot);
  const train = trainSet(snapshot.protocols);
  const curve: CurvePoint[] = train.map((p) => ({
    log_tvl: round4(Math.log(p.tvl_usd)),
    log_mcap: round4(Math.log(p.mcap_usd as number)),
    slug: p.slug,
    category: p.category,
  }));
  const counts = { pre: 0, live: 0, trades: 0, notLaunch: 0 };
  for (const r of board) {
    if (r.token_status === 'pre-token') counts.pre += 1;
    else if (r.token_status === 'live') counts.live += 1;
    else if (r.token_status === 'token-trades') counts.trades += 1;
    else counts.notLaunch += 1;
  }
  const oddsData: OddsData = { global: odds.global, by_category: {}, by_tier: {} };
  for (const [c, v] of odds.catCounts) oddsData.by_category[c] = v;
  for (const [t, v] of odds.tierCounts) oddsData.by_tier[t] = v;
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model,
    model_version: MODEL_VERSION,
    // Full precision here, not rounded: the dashboard draws the fitted curve from these.
    intercept: fit.intercept,
    slope: fit.slope,
    global_token_rate: round4(odds.global),
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    pre_token_size: counts.pre,
    live_size: counts.live,
    token_trades_size: counts.trades,
    not_a_launch_size: counts.notLaunch,
    validation,
    curve,
    odds: oddsData,
    rows: board.map(radarRow),
    live,
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return {
    as_of: o.as_of,
    source: o.source,
    floor_usd: o.floor_usd,
    protocols,
    watchlist: o.watchlist as string[],
  };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, one shared implied value, and farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}: a board that ranks nothing by value is not a ranking`;
  }
  for (const r of board) {
    const steps = r.farm_steps;
    if (steps.length < 3 || steps.length > 12) return `${r.slug} has ${steps.length} farm steps`;
    for (const s of steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
