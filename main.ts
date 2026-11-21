// Token Radar's entrypoint (Node 22, standard library only).
//
// It reads the sealed snapshot from /input/snapshot.json (or --input <file|dir>), scores every
// protocol in it with the radar-8 model in lib/model.ts, writes the ranked board to
// /output/board.json, writes the dashboard's data to /output/data.js, copies this repository's
// dashboard/ and screenshots/ to /output, and writes /output/dossier.json with the vendored
// make.fun library. The program runs twice in grading and both runs must write the same bytes,
// so nothing here reads the clock or a random source.
//
// A tripwire runs before anything is written: the fitted slope must lie in [0.3, 0.8] and the
// board must cover the whole watchlist, or the run refuses to publish. A slope outside that
// range means the comps curve no longer looks like a valuation curve at all.
//
// Outside the grader's container, MAKEFUN_OUTPUT_DIR stands in for /output:
//   mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input ../input/snapshot.json

import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { boardIssue, buildRadar, dataScript, parseSnapshot } from './lib/model.ts';
import { buildDossier } from './vendor/makefun-board.ts';

const here = fileURLToPath(new URL('.', import.meta.url));

/** The scorer's own caps: the text cap shapes the bytes, the rest refuse. */
const TEXT_MAX_BYTES = 65_536;
const SCREENSHOT_CAPS = { max: 4, max_bytes: 524_288 };
const IMAGE_NAME = /^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$/;

/** --name <value> out of the arguments, or the fallback. */
function arg(name: string, fallback: string): string {
  const at = process.argv.indexOf(`--${name}`);
  if (at >= 0 && at + 1 < process.argv.length) return process.argv[at + 1]!;
  const withEquals = process.argv.find((a) => a.startsWith(`--${name}=`));
  return withEquals === undefined ? fallback : withEquals.slice(name.length + 3);
}

const inputArg = arg('input', '/input');
const inputFile = inputArg.endsWith('.json') ? inputArg : join(inputArg, 'snapshot.json');
if (!existsSync(inputFile)) {
  process.stderr.write(`token radar: no snapshot at ${inputFile}: pass --input <snapshot.json|directory> or mount the sealed /input\n`);
  process.exit(2);
}
const out = arg('output', process.env.MAKEFUN_OUTPUT_DIR ?? '/output');

const snapshot = parseSnapshot(readFileSync(inputFile, 'utf8'));
const { board, radar } = buildRadar(snapshot);
const issue = boardIssue(board, snapshot);
if (issue !== null) {
  process.stderr.write(`token radar: the model wrote a board the gate refuses: ${issue}\n`);
  process.exit(1);
}

// Tripwire: the fitted slope must lie in [0.3, 0.8] or the run refuses to publish, rather than
// hand in a board whose comps curve no longer looks like a valuation curve. Watchlist coverage
// is the same kind of tripwire and boardIssue above already enforces it (40/40 or refuse).
if (radar.slope < 0.3 || radar.slope > 0.8) {
  process.stderr.write(`token radar: TRIPWIRE: fitted slope ${radar.slope} is outside [0.3, 0.8]: refusing to publish\n`);
  process.exit(1);
}

mkdirSync(out, { recursive: true });
writeFileSync(join(out, 'board.json'), `${JSON.stringify({ rows: board })}\n`);
writeFileSync(join(out, 'data.js'), dataScript(radar));

/** Copy one directory's regular files into `to`, skipping names that start with a dot. */
function copyFiles(from: string, to: string): string[] {
  if (!existsSync(from)) return [];
  mkdirSync(to, { recursive: true });
  const copied: string[] = [];
  for (const name of readdirSync(from)) {
    if (name.startsWith('.')) continue;
    const path = join(from, name);
    if (!statSync(path).isFile()) continue;
    copyFileSync(path, join(to, name));
    copied.push(name);
  }
  return copied;
}

copyFiles(join(here, 'dashboard'), join(out, 'dashboard'));
const shots = copyFiles(join(here, 'screenshots'), join(out, 'screenshots'));
if (shots.length < 3) {
  process.stderr.write(`token radar: screenshots/ holds ${shots.length} image${shots.length === 1 ? '' : 's'}: hand in 3, the board, a detail view and a filtered view\n`);
  process.exit(1);
}
for (const name of shots) {
  if (!IMAGE_NAME.test(name)) {
    process.stderr.write(`token radar: screenshots/${name}: a name is letters, digits and . _ -, 1 to 128 characters\n`);
    process.exit(1);
  }
}

// The dossier is written last: the scorer rebuilds it from this /output and compares bytes.
writeFileSync(join(out, 'dossier.json'), buildDossier(out, { text_max_bytes: TEXT_MAX_BYTES, screenshots: SCREENSHOT_CAPS, dashboard: { max_files: 400, max_bytes: 4_194_304 } }));

const cv = radar.validation.folds[0]!;
const nc = radar.validation.naive_control;
const split = radar.validation.split_80_20;
const baseSplit = radar.validation.baseline_split_80_20;
const wf = radar.validation.walk_forward;
const loo = radar.validation.loo;
const oc = radar.validation.odds_comparison;
const ft = radar.validation.fitted_terms;
process.stdout.write(
  `token radar: ${snapshot.protocols.length} protocols, ${snapshot.watchlist.length} on the watchlist, ${board.length} rows scored (${radar.pre_token_size} pre-token, ${radar.live_size} live, ${radar.token_trades_size} token-trades, ${radar.not_a_launch_size} not-a-launch)\n` +
    `token radar: ${radar.model_version} log(mcap) = ${radar.intercept} + ${radar.slope} x log(tvl), ${radar.validation.train_protocols} live peers, ${cv.name}: log R2 ${cv.log_r2}, median APE ${cv.median_ape}, log RMSE ${cv.log_rmse}\n` +
    `token radar: negative control (same 5-fold held-out folds): naive median mcap/TVL multiple log R2 ${nc.log_r2}, median APE ${nc.median_ape}, log RMSE ${nc.log_rmse} — fitted beats it by ${(cv.log_r2 - nc.log_r2).toFixed(3)} log R2\n` +
    `token radar: value band fitted x ${radar.value_band.lo_mult} to x ${radar.value_band.hi_mult} (p10-p90 of live-peer log residuals, sd ${radar.value_band.resid_sd}); farm verdicts are from the low end\n` +
    `token radar: ${split.name}: log R2 ${split.log_r2}, median APE ${split.median_ape}, log RMSE ${split.log_rmse} (radar-2 median on the same fold: log R2 ${baseSplit.log_r2}, median APE ${baseSplit.median_ape})\n` +
    `token radar: fitted terms (5-fold log R2): ${ft.map((t) => `${t.variant} ${t.log_r2}`).join(', ')}\n` +
    `token radar: walk-forward folds ${wf.folds.map((f) => `#${f.fold} ${f.log_r2}`).join(', ')}, pooled log R2 ${wf.pooled.log_r2}, median APE ${wf.pooled.median_ape} (n=${wf.pooled.test_protocols}); leave-one-out log R2 ${loo.log_r2}, median APE ${loo.median_ape}\n` +
    `token radar: odds (5-fold held-out) logistic Brier ${oc.logistic.brier} log-loss ${oc.logistic.log_loss} vs smoothed Brier ${oc.smoothed.brier} log-loss ${oc.smoothed.log_loss}; logit beta [${radar.validation.logit_coefficients.join(', ')}]\n` +
    `token radar: verdicts (low end) FARM ${radar.verdict_counts.FARM} / THIN ${radar.verdict_counts.THIN} / SKIP ${radar.verdict_counts.SKIP} / YIELD ${radar.verdict_counts.YIELD}; bootstrap ${radar.bootstrap.refits} refits, seed ${radar.bootstrap.seed}\n` +
    `token radar: wrote ${join(out, 'board.json')}, ${join(out, 'data.js')}, ${join(out, 'dashboard')} (3 files), ${join(out, 'screenshots')} (${shots.length} images) and ${join(out, 'dossier.json')}\n`,
);
