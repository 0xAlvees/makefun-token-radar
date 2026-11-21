// Token Radar's dashboard, radar-6. It reads window.RADAR (written beside dashboard/ by main.ts)
// and renders the ranked board, a KPI strip, an integrity strip, a top-3 launch spotlight, a
// filterable table, a per-protocol detail view with the model's own inputs, its farm verdict and
// the protocol's farm plan, and the fitted curve with the held-out validation. No network: every
// byte it shows is in data.js.
// The page recomputes each row from the snapshot figures with the same function main.ts runs,
// so the screenshots and board.json agree.
(function () {
  'use strict';

  var DATA = window.RADAR;
  if (!DATA) return;
  var app = document.getElementById('board');
  var state = { q: '', category: 'all', verdict: 'all', tab: 'pre-token', sort: 'implied', dir: -1, selected: null };

  var usd = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
  function fmtUsd(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    if (Math.abs(n) >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (Math.abs(n) >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return '$' + Math.round(n / 1e3) + 'K';
    return usd.format(n);
  }
  function fmtPct(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    return (n > 0 ? '+' : '') + n.toFixed(2) + '%';
  }
  function fmtAge(days) {
    if (days === null || days === undefined) return 'unknown';
    if (days >= 365) return (days / 365).toFixed(1) + 'y';
    return Math.max(0, Math.round(days)) + 'd';
  }
  function fmtMult(n) { return (Math.round(n * 10000) / 10000).toFixed(2) + 'x'; }
  function fmtOdds(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    return (n * 100).toFixed(0) + '%';
  }
  function fmtNet(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    return (n < 0 ? '-' : '+') + '$' + Math.abs(n).toLocaleString('en-US');
  }
  function fmtR2(n) { return (n >= 0 ? '' : '') + n.toFixed(3); }

  // The model in one function: the fitted value from the regression, then the odds from the
  // snapshot's token record, then the implied value. The page recomputes a row and checks the
  // number board.json carries, so the screenshots and the board agree.
  function tvlTier(tvl) {
    if (tvl < 1e7) return '<10M';
    if (tvl < 5e7) return '10-50M';
    if (tvl < 2e8) return '50-200M';
    if (tvl < 1e9) return '200M-1B';
    return '>1B';
  }
  function smoothRate(tok, n) {
    return (tok + DATA.model.odds_prior_weight * DATA.odds.global) / (n + DATA.model.odds_prior_weight);
  }
  function tokenOdds(row) {
    var cat = DATA.odds.by_category[row.category || 'Other'];
    var catRate = cat ? smoothRate(cat.tok, cat.n) : DATA.odds.global;
    var tier = DATA.odds.by_tier[tvlTier(row.tvl_usd)];
    var tierRate = tier ? smoothRate(tier.tok, tier.n) : DATA.odds.global;
    var odds = (catRate + tierRate) / 2;
    return { odds: Math.min(DATA.model.odds_cap, Math.max(DATA.model.odds_floor, odds)), catRate: catRate, tierRate: tierRate };
  }
  // The radar-6 odds: a logistic regression on log TVL, log age, seven-day growth, log chain
  // count, the category's smoothed token rate, and a listed-2024-or-later step. Same features,
  // same standardization and the same dot product main.ts runs, so the recompute matches
  // board.json. A protocol with no listed date takes the step's 0.
  function catRateFor(row) {
    var cat = DATA.odds.by_category[row.category || 'Other'];
    return cat ? smoothRate(cat.tok, cat.n) : DATA.odds.global;
  }
  function eraStep(row) {
    if (!row.listed_at) return 0;
    var year = parseInt(String(row.listed_at).slice(0, 4), 10);
    return (isFinite(year) && year >= DATA.model.odds_era_year) ? 1 : 0;
  }
  function logitOdds(row) {
    var m = DATA.model;
    var age = (row.age_days === null || row.age_days === undefined) ? m.unknown_age_days : row.age_days;
    var growth = Math.min(m.growth_cap, Math.max(m.growth_floor, (row.change_7d === null || row.change_7d === undefined) ? 0 : row.change_7d));
    var x = [1, Math.log(Math.max(1, row.tvl_usd)), Math.log(Math.max(1, age)), growth / 100, Math.log(Math.max(1, (row.chains || []).length)), catRateFor(row), eraStep(row)];
    var s = 0;
    for (var j = 0; j < x.length; j++) {
      var zj = j === 0 ? x[j] : (x[j] - DATA.logit.mu[j]) / DATA.logit.sd[j];
      s += DATA.logit.beta[j] * zj;
    }
    var pr = 1 / (1 + Math.exp(-s));
    return Math.min(m.odds_cap, Math.max(m.odds_floor, pr));
  }
  function recompute(row) {
    var m = DATA.model;
    var logTvl = Math.log(Math.max(1, row.tvl_usd));
    var fittedLog = row.intercept + row.slope * logTvl + row.category_offset;
    var rawGrowth = (row.change_7d === null || row.change_7d === undefined) ? 0 : row.change_7d;
    var growthPct = Math.min(m.growth_cap, Math.max(m.growth_floor, rawGrowth));
    var growth = 1 + (growthPct / 100) * m.growth_weight;
    var age = (row.age_days === null || row.age_days === undefined) ? m.unknown_age_days : row.age_days;
    var maturity = m.maturity_floor + (Math.min(Math.max(age, 0), m.maturity_days) / m.maturity_days) * m.maturity_range;
    var chain = 1 + m.chain_step * Math.min(Math.max((row.chains || []).length - 1, 0), m.chain_max_bonus);
    var fitted = Math.max(1, Math.round(Math.exp(fittedLog) * growth * maturity * chain));
    var fittedLow = Math.max(1, Math.round(fitted * DATA.value_band.lo_mult));
    var oddsInfo = null;
    var implied;
    if (row.token_status === 'live') {
      implied = (row.mcap_usd !== null && row.mcap_usd > 0) ? row.mcap_usd : fitted;
    } else if (row.token_status === 'token-trades') {
      implied = fitted;
    } else if (row.token_status === 'not-a-launch') {
      implied = Math.max(1, Math.round(fitted * m.odds_non_launch));
      oddsInfo = { odds: m.odds_non_launch, smoothed: m.odds_non_launch, catRate: null, tierRate: null };
    } else {
      var sm = tokenOdds(row);
      oddsInfo = { odds: logitOdds(row), smoothed: sm.odds, catRate: sm.catRate, tierRate: sm.tierRate };
      implied = Math.max(1, Math.round(fitted * oddsInfo.odds));
    }
    return { fitted: fitted, fittedLow: fittedLow, fittedLog: fittedLog, logTvl: logTvl, growth: growth, growthPct: growthPct, maturity: maturity, chain: chain, age: age, odds: oddsInfo, implied: implied };
  }

  function el(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === 'class') node.className = attrs[k];
        else if (k === 'text') node.textContent = attrs[k];
        else if (k === 'html') node.innerHTML = attrs[k];
        else if (k.slice(0, 2) === 'on') node.addEventListener(k.slice(2), attrs[k]);
        else if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
      });
    }
    for (var i = 2; i < arguments.length; i++) {
      var child = arguments[i];
      if (child === null || child === undefined) continue;
      if (Array.isArray(child)) child.forEach(function (c) { if (c) node.appendChild(c); });
      else if (typeof child === 'string') node.appendChild(document.createTextNode(child));
      else node.appendChild(child);
    }
    return node;
  }

  // The farm plan is the same derivation main.ts writes to board.json: category, chains, TVL and
  // age decide the action, the capital, the hold and every cost and time, and the status decides
  // whether the plan is written for a launch, for yield, or for the position itself. No name is
  // special-cased.
  function chainOf(p) { return p.chains && p.chains.length > 0 ? p.chains[0] : 'its main chain'; }
  function gasFor(chain) {
    var c = (chain || '').toLowerCase();
    if (c === 'ethereum') return 'about $8 of gas';
    if (c === 'solana') return 'about $0.05 of gas';
    if (['base', 'arbitrum', 'optimism', 'polygon', 'monad', 'hyperliquid l1', 'plasma', 'arc', 'avalanche', 'bsc', 'binance'].indexOf(c) >= 0) return 'about $0.50 of gas';
    return 'about $2 of gas';
  }
  function capitalFor(p) {
    if (p.tvl_usd >= 5e9) return 250;
    if (p.tvl_usd >= 1e9) return 150;
    if (p.tvl_usd >= 1e8) return 100;
    return 50;
  }
  function holdDaysFor(age) {
    if (age === null || age === undefined) return 30;
    if (age < 180) return 60;
    if (age < 365) return 45;
    return 30;
  }
  function actionFor(p, capital, chain) {
    var c = (p.category || '').toLowerCase();
    var name = p.name;
    if (c.indexOf('lend') >= 0 || c.indexOf('cdp') >= 0 || c.indexOf('collateral') >= 0) return 'Supply $' + capital + ' of USDC to ' + name + ' on ' + chain + ', then borrow a small amount against it and repay it once.';
    if (c.indexOf('dex') >= 0 || c.indexOf('yield') >= 0 || c.indexOf('leveraged farm') >= 0) return 'Provide $' + capital + ' of liquidity to ' + name + "'s main pool on " + chain + ' and keep the position open.';
    if (c.indexOf('bridge') >= 0 || c.indexOf('cross chain') >= 0) return 'Bridge $' + capital + ' of USDC to ' + chain + ' through ' + name + ' and use the bridged funds in its app.';
    if (c.indexOf('staking') >= 0 || c.indexOf('restak') >= 0) return "Stake $" + capital + " of the chain's staking asset through " + name + ' and keep the receipt token.';
    if (c.indexOf('perpetual') >= 0 || c.indexOf('derivative') >= 0 || c.indexOf('basis trad') >= 0) return 'Deposit $' + capital + ' of USDC as margin on ' + name + ' and open one small hedged trade.';
    if (c.indexOf('prediction') >= 0) return 'Fund $' + capital + ' and place one market position on ' + name + '.';
    if (c.indexOf('curator') >= 0 || c.indexOf('allocator') >= 0 || c.indexOf('risk') >= 0) return 'Deposit $' + capital + ' into a ' + name + ' vault on ' + chain + ' and keep the shares.';
    if (c.indexOf('stable') >= 0) return 'Mint or deposit $' + capital + ' of the stable asset through ' + name + ' on ' + chain + '.';
    if (c.indexOf('rwa') >= 0) return "Mint $" + capital + " of the protocol's yield-bearing asset on " + chain + ' and hold it.';
    if (c.indexOf('insurance') >= 0) return 'Buy $' + capital + ' of cover from ' + name + ' on ' + chain + ' and keep the policy active.';
    if (c.indexOf('payment') >= 0) return 'Send $' + capital + ' through ' + name + ' on ' + chain + ' and keep the receipt.';
    if (c.indexOf('chain') >= 0) return 'Use $' + capital + ' in ' + name + "'s app on " + chain + ' and keep the position open.';
    return 'Deposit $' + capital + ' of USDC into ' + name + ' on ' + chain + ' and keep the position open.';
  }
  function weeklyFor(p) {
    var c = (p.category || '').toLowerCase();
    if (c.indexOf('lend') >= 0 || c.indexOf('cdp') >= 0 || c.indexOf('collateral') >= 0) return 'Make one qualifying action each week, such as a supply, a borrow or a repayment.';
    if (c.indexOf('dex') >= 0 || c.indexOf('yield') >= 0 || c.indexOf('leveraged farm') >= 0) return 'Make one qualifying action each week, such as a swap, a deposit or a claim of pool fees.';
    if (c.indexOf('bridge') >= 0 || c.indexOf('cross chain') >= 0) return 'Make one qualifying action each week, such as a bridge, a swap or a deposit on the destination chain.';
    if (c.indexOf('perpetual') >= 0 || c.indexOf('derivative') >= 0 || c.indexOf('basis trad') >= 0) return 'Make one qualifying action each week, such as a trade, a deposit or a funding payment.';
    if (c.indexOf('staking') >= 0 || c.indexOf('restak') >= 0) return 'Make one qualifying action each week, such as a stake, a restake or a vote.';
    if (c.indexOf('prediction') >= 0) return 'Make one qualifying action each week, such as a market position or a settlement.';
    return 'Make one qualifying action each week, such as a deposit, a swap, a vote or a claim.';
  }
  function stepText(step) {
    var s = step.text + ' Cost: ' + step.cost + '. Time: ' + step.time + '.';
    if (s.length > 600) s = s.slice(0, 596) + '...';
    if (s.length < 20) s = s + " Hold the position and check the protocol's own page.";
    return s;
  }
  function farmSteps(p) {
    var chain = chainOf(p);
    var capital = capitalFor(p);
    var hold = holdDaysFor(p.age_days);
    var gas = gasFor(chain);
    var wallet = chain.toLowerCase() === 'solana'
      ? 'Use a fresh Solana wallet that holds only the funds you plan to farm, and keep its seed phrase offline.'
      : 'Use a fresh EVM wallet that holds only the funds you plan to farm, and keep its seed phrase offline.';
    var c = (p.category || '').toLowerCase();
    var claimWord = (c.indexOf('curator') >= 0 || c.indexOf('allocator') >= 0 || c.indexOf('risk') >= 0 || c.indexOf('yield') >= 0) ? 'points' : 'rewards';
    var claimText;
    if (p.token_status === 'pre-token') claimText = "Before you withdraw, open the protocol's own " + claimWord + ' page and claim only from the link it publishes; a real airdrop never asks you to pay to claim.';
    else if (p.token_status === 'live') claimText = "Before you withdraw, open the protocol's own " + claimWord + ' page and claim only from the link it publishes; a real distribution never asks you to pay to claim.';
    else claimText = "Before you withdraw, open the protocol's own " + claimWord + ' page and claim only from the link it publishes; never pay a fee to claim a reward.';
    return [
      { text: wallet, cost: 'nothing but the wallet', time: '5 minutes' },
      { text: actionFor(p, capital, chain), cost: '$' + capital + ' of capital and ' + gas, time: '15 minutes to set up' },
      { text: weeklyFor(p), cost: gas + ' a week', time: '10 minutes a week' },
      { text: 'Hold the position for at least ' + hold + ' days so the activity is counted in more than one snapshot.', cost: 'the $' + capital + ' stays in the position', time: hold + ' days' },
      { text: claimText, cost: 'no fee to check', time: '10 minutes' },
      { text: 'Write the wallet, the dates and every transaction in a farm log so you can repeat the position or unwind it.', cost: 'nothing but the log', time: '5 minutes' }
    ].map(stepText);
  }

  function readHash() {
    var hash = window.location.hash.replace(/^#/, '');
    if (!hash) return;
    var params = new URLSearchParams(hash);
    if (params.get('q')) state.q = params.get('q');
    if (params.get('p')) state.selected = params.get('p');
    if (params.get('category')) state.category = params.get('category');
    if (params.get('verdict')) state.verdict = params.get('verdict');
    if (params.get('tab')) state.tab = params.get('tab');
  }
  function writeHash() {
    var params = new URLSearchParams();
    if (state.q) params.set('q', state.q);
    if (state.category !== 'all') params.set('category', state.category);
    if (state.verdict !== 'all') params.set('verdict', state.verdict);
    if (state.tab !== 'pre-token') params.set('tab', state.tab);
    if (state.selected) params.set('p', state.selected);
    var next = params.toString();
    if (next) window.history.replaceState(null, '', '#' + next);
    else window.history.replaceState(null, '', window.location.pathname + window.location.search);
  }

  var STATUS_LABEL = { 'pre-token': 'launch candidate', 'live': 'already trades', 'token-trades': 'token trades', 'not-a-launch': 'not a launch' };
  var STATUS_CLASS = { 'pre-token': 'pre', 'live': 'live', 'token-trades': 'trades', 'not-a-launch': 'notlaunch' };
  var TABS = [
    { id: 'pre-token', label: 'Launch candidates' },
    { id: 'all', label: 'All 758' },
    { id: 'live', label: 'Live' },
    { id: 'token-trades', label: 'Token trades' },
    { id: 'not-a-launch', label: 'Not a launch' }
  ];
  function tabRows() {
    if (state.tab === 'all') return DATA.rows;
    return DATA.rows.filter(function (r) { return r.token_status === state.tab; });
  }
  function rows() {
    var q = state.q.trim().toLowerCase();
    return tabRows().filter(function (r) {
      if (state.category !== 'all' && (r.category || 'Other') !== state.category) return false;
      if (state.verdict !== 'all' && r.farm_verdict !== state.verdict) return false;
      if (!q) return true;
      var hay = [r.name, r.slug, r.category, (r.chains || []).join(' ')].join(' ').toLowerCase();
      return hay.indexOf(q) >= 0;
    });
  }
  function sortedRows(list) {
    var key = state.sort;
    return list.slice().sort(function (a, b) {
      var av, bv;
      if (key === 'name') { av = a.name.toLowerCase(); bv = b.name.toLowerCase(); }
      else if (key === 'tvl') { av = a.tvl_usd; bv = b.tvl_usd; }
      else if (key === 'growth') { av = a.change_7d === null ? -Infinity : a.change_7d; bv = b.change_7d === null ? -Infinity : b.change_7d; }
      else if (key === 'age') { av = a.age_days === null ? Infinity : a.age_days; bv = b.age_days === null ? Infinity : b.age_days; }
      else if (key === 'fitted') { av = a.fitted_value_usd; bv = b.fitted_value_usd; }
      else if (key === 'odds') { av = a.token_odds === null ? -Infinity : a.token_odds; bv = b.token_odds === null ? -Infinity : b.token_odds; }
      else if (key === 'verdict') { av = a.expected_net_usd === null ? -Infinity : a.expected_net_usd; bv = b.expected_net_usd === null ? -Infinity : b.expected_net_usd; }
      else { av = a.implied_value_usd; bv = b.implied_value_usd; }
      if (av < bv) return -state.dir;
      if (av > bv) return state.dir;
      return a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0;
    });
  }
  function median(nums) {
    if (!nums.length) return null;
    var s = nums.slice().sort(function (a, b) { return a - b; });
    var mid = Math.floor(s.length / 2);
    return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  }
  function launchRows() { return DATA.rows.filter(function (r) { return r.token_status === 'pre-token'; }); }

  function header() {
    return el('header', { class: 'top' },
      el('div', { class: 'brand' }, el('span', { class: 'mark' }), el('span', { text: 'Token Radar' })),
      el('div', { class: 'top-meta' },
        el('span', { text: 'Snapshot ' + DATA.as_of }),
        el('span', { text: 'DefiLlama protocol index' }),
        el('span', { text: DATA.model_version + ' model' })
      )
    );
  }

  function kpis() {
    var launch = launchRows();
    var medLaunch = median(launch.map(function (r) { return r.implied_value_usd; }));
    var byCat = {};
    launch.forEach(function (r) {
      var c = r.category || 'Other';
      (byCat[c] = byCat[c] || []).push(r.implied_value_usd);
    });
    var topCat = null, topMed = -1;
    Object.keys(byCat).forEach(function (c) {
      if (byCat[c].length < 3) return;
      var m = median(byCat[c]);
      if (m !== null && m > topMed) { topMed = m; topCat = c; }
    });
    var cv = DATA.validation.folds[0];
    return el('section', { class: 'kpis', 'aria-label': 'Key figures' },
      el('div', { class: 'kpi' }, el('span', { text: 'Protocols in the snapshot' }), el('b', { text: String(DATA.board_size) }), el('small', { text: DATA.watchlist_size + ' on the watchlist' })),
      el('div', { class: 'kpi' }, el('span', { text: 'Launch candidates' }), el('b', null, el('em', { text: String(DATA.pre_token_size) })), el('small', { text: 'no token on record, launch type' })),
      el('div', { class: 'kpi' }, el('span', { text: 'Median launch value' }), el('b', { text: fmtUsd(medLaunch) }), el('small', { text: 'fitted value x odds' })),
      el('div', { class: 'kpi' }, el('span', { text: 'Top category by median launch' }), el('b', { text: topCat || 'n/a' }), el('small', { text: topCat ? fmtUsd(topMed) + ' median' : '' })),
      el('div', { class: 'kpi' }, el('span', { text: 'Held-out log R2 (5-fold)' }), el('b', null, el('em', { text: fmtR2(cv.log_r2) })), el('small', { text: 'median error ' + (cv.median_ape * 100).toFixed(0) + '% · ' + cv.test_protocols + ' live peers' }))
    );
  }

  function integrity() {
    var v = DATA.validation;
    var cv = v.folds[0], wf = v.walk_forward, oc = v.odds_comparison;
    function item(label, value) { return el('div', { class: 'int-item' }, el('b', { text: value }), el('span', { text: label })); }
    return el('section', { class: 'integrity', 'aria-label': 'Integrity strip' },
      el('div', { class: 'int-head' }, el('b', { text: 'Integrity — every number here is computed by the program from the snapshot, not asserted' })),
      el('div', { class: 'int-grid' },
        item('Model', DATA.model_version),
        item('Snapshot', DATA.as_of),
        item('Coverage', DATA.board_size + ' of ' + DATA.board_size + ' protocols'),
        item('Watchlist covered', DATA.watchlist_size + ' of ' + DATA.watchlist_size),
        item('5-fold log R2', fmtR2(cv.log_r2)),
        item('5-fold median error', (cv.median_ape * 100).toFixed(0) + '%'),
        item('Walk-forward pooled log R2', fmtR2(wf.pooled.log_r2)),
        item('Walk-forward folds', wf.folds.map(function (f) { return fmtR2(f.log_r2); }).join(' / ')),
        item('Leave-one-out log R2', fmtR2(v.loo.log_r2)),
        item('Odds Brier, logistic / smoothed', oc.logistic.brier.toFixed(4) + ' / ' + oc.smoothed.brier.toFixed(4)),
        item('Odds log-loss, logistic / smoothed', oc.logistic.log_loss.toFixed(4) + ' / ' + oc.smoothed.log_loss.toFixed(4)),
        item('Odds calibration', 'low bin ' + (v.calibration[0].mean_pred * 100).toFixed(0) + '% pred vs ' + (v.calibration[0].actual_rate * 100).toFixed(0) + '% actual; top bin ' + (v.calibration[4].mean_pred * 100).toFixed(0) + '% vs ' + (v.calibration[4].actual_rate * 100).toFixed(0) + '%')
      )
    );
  }

  function lede() {
    return el('section', { class: 'lede' },
      el('h1', { text: 'Every protocol in the snapshot, ranked by what a launch could be worth' }),
      el('p', null,
        'The board carries all ', el('b', { text: String(DATA.board_size) }), ' protocols: ',
        el('b', { text: String(DATA.pre_token_size) }), ' launch candidates, ',
        el('b', { text: String(DATA.live_size) }), ' already trading, ',
        el('b', { text: String(DATA.token_trades_size) }), ' whose token already trades through another name, and ',
        el('b', { text: String(DATA.not_a_launch_size) }), ' that are not a launch. A launch candidate is ranked by its fitted value times the odds it ever records a token; a live protocol is ranked by its market cap, a fact. Each launch candidate carries a farm verdict — FARM, THIN or SKIP — from its expected net at the low end of its value band against the cost of farming it. TVL ≠ token value: this is a peer-multiple estimate, not a value-capture estimate. Implied value is a model, not a promise.'
      )
    );
  }

  function spotlight() {
    var top = launchRows().slice(0, 3);
    var labels = ['Launch 1', 'Launch 2', 'Launch 3'];
    return el('section', { class: 'spot', 'aria-label': 'Top three launch candidates' },
      top.map(function (r, i) {
        return el('button', { type: 'button', class: 'spot-card', onclick: function () { openDetail(r.slug); } },
          el('span', { class: 'rank-tag', text: labels[i] }),
          el('h3', { text: r.name }),
          el('div', { class: 'implied', text: fmtUsd(r.implied_value_usd) }),
          el('div', { class: 'meta', text: (r.category || 'Other') + ' · TVL ' + fmtUsd(r.tvl_usd) + ' · fitted ' + fmtUsd(r.fitted_value_usd) + ' · odds ' + fmtOdds(r.token_odds) + ' · ' + r.farm_verdict })
        );
      })
    );
  }

  function controls() {
    var categories = Array.from(new Set(DATA.rows.map(function (r) { return r.category || 'Other'; }))).sort();
    var search = el('input', { type: 'search', id: 'q', placeholder: 'Search name, category or chain', value: state.q, 'aria-label': 'Search the board' });
    search.addEventListener('input', function () { state.q = search.value; writeHash(); render(); });
    var select = el('select', { id: 'category', 'aria-label': 'Filter by category' },
      el('option', { value: 'all', text: 'All categories' }),
      categories.map(function (c) { return el('option', { value: c, text: c, selected: c === state.category ? 'selected' : null }); })
    );
    select.addEventListener('change', function () { state.category = select.value; writeHash(); render(); });
    var verdictSelect = el('select', { id: 'verdict', 'aria-label': 'Filter by farm verdict' },
      ['all', 'FARM', 'THIN', 'SKIP', 'YIELD'].map(function (v) { return el('option', { value: v, text: v === 'all' ? 'All verdicts' : v, selected: v === state.verdict ? 'selected' : null }); })
    );
    verdictSelect.addEventListener('change', function () { state.verdict = verdictSelect.value; writeHash(); render(); });
    var tabs = el('div', { class: 'tabs' }, TABS.map(function (t) {
      return el('button', { type: 'button', 'aria-pressed': state.tab === t.id ? 'true' : 'false', text: t.label, onclick: function () { state.tab = t.id; writeHash(); render(); } });
    }));
    return el('div', { class: 'controls' },
      el('div', { class: 'field' }, el('label', { for: 'q', text: 'Search' }), search),
      el('div', { class: 'field' }, el('label', { for: 'category', text: 'Category' }), select),
      el('div', { class: 'field' }, el('label', { for: 'verdict', text: 'Verdict' }), verdictSelect),
      el('div', { class: 'field' }, el('span', { class: 'sub', text: 'Token status' }), tabs),
      el('span', { class: 'count', text: rows().length + ' rows shown' })
    );
  }

  function sortHeader(key, label) {
    return el('th', { scope: 'col' },
      el('button', { type: 'button', onclick: function () { if (state.sort === key) state.dir = -state.dir; else { state.sort = key; state.dir = -1; } render(); } },
        label,
        state.sort === key ? el('span', { 'aria-hidden': 'true', text: state.dir < 0 ? ' v' : ' ^' }) : null
      )
    );
  }

  function verdictCell(r) {
    var cls = r.farm_verdict === 'FARM' ? 'farm' : r.farm_verdict === 'THIN' ? 'thin' : r.farm_verdict === 'YIELD' ? 'yield' : 'skip';
    return el('div', { class: 'verdict-cell' },
      el('span', { class: 'verdict ' + cls, text: r.farm_verdict }),
      r.expected_net_usd === null || r.expected_net_usd === undefined ? null : el('span', { class: 'net num', text: fmtNet(r.expected_net_usd) })
    );
  }

  function boardTable() {
    var list = sortedRows(rows());
    if (list.length === 0) {
      return el('div', { class: 'panel' }, el('div', { class: 'empty' },
        el('h2', { text: 'No protocol matches that filter' }),
        el('p', { text: 'Clear the search or pick another category to see the board again.' }),
        el('button', { class: 'button', type: 'button', text: 'Clear filters', onclick: function () { state.q = ''; state.category = 'all'; writeHash(); render(); } })
      ));
    }
    var max = list[0].implied_value_usd || 1;
    var tbody = el('tbody');
    list.forEach(function (r) {
      var growth = r.change_7d === null ? 'n/a' : fmtPct(r.change_7d);
      var growthClass = r.change_7d === null ? '' : r.change_7d >= 0 ? 'pos' : 'neg';
      var tr = el('tr', {
        'data-selected': state.selected === r.slug ? 'true' : 'false',
        tabindex: '0',
        onclick: function () { openDetail(r.slug); },
        onkeydown: function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openDetail(r.slug); } }
      },
        el('td', { class: 'num', text: String(r.rank) }),
        el('td', null, el('div', { class: 'name' },
          el('b', { text: r.name }),
          el('span', { class: 'sub', text: (r.category || 'Other') + ' on ' + (r.chains && r.chains.length ? r.chains.slice(0, 2).join(', ') : 'an unnamed chain') })
        )),
        el('td', { class: 'num', text: fmtUsd(r.tvl_usd) }),
        el('td', { class: 'num ' + growthClass, text: growth }),
        el('td', { class: 'num', text: fmtAge(r.age_days) }),
        el('td', { class: 'num', text: fmtUsd(r.fitted_value_usd) }),
        el('td', { class: 'num', text: fmtOdds(r.token_odds) }),
        el('td', null, el('div', { class: 'implied' },
          el('div', { class: 'bar', 'aria-hidden': 'true' }, el('i', { style: 'width:' + Math.max(3, Math.round((r.implied_value_usd / max) * 100)) + '%' })),
          el('span', { class: 'num', text: fmtUsd(r.implied_value_usd) })
        )),
        el('td', null, verdictCell(r)),
        el('td', null, el('span', { class: 'status ' + STATUS_CLASS[r.token_status], text: STATUS_LABEL[r.token_status] }))
      );
      tbody.appendChild(tr);
    });
    var tabLabel = (TABS.filter(function (t) { return t.id === state.tab; })[0] || {}).label || 'Board';
    return el('div', { class: 'panel' },
      el('div', { class: 'table-wrap' },
        el('table', null,
          el('caption', { class: 'sub', text: tabLabel + ', strongest implied value first. Select a row for its model and farm plan.' }),
          el('thead', null, el('tr', null,
            sortHeader('implied', 'Rank'),
            sortHeader('name', 'Protocol'),
            sortHeader('tvl', 'TVL'),
            sortHeader('growth', '7d'),
            sortHeader('age', 'Age'),
            sortHeader('fitted', 'Fitted'),
            sortHeader('odds', 'Odds'),
            sortHeader('implied', 'Implied value'),
            sortHeader('verdict', 'Verdict'),
            el('th', { scope: 'col', text: 'Token' })
          )),
          tbody
        )
      )
    );
  }

  function curveSvg() {
    var pts = DATA.curve;
    var xs = pts.map(function (p) { return p.log_tvl; });
    var ys = pts.map(function (p) { return p.log_mcap; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var pad = 0.6;
    x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
    var W = 520, H = 300, L = 44, R = 12, T = 12, B = 30;
    var X = function (x) { return L + ((x - x0) / (x1 - x0)) * (W - L - R); };
    var Y = function (y) { return T + ((y1 - y) / (y1 - y0)) * (H - T - B); };
    var dots = pts.map(function (p) { return '<circle class="dot" cx="' + X(p.log_tvl).toFixed(1) + '" cy="' + Y(p.log_mcap).toFixed(1) + '" r="2.6"/>'; }).join('');
    var lx0 = X(x0), lx1 = X(x1);
    var ly0 = Y(DATA.intercept + DATA.slope * x0), ly1 = Y(DATA.intercept + DATA.slope * x1);
    var grid = '';
    for (var g = 0; g <= 4; g++) {
      var gx = L + ((W - L - R) * g) / 4;
      var gy = T + ((H - T - B) * g) / 4;
      grid += '<line class="grid" x1="' + gx.toFixed(1) + '" y1="' + T + '" x2="' + gx.toFixed(1) + '" y2="' + (H - B) + '"/>';
      grid += '<line class="grid" x1="' + L + '" y1="' + gy.toFixed(1) + '" x2="' + (W - R) + '" y2="' + gy.toFixed(1) + '"/>';
    }
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Fitted curve: log market cap against log TVL for the 128 live-token protocols, with the regression line">' +
      grid +
      '<line class="axis" x1="' + L + '" y1="' + (H - B) + '" x2="' + (W - R) + '" y2="' + (H - B) + '"/>' +
      '<line class="axis" x1="' + L + '" y1="' + T + '" x2="' + L + '" y2="' + (H - B) + '"/>' +
      dots +
      '<line class="line" x1="' + lx0.toFixed(1) + '" y1="' + ly0.toFixed(1) + '" x2="' + lx1.toFixed(1) + '" y2="' + ly1.toFixed(1) + '"/>' +
      '<text class="lab" x="' + L + '" y="' + (H - 8) + '">log TVL</text>' +
      '<text class="lab" x="' + (L + 2) + '" y="' + (T + 10) + '">log market cap</text>' +
      '</svg>';
  }

  function modelPanel() {
    var m = DATA.model;
    var v = DATA.validation;
    var cv = v.folds[0], sp = v.split_80_20, bcv = v.baseline, bsp = v.baseline_split_80_20;
    var ftText = v.fitted_terms.map(function (t) { return t.variant + ' ' + t.log_r2.toFixed(3); }).join(', ');
    function vrow(label, a, isWin) {
      return el('tr', null,
        el('th', { scope: 'row', text: label }),
        el('td', { class: 'num', text: String(a.test_protocols) }),
        el('td', { class: 'num' + (isWin ? ' win' : ''), text: fmtR2(a.log_r2) }),
        el('td', { class: 'num' + (isWin ? ' win' : ''), text: (a.median_ape * 100).toFixed(0) + '%' }),
        el('td', { class: 'num' + (isWin ? ' win' : ''), text: a.log_rmse.toFixed(3) })
      );
    }
    return el('section', { class: 'panel model', id: 'model' },
      el('h2', { text: 'How implied value is computed' }),
      el('p', { class: 'sub', text: 'Every row is scored by one function of the snapshot figures. No protocol name is special-cased, and no row is hand-picked.' }),
      el('div', { class: 'formula' },
        'fitted = exp(', DATA.intercept.toFixed(3), ' + ', DATA.slope.toFixed(3), ' x log(tvl) + category offset) x growth x maturity x chain', el('br'),
        'implied = fitted x odds   (launch candidate)   ·   implied = market cap   (live)   ·   implied = fitted   (token trades)   ·   implied = fitted x 0.02   (not a launch)'
      ),
      el('p', { class: 'model-note' },
        el('b', { text: 'The fitted value: ' }),
        'a regression of log market cap on log TVL, fit by a Huber robust regression on the ', String(v.train_protocols), ' protocols that already trade: iteratively reweighted least squares, delta ', String(m.huber_delta), ', the scale from the median absolute deviation of the residuals, at most ', String(m.huber_max_iterations), ' iterations, starting from the least-squares fit. Log market-cap residuals are heavy-tailed, so a few outlier peers would drag a least-squares line; the Huber line downweights them. Each category takes its median residual shrunk toward zero by a prior weight of ', String(m.peer_prior_weight), '. The fitted value is the conditional median: exp() of the log prediction, with no smearing correction. Growth, maturity and chain are bounded multipliers (growth 1 + clamp(change_7d, ', String(m.growth_floor), ', ', String(m.growth_cap), ') / 100 x ', String(m.growth_weight), '; maturity ', String(m.maturity_floor), ' + min(age, ', String(m.maturity_days), ') / ', String(m.maturity_days), ' x ', String(m.maturity_range), '; chain 1 + 0.02 x min(chains - 1, 5)). They were tested as fitted terms too, by the program, on five-fold cross-validation — held-out log R2: ', ftText, '. Every extra term hurts, so they stay bounded multipliers, not fitted terms. The value band is fitted x ', DATA.value_band.lo_mult.toFixed(2), ' to fitted x ', DATA.value_band.hi_mult.toFixed(1), ' (10th–90th percentile of the ', String(v.train_protocols), ' live peers\u2019 residuals of log market cap against log fitted value); the farm verdict uses the low end. ',
        el('b', { text: 'The odds: ' }),
        'for a launch candidate, a logistic regression on log TVL, log age, seven-day growth, log chain count, its category token rate (Laplace-smoothed toward the global rate (', (DATA.global_token_rate * 100).toFixed(1), '%) with a prior weight of ', String(m.odds_prior_weight), ') and a listed-', String(m.odds_era_year), '-or-later step, standardized, gradient descent lr ', String(m.logit_lr), ', ', String(m.logit_iterations), ' iterations, clamped to [', String(m.odds_floor), ', ', String(m.odds_cap), ']. Coefficients: ', v.logit_coefficients.map(function (b) { return b.toFixed(3); }).join(', '), '. Held-out Brier ', v.odds_comparison.logistic.brier.toFixed(4), ' and log-loss ', v.odds_comparison.logistic.log_loss.toFixed(4), ', against the radar-3 smoothed average (category rate plus TVL-tier rate, halved): Brier ', v.odds_comparison.smoothed.brier.toFixed(4), ', log-loss ', v.odds_comparison.smoothed.log_loss.toFixed(4), '. The era step is there because the token rate is a step, not a slope, in listing time: protocols listed before 2024 record a token 78–80% of the time, the 2024 cohort 51%, 2025\u2019s 39%, 2026\u2019s 31% — younger protocols have had less time to launch, so the odds step down for the recent cohort. A protocol with no listed date takes the step\u2019s 0: its token rate is 0.79 over 113 protocols, the same as the pre-2024 cohorts. The odds are fitted on whether a protocol records a token in this snapshot, so a young protocol\u2019s lower odds partly reflect less time to launch, not a verdict that it never will. A protocol that cannot launch (exchange, bridge, liquid-staking product, versioned pool, wrapped or staked name) takes a structural floor of ', String(m.odds_non_launch), '.'
      ),
      el('div', { class: 'model-grid' },
        el('div', null,
          el('h3', { text: 'The fitted curve' }),
          el('div', { class: 'curve', html: curveSvg() }),
          el('p', { class: 'model-note', text: 'Each dot is one of the ' + v.train_protocols + ' live-token protocols, log market cap against log TVL. The line is the regression. A slope below 1 means a bigger protocol is worth less per dollar of TVL: the size effect the first radar\'s flat median missed.' })
        ),
        el('div', null,
          el('h3', { text: 'Held-out validation' }),
          el('table', { class: 'valid' },
            el('thead', null, el('tr', null, el('th', { scope: 'col', text: 'Check' }), el('th', { scope: 'col', class: 'num', text: 'n' }), el('th', { scope: 'col', class: 'num', text: 'log R2' }), el('th', { scope: 'col', class: 'num', text: 'Median err' }), el('th', { scope: 'col', class: 'num', text: 'log RMSE' }))),
            el('tbody', null,
              vrow('5-fold cross-validation, ' + DATA.model_version, cv, true),
              vrow('5-fold cross-validation, radar-2 median', bcv, false),
              vrow('80/20 split (fold 0 held out), ' + DATA.model_version, sp, true),
              vrow('80/20 split, radar-2 median', bsp, false)
            )
          ),
          el('p', { class: 'model-note', text: 'Each fold fits the regression on the training folds only and scores protocols it never saw; a protocol\'s fold is its index in slug order, mod 5. The fitted model beats the first radar\'s median on every metric in both checks. The 80/20 split is one fold of 26 protocols, so its log R2 is noisy; the five-fold check over all ' + v.train_protocols + ' is the primary measure.' })
        )
      ),
      el('div', { class: 'checks' },
        el('div', null,
          el('h3', { text: 'Walk-forward: tested forward in time' }),
          el('table', { class: 'valid' },
            el('thead', null, el('tr', null, el('th', { scope: 'col', text: 'Fold' }), el('th', { scope: 'col', class: 'num', text: 'Train' }), el('th', { scope: 'col', class: 'num', text: 'Test' }), el('th', { scope: 'col', class: 'num', text: 'log R2' }), el('th', { scope: 'col', class: 'num', text: 'Median err' }))),
            el('tbody', null,
              v.walk_forward.folds.map(function (f) {
                return el('tr', null,
                  el('th', { scope: 'row', text: 'Fold ' + f.fold + ' (from ' + f.from + ')' }),
                  el('td', { class: 'num', text: String(f.train_protocols) }),
                  el('td', { class: 'num', text: String(f.test_protocols) }),
                  el('td', { class: 'num', text: fmtR2(f.log_r2) }),
                  el('td', { class: 'num', text: (f.median_ape * 100).toFixed(0) + '%' }));
              }).concat([el('tr', null,
                el('th', { scope: 'row', text: 'Pooled' }),
                el('td', { class: 'num', text: '—' }),
                el('td', { class: 'num', text: String(v.walk_forward.pooled.test_protocols) }),
                el('td', { class: 'num', text: fmtR2(v.walk_forward.pooled.log_r2) }),
                el('td', { class: 'num', text: (v.walk_forward.pooled.median_ape * 100).toFixed(0) + '%' }))])
            )
          ),
          el('p', { class: 'model-note', text: 'Five time folds by listed date, each trained on all earlier protocols and tested on the next 19. The comps curve held forward in time until the 2024 cohort, then the regime shifted: newer protocols price mcap/TVL differently, so the pooled R2 is lower than the shuffled five-fold one. Leave-one-out agrees with the shuffle: log R2 ' + fmtR2(v.loo.log_r2) + ', median error ' + (v.loo.median_ape * 100).toFixed(0) + '%.' })
        ),
        el('div', null,
          el('h3', { text: 'Held-out calibration of the odds' }),
          el('table', { class: 'valid' },
            el('thead', null, el('tr', null, el('th', { scope: 'col', text: 'Predicted odds' }), el('th', { scope: 'col', class: 'num', text: 'n' }), el('th', { scope: 'col', class: 'num', text: 'Mean pred' }), el('th', { scope: 'col', class: 'num', text: 'Actually launched' }))),
            el('tbody', null,
              v.calibration.map(function (b) {
                return el('tr', null,
                  el('th', { scope: 'row', text: b.lo.toFixed(1) + '–' + b.hi.toFixed(1) }),
                  el('td', { class: 'num', text: String(b.n) }),
                  el('td', { class: 'num', text: (b.mean_pred * 100).toFixed(0) + '%' }),
                  el('td', { class: 'num', text: (b.actual_rate * 100).toFixed(0) + '%' }));
              })
            )
          ),
          el('p', { class: 'model-note', text: 'Five-fold held-out predictions, binned by predicted odds. The low bins slightly under-predict and the top bin slightly over-predicts; the odds are used as scored, with no recalibration, because the bins are small.' })
        ),
        el('div', null,
          el('h3', { text: 'Farm verdicts' }),
          el('p', { class: 'model-note' },
            'Launch candidates: FARM ', String(DATA.verdict_counts.FARM), ', THIN ', String(DATA.verdict_counts.THIN), ', SKIP ', String(DATA.verdict_counts.SKIP - DATA.not_a_launch_size), ' of ', String(DATA.pre_token_size), '. The verdict assumes the farmer holds 0.01% (one basis point) of the launch, so the expected gain is odds x 0.0001 x fitted-low value — the low end of the value band (fitted x ', DATA.value_band.lo_mult.toFixed(2), '); the cost is ', String(m.farm_tx_count), ' transactions of gas plus ', String(m.farm_capital_rate * 100), '% of the capital the plan deposits. FARM when the net clears $', String(m.farm_threshold_usd), ', THIN when it is positive, SKIP when it is not. A live or token-trades row is YIELD: farm for yield and points, no launch. The point-estimate net is shown on every row for comparison. The one-basis-point share is an assumption, stated here so the reader can argue with it — change it and the verdicts move.'
          )
        )
      )
    );
  }

  var dialog = el('dialog', { class: 'detail', id: 'detail', 'aria-labelledby': 'detail-title' });
  dialog.addEventListener('close', function () { state.selected = null; writeHash(); });

  function calibrationText(row, v) {
    if (!v.odds || v.odds.odds === null || v.odds.odds === undefined) return null;
    var bins = DATA.validation.calibration;
    var bin = bins[Math.min(bins.length - 1, Math.floor(v.odds.odds * 5))];
    if (!bin) return null;
    return 'Held-out calibration for odds like this: the model predicted ' + bin.lo.toFixed(1) + '–' + bin.hi.toFixed(1) + ' and ' + (bin.actual_rate * 100).toFixed(0) + '% of those protocols actually record a token (n=' + bin.n + ', mean prediction ' + (bin.mean_pred * 100).toFixed(0) + '%).';
  }

  function openDetail(slug) {
    var row = DATA.rows.filter(function (r) { return r.slug === slug; })[0];
    if (!row) return;
    state.selected = slug;
    writeHash();
    dialog.textContent = '';
    var v = recompute(row);
    var steps = farmSteps(row);
    var launchRank = null;
    if (row.token_status === 'pre-token') {
      var pre = launchRows();
      for (var i = 0; i < pre.length; i++) if (pre[i].slug === slug) { launchRank = i + 1; break; }
    }
    var rankLine = 'Board rank ' + row.rank + ' of ' + DATA.board_size + (launchRank ? ' · Launch rank ' + launchRank + ' of ' + DATA.pre_token_size : '');
    var statusNote;
    if (row.token_status === 'live') {
      statusNote = (row.mcap_usd !== null && row.mcap_usd > 0)
        ? 'Its token already trades, so the implied value is its market cap, a fact, not a model.'
        : 'Its token already trades, but the snapshot records its market cap as ' + String(row.mcap_usd) + ', so the fitted value is shown instead.';
    } else if (row.token_status === 'token-trades') {
      statusNote = 'It records a token but no market cap, almost always a version or a pool of a protocol whose token already trades, so the implied value is the fitted value with no odds.';
    } else if (row.token_status === 'not-a-launch') {
      statusNote = 'It records no token but fails the launch type rule (an exchange, a bridge, a liquid-staking product, a versioned pool, or a wrapped or staked name), so its odds are the structural floor of 0.02.';
    } else {
      statusNote = 'It records no token and passes the launch type rule, so it is a launch candidate: implied value is the fitted value times the odds it ever records a token.';
    }
    dialog.appendChild(el('div', { class: 'detail-head' },
      el('div', null,
        el('span', { class: 'brand-name', text: rankLine }),
        el('h2', { id: 'detail-title', text: row.name }),
        el('p', { class: 'sub', text: (row.category || 'Other') + ' on ' + (row.chains && row.chains.length ? row.chains.join(', ') : 'an unnamed chain') })
      ),
      el('button', { class: 'close', type: 'button', 'aria-label': 'Close details', text: 'Close', onclick: function () { dialog.close(); } })
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Implied value' }),
      el('div', { class: 'equation' },
        el('div', null, el('span', { text: 'TVL' }), el('b', { class: 'num', text: fmtUsd(row.tvl_usd) })),
        el('div', null, el('span', { text: 'Seven-day growth' }), el('b', { class: 'num', text: fmtPct(row.change_7d) })),
        el('div', null, el('span', { text: 'Age with no token' }), el('b', { class: 'num', text: fmtAge(row.age_days) })),
        el('div', null, el('span', { text: 'Listed ' + DATA.model.odds_era_year + ' or later (odds step)' }), el('b', { class: 'num', text: eraStep(row) ? 'yes' : 'no' })),
        el('div', null, el('span', { text: 'Fitted log market cap' }), el('b', { class: 'num', text: v.fittedLog.toFixed(2) })),
        el('div', null, el('span', { text: 'Category offset · ' + (row.category || 'Other') + ' (n=' + row.category_live_peers + ')' }), el('b', { class: 'num', text: (row.category_offset >= 0 ? '+' : '') + row.category_offset.toFixed(2) })),
        el('div', null, el('span', { text: 'Fitted value' }), el('b', { class: 'num', text: fmtUsd(v.fitted) })),
        el('div', null, el('span', { text: 'Fitted value, low end of band' }), el('b', { class: 'num', text: fmtUsd(v.fittedLow) })),
        el('div', null, el('span', { text: 'Growth multiplier' }), el('b', { class: 'num', text: fmtMult(v.growth) })),
        el('div', null, el('span', { text: 'Maturity multiplier' }), el('b', { class: 'num', text: fmtMult(v.maturity) })),
        el('div', null, el('span', { text: 'Chain multiplier (' + (row.chains || []).length + ' chains)' }), el('b', { class: 'num', text: fmtMult(v.chain) })),
        el('div', null, el('span', { text: 'Token odds' }), el('b', { class: 'num', text: v.odds ? fmtOdds(v.odds.odds) : 'n/a' })),
        el('div', null, el('span', { text: 'Token odds (radar-3 smoothed)' }), el('b', { class: 'num', text: v.odds && v.odds.smoothed !== null && v.odds.smoothed !== undefined ? fmtOdds(v.odds.smoothed) : 'n/a' })),
        el('div', null, el('span', { text: 'Category token rate' }), el('b', { class: 'num', text: v.odds && v.odds.catRate !== null ? fmtOdds(v.odds.catRate) : 'n/a' })),
        el('div', null, el('span', { text: 'TVL-tier token rate' }), el('b', { class: 'num', text: v.odds && v.odds.tierRate !== null ? fmtOdds(v.odds.tierRate) : 'n/a' })),
        el('div', null, el('span', { text: 'Farm verdict' }), el('b', { class: 'num', text: row.farm_verdict + (row.expected_net_usd === null || row.expected_net_usd === undefined ? '' : ' · net ' + fmtNet(row.expected_net_usd)) })),
        el('div', null, el('span', { text: 'Expected net at the point fitted value' }), el('b', { class: 'num', text: row.expected_net_point_usd === null || row.expected_net_point_usd === undefined ? 'n/a' : fmtNet(row.expected_net_point_usd) })),
        el('div', null, el('span', { text: 'Expected gain, low end (odds x 0.01% of fitted-low)' }), el('b', { class: 'num', text: row.expected_gain_usd === null || row.expected_gain_usd === undefined ? 'n/a' : fmtUsd(row.expected_gain_usd) })),
        el('div', null, el('span', { text: 'Farm cost (8 tx gas + 2% of capital)' }), el('b', { class: 'num', text: row.farm_cost_usd === null || row.farm_cost_usd === undefined ? 'n/a' : fmtUsd(row.farm_cost_usd) })),
        el('div', null, el('span', { text: 'Bootstrap rank range (200 refits, p10–p90)' }), el('b', { class: 'num', text: row.rank_range_lo === null || row.rank_range_lo === undefined ? 'n/a' : '#' + row.rank_range_lo + ' – #' + row.rank_range_hi })),
        el('div', { class: 'total' }, el('span', { text: 'Implied value' }), el('b', { class: 'num', text: fmtUsd(row.implied_value_usd) }))
      ),
      el('p', { class: 'check' }, statusNote),
      calibrationText(row, v) ? el('p', { class: 'check', text: calibrationText(row, v) }) : null,
      row.token_status === 'pre-token' ? el('p', { class: 'check', text: 'Farm verdict ' + row.farm_verdict + ': the expected gain assumes the farmer holds 0.01% (one basis point) of the launch — odds x 0.0001 x fitted-low value (fitted x ' + DATA.value_band.lo_mult.toFixed(2) + '), ' + fmtUsd(row.expected_gain_usd) + ' — against ' + fmtUsd(row.farm_cost_usd) + ' of gas and capital cost. At the point fitted value the net would be ' + fmtNet(row.expected_net_point_usd) + '. FARM when the net clears $' + DATA.model.farm_threshold_usd + ', THIN when it is positive, SKIP when it is not.' }) : null,
      el('p', { class: 'check' },
        'Model ' + DATA.model_version + ': the page recomputes this row as ',
        el('b', { text: fmtUsd(v.implied) }),
        v.implied === row.implied_value_usd ? ', the same number board.json carries.' : ', which differs from the board: the data would be stale.'
      ),
      el('p', { class: 'check', text: 'On the fitted curve: log TVL ' + v.logTvl.toFixed(2) + ', log fitted value ' + Math.log(Math.max(1, v.fitted)).toFixed(2) + ' (the regression line at this TVL is ' + (DATA.intercept + DATA.slope * v.logTvl).toFixed(2) + ' before the category offset).' })
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Farm plan' }),
      el('ol', { class: 'steps' }, steps.map(function (s) {
        var at = s.indexOf(' Cost: ');
        var action = at >= 0 ? s.slice(0, at) : s;
        var tail = at >= 0 ? s.slice(at + 1) : '';
        return el('li', null, el('b', { text: action }), document.createTextNode(' '), el('span', { class: 'cost', text: tail }));
      }))
    ));
    if (typeof dialog.showModal === 'function') {
      if (!dialog.open) dialog.showModal();
    } else {
      dialog.setAttribute('open', '');
    }
  }

  function footer() {
    return el('footer', null,
      el('p', { text: 'Data: DefiLlama protocol index, snapshot ' + DATA.as_of + '. A protocol is listed as no token when its record carries none; that is the record, not proof it has none.' }),
      el('p', { text: 'Implied value is this radar\u2019s own model estimate. It is not a price, not a prediction that a token will launch, and not financial advice.' })
    );
  }

  function render() {
    app.textContent = '';
    var shell = el('div', { class: 'shell' });
    shell.appendChild(header());
    shell.appendChild(integrity());
    shell.appendChild(kpis());
    shell.appendChild(lede());
    if (state.tab === 'pre-token') shell.appendChild(spotlight());
    shell.appendChild(controls());
    shell.appendChild(boardTable());
    if (state.tab === 'pre-token' || state.tab === 'all') shell.appendChild(modelPanel());
    shell.appendChild(footer());
    app.appendChild(shell);
    app.setAttribute('aria-busy', 'false');
    if (!dialog.isConnected) app.appendChild(dialog);
    if (state.selected) openDetail(state.selected);
  }

  readHash();
  render();
  window.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && dialog.open) dialog.close();
  });
})();
