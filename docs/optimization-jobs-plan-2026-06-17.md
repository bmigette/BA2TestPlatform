# Optimization Jobs & Experts — Plan for Review (2026-06-17)

What we're optimizing in the BA2 backtester, per expert × strategy, what knobs the GA tunes,
and which jobs are queued vs deferred. Driver: `ba2-test optimize-batch` (one GA per job;
fitness = **Calmar** by default). Reproducible via `scripts/run_phase1_grid.sh`.

## Run defaults (all jobs unless noted)
- **Universe:** 30 large-cap NASDAQ names (NDQ30) — except the screener jobs (see Phase 2).
- **Window:** 2023-01-01 → 2026-01-01 (3 yr). **Fill clock:** 5min. **Analysis:** weekly (Mon 09:30).
- **GA:** population 40, generations 8, Calmar fitness, parallel 6. Initial capital $10k.
- Engine fixes that make results trustworthy (all landed): per-position TP/SL bracket (#39),
  DD forward-fill (sane drawdowns/Calmar), **OCO-leg preservation** (a stop move no longer
  drops the TP), 5min perf (~18× faster), Calmar cadence fix.

---

## Experts

| Expert | Universe | Notes |
|---|---|---|
| **FMPRating** | NDQ30 | Analyst price-target/ratings. Works well on large caps. Primary expert. |
| **FactorRanker** | NDQ30 | Bypass/rebalance (its own portfolio manager); runs ONCE (no S1/S2/S3 variants). |
| **FMPEarningsDrift** | **screener midcap** | Earnings-drift edge is a small/midcap phenomenon — *useless on NDQ30*. Deferred. |
| **FMPInsiderClusterBuy** | **screener midcap** | FMP insider data not available for large caps — ~no signals on NDQ30. Deferred. |

## Strategy variants (for ruleset experts)

| Kind | Idea | TP | Exits |
|---|---|---|---|
| **S1** | The expert's LIVE dev-account ruleset | entry +%, optimizable 5–25% | trail-TP-to-target (live-34) + trailing-SL ladder + rating/time closes |
| **S2** | Hand-built **bracket** | fixed entry +%, opt 5–25% | confidence + expected-profit entry gates; bearish / downgrade / break-even-lock / time exits |
| **S3** | **Momentum / trailing** | none (wide cap) — let winners run | light entry gate; staged trailing-stop tiers (raise stop as profit climbs) |
| **S4** ⭐ NEW | **Target-anchored TP** | TP anchored on the analyst **target price**; gene = offset-from-target, **−20…+10%** (negative = below target) | S1's trailing ruleset (trail-TP-to-target + trailing-SL ladder) |
| **FACTOR** | FactorRanker only | n/a | n/a (rebalances to target weights) |

**Why S4:** validated that the analyst consensus target *trails the uptrend* (AVGO/NVDA/AMD
targets rose with price, staying ~15–30% above spot). Anchoring the TP on that rising target
(vs a static entry +%) more than **doubled return** on the big-winner set (177% vs 78%,
Calmar 1.54 vs 1.36) and let NVDA ride to **+1065% and exit via take-profit**. Only works now
that the OCO-leg bug is fixed (a stop move used to silently drop the TP).

## What the GA optimizes per job

- **FMPRating expert params:** `profit_ratio`, `min_analysts`, `price_target_window_days`,
  and **`target_price_type`** (NEW categorical gene — low / consensus / median / high /
  low_consensus_avg — i.e. *which analyst reference price* to use, and for S4 to anchor the TP on).
- **Risk-manager sizing** (ruleset experts): `risk_per_trade_pct`, `atr_multiplier`,
  `min_stop_loss_pct`, `max_virtual_equity_per_instrument_percent`.
- **TP/SL:** initial TP % + SL % (S1/S2) — or the **offset-from-target** (S4, negative-capable).
- **Entry conditions:** each gate's threshold + an on/off toggle (the GA can drop a gate).
- **Exit rules:** each rule's `action_value` (incl. **negative** values, e.g. trail-to-target
  −X%) + an on/off toggle.
- **FactorRanker:** its factor weights / top_n / winsorize_pct (bypass — no TP/SL/conditions).

(New optimizer capabilities added for this: **categorical genes** for `target_price_type`, and
**negative-capable ranges** for the target offset — the GA already handled negative bounds.)

---

## Job grid

### Phase 1 — NDQ30 (the experts that work on large caps) — RUNNING / re-runnable
1. FMPRating × **S1**
2. FMPRating × **S2**
3. FMPRating × **S3**
4. FMPRating × **S4** ⭐ (target-anchored — add to the next run)
5. **FactorRanker** (FACTOR)

Command:
```
ba2-test optimize-batch --experts FMPRating,FactorRanker --strategies S1,S2,S3,S4 \
  --universe <NDQ30> --start 2023-01-01 --end 2026-01-01 \
  --fitness calmar_ratio --interval 5min --population 40 --generations 8
```

### Phase 2 — screener midcap (DEFERRED until after a perf pass) — task #46
6. FMPEarningsDrift × {S1,S2,S3(,S4)} on a **screener-derived small/midcap** universe
7. FMPInsiderClusterBuy × {S1,S2,S3(,S4)} on the same midcap universe
8. **The FMP screener itself in optimization** (screener:* gene namespace exists)

Run **last, with a pause before** (per direction). Needs per-expert universes (the driver
currently shares one `--universe`) + the screener-history cache (`fetch-screener`).

---

## Status / open items
- **Phase 1 re-run** needed on the OCO-fixed engine for trustworthy numbers (prior runs
  overstated returns — winners rode past their TP after a stop move). Add **S4** to it.
- **S4 + categorical target_price_type + negative offset:** implemented & unit-tested; ready
  to include in the next grid.
- **Deferred (task #46):** Phase 2 screener/midcap jobs + screener-in-opt, after a perf pass.
- Best clean Phase-1 numbers so far (FMPRating, OCO/DD-fixed): S1 Calmar 5.65, S2 4.89,
  S3 4.72, drawdowns ~10–17%.

---

# Expert coverage review (2026-06-18)

Audit of every expert configured in the **dev account** (`expertinstance`) vs the optimization
plan. Goal: each expert has mixed settings/strategies in use; the optimizer finds the best
config per expert. LLM experts are **out of scope** (per direction).

| Expert | Dev instances | Backtestable | In opt plan | Strategies | Universe | Status |
|---|---|---|---|---|---|---|
| **FMPRating** | 21 | yes | ✅ | S1/S2/S3/S4 | NDQ30 | **running** (S4 re-run) |
| **FactorRanker** | 10 | yes (bypass) | ✅ | FACTOR | NDQ30 | deferred — perf pass (#47) |
| **FinnHubRating** | 2 | yes (clean) | ✅ **added** | S1/S2/S3¹ | NDQ30 | ready to run |
| **FMPEarningsDrift** | 1 | yes | ✅ | S1/S2/S3 | screener midcap | deferred (#46) |
| **FMPInsiderClusterBuy** | 1 | yes | ✅ | S1/S2/S3 | screener midcap | deferred (#46) |
| **FMPSenateTraderWeight** | 1 | yes (clean) | ✅ **added** | S1/S2/S3 | **broad (TBD²)** | needs universe |
| PennyMomentumTrader | 1 | LLM | ❌ skip | — | — | **out of scope (LLM)** |
| TradingAgents | — | LLM | ❌ skip | — | — | **out of scope (LLM)** |

¹ FinnHubRating gives a rating bucket, **not** an analyst target price → S4 (target-anchored TP)
  degrades to entry-percent, so run S1/S2/S3. Genes: buy/overweight/hold/underweight thresholds.
² FMPSenateTraderWeight signal is **sparse per symbol** (disclosed congressional trades), so
  NDQ30 is too narrow — it needs a **broad universe** of names senators actually traded
  (assess + cache that universe before running). Genes: disclosure-recency, exec-window,
  price-delta, confidence multipliers, min_traders/min_trades.

**Newly added to `_EXPERT_OPT`** (this commit): FinnHubRating (4 genes), FMPSenateTraderWeight
(7 genes). With the RM sizing block + S1/S2/S3 TP/SL + condition/exit genes, the optimizer now
searches the full space for both.

**Net:** all 6 non-LLM dev experts are now in the plan. Run order:
1. **Now:** FMPRating S1/S2/S3/S4 (running) — large-cap, the validated path.
2. **Next (NDQ30, ready):** FinnHubRating S1/S2/S3.
3. **Deferred to perf pass:** FactorRanker (#47), and the screener/midcap experts —
   FMPEarningsDrift, FMPInsiderClusterBuy, **FMPSenateTraderWeight** — on a screener-derived
   universe (#46), plus the screener itself in-opt.

---

# Options backtest (defined; tested later)

The backtest already supports options end-to-end — defining it here for when we run it.

- **Trigger:** a run is an *options run* iff any exit/RM rule names an **option action**
  (`buy_call`, `buy_put`, `sell_covered_call`, `buy_protective_put`) — detected by
  `strategy_uses_options(cfg)`. The handler then builds a **HistoricalOptionsProvider** from an
  `options_cache.sqlite` (built via `ba2-test fetch-options`) and injects it into the account.
- **Data window:** options history floor is **2024-02-01** (`validate_options_window`) — options
  runs must start on/after that, so this is a **2024-2025** test, not the full 2023-2026 window.
- **Optimizable option genes (per option exit rule):** `option_delta` (strike delta) and
  `option_dte` (days-to-expiry window center) — already in `collect_param_space` / `decode_params`
  (tests: `test_option_optimize_genes`, `test_options_optimization_ga_e2e`).
- **Capital:** options-expert optimization should use a **$20k** balance (per project note), not $10k.
- **Strategy shape (proposed S-variant "SO"):** an FMPRating/FinnHub long entry whose exit ruleset
  uses option actions — e.g. **sell a covered call** against a held long (income) and/or **buy a
  protective put** (hedge), with delta + DTE optimized. Calmar fitness, 5min fill clock.
- **Status:** infrastructure done + unit-tested; **run after** the equity grids (needs the
  options cache fetched for the universe + the 2024-02 floor).
