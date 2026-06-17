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
