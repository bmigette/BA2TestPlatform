# TODO - Exploration Topics

## Quality Diversity & MapElites

### Overview
Quality Diversity (QD) algorithms aim to find a diverse collection of high-performing solutions rather than a single optimal solution. This approach is particularly valuable for neural network training where multiple effective architectures/weights may exist.

### MapElites
- [ ] Investigate MapElites algorithm for parallel network training
- [ ] MapElites divides the solution space into a grid of niches based on behavioral descriptors
- [ ] Each cell stores the best-performing solution for that behavior
- [ ] Enables exploration of diverse solutions simultaneously

### Key Areas to Explore

1. **MapElites with Neural Networks**
   - [ ] How to define behavioral descriptors for trading networks
   - [ ] Grid resolution and dimensionality trade-offs
   - [ ] Mutation operators for network weights

2. **DCRL (Differentiable QD with RL)**
   - [ ] Research DCRL enhancements that combine MapElites with Reinforcement Learning
   - [ ] Evaluate if DCRL achieves better results than vanilla MapElites
   - [ ] Implementation complexity vs performance gains

3. **Shinka Framework**
   - [ ] Check if Shinka already uses MapElites by default
   - [ ] Document Shinka's QD implementation details
   - [ ] Identify any configuration options for QD parameters
   - [ ] Compare Shinka's approach with other QD libraries

### References to Research
- [ ] Original MapElites paper: Mouret & Clune (2015)
- [ ] DCRL papers and implementations
- [ ] Shinka documentation and source code

### Implementation Notes
- Consider behavioral descriptors: risk metrics, drawdown patterns, trade frequency
- GPU parallelization for fitness evaluation
- Storage strategy for elite archive


Check price that is used for target predictions

## Backtest Strategy Engine
- [ ] **Multiple open trades**: Support `max_open_trades` setting to allow opening new positions while existing ones are active. Currently limited to 1 position at a time.
- [ ] **Trade open delay**: Configurable minimum bars/time between trades to avoid overtrading.
- [ ] **Per-bar evaluation for multi-trade**: When multiple trades enabled, condition evaluation on every execution bar is needed (currently optimized to skip non-prediction bars in single-trade mode).
- [ ] **Backtest optimization engine**: Backend endpoint that sweeps TP/SL/strategy params in a single request (load model once, predict once, sweep params) instead of one backtest per combination.

## Performance
- [ ] **SQLite → PostgreSQL**: Large blob columns (equity_curve, drawdown_curve) cause page fragmentation. PostgreSQL with TOAST would handle better.
- [ ] **Backtest curve storage**: Store equity/drawdown curves in files or compress them instead of raw JSON blobs in DB.
- [ ] **Uvicorn workers**: Test `--workers N` on Windows for API concurrency.
