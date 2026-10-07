# MoE regime router (bull / sideways / bear)

Mixture-of-Experts pipeline: a **router (gating network)** reads causal price + sentiment
features and outputs soft probabilities over three regimes; one **expert** per regime
predicts the next-day return; output = Σ gate_e · expert_e.

```
data -> features (causal) -> regime labels (forward return / vol) -> chronological split
     -> Router (softmax MLP, class-weighted, optional top-k) + 3 ridge experts -> evaluation
```

Run: `python run_pipeline.py [--config configs/default.yaml] [--csv your.csv]`
(CSV columns: `date, close[, volume, sentiment]`; without `--csv` a synthetic regime-switching series is used.)
Tests: `python -m pytest tests`. Outputs: `outputs/metrics.json`, `outputs/routing_test.csv`.

Notes
- Regime labels look forward `horizon` days and are used **only** as router supervision; train/test are split
  chronologically with `horizon` rows purged. Router inputs are strictly causal (tested).
- Metrics reported vs. a single ridge baseline and buy&hold. On synthetic data router accuracy is ~0.5 (chance 0.33)
  and MoE Sharpe is roughly on par with the single model, i.e. no proven edge yet; validate on real BTC/sentiment data.
- Deps: numpy, pandas, pyyaml (pytest for tests). Swap `router.py`/`experts.py` for torch/sklearn models if desired.
