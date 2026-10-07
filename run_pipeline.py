"""Usage: python run_pipeline.py [--config configs/default.yaml] [--csv data.csv]"""
import argparse, json, yaml
from moe_regime.pipeline import run

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--csv", default=None, help="CSV with date, close[, volume, sentiment]; omit for synthetic data")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
if a.csv:
    cfg["csv"] = a.csv
print(json.dumps(run(cfg), indent=2))
