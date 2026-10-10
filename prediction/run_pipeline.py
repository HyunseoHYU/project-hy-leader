"""
MoE 국면 라우팅 파이프라인 실행 진입점 (예측 트랙).

입력 : 설정 YAML (기본 prediction/configs/default.yaml), 선택적으로 CSV (date, close[, volume, sentiment])
       CSV를 주지 않으면 합성 국면전환 시계열로 실행
출력 : prediction/outputs/metrics.json, routing_test.csv (git 제외)
실행 : python prediction/run_pipeline.py [--config PATH] [--csv PATH]
"""
import argparse
import json
from pathlib import Path

import yaml

from moe_regime.pipeline import run

# 어느 위치에서 실행해도 기본 설정 파일을 찾도록 이 파일 기준 경로로 고정
PREDICTION_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PREDICTION_DIR / "configs" / "default.yaml"

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
parser.add_argument("--csv", default=None, help="CSV with date, close[, volume, sentiment]; omit for synthetic data")
args = parser.parse_args()

with open(args.config) as config_file:
    config = yaml.safe_load(config_file)
if args.csv:
    config["csv"] = args.csv
# 결과 폴더도 실행 위치와 무관하게 prediction/ 아래로 (상대경로일 때만)
if not Path(config["out_dir"]).is_absolute():
    config["out_dir"] = str(PREDICTION_DIR / config["out_dir"])

print(json.dumps(run(config), indent=2))
