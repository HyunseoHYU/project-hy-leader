# prediction/ — 트랙 2: 예측 모델 (MoE 국면 라우팅)

> 상위: [../CLAUDE.md](../CLAUDE.md) · 이 폴더 파일을 만지기 전에 읽는다.
> 연결 트랙: [../impact/CLAUDE.md](../impact/CLAUDE.md) · 데이터: [../data/CLAUDE.md](../data/CLAUDE.md)

## 역할
가격·거래량·(감성) 피처로 시장 국면(상승/횡보/하락)을 판별하고 국면별 expert로 수익률을 예측하는 Mixture-of-Experts 파이프라인.
impact 트랙에서 유의한 유동성 반응이 확인되면 그 지표를 이 트랙의 피처·타깃으로 연결한다.

## 구성
| 경로 | 내용 |
|---|---|
| `run_pipeline.py` | 실행 진입점. `--config`, `--csv` (CSV 없으면 합성 국면전환 시계열) |
| `configs/default.yaml` | 데이터·라벨·라우터·expert 설정 |
| `moe_regime/` | `data` · `features`(인과적) · `labels` · `router`(게이트) · `experts`(ridge) · `moe` · `metrics` · `pipeline` |
| `tests/test_pipeline.py`, `conftest.py` | 피처 인과성·라벨·라우터 테스트 |
| `outputs/` | 실행 결과 `metrics.json`, `routing_test.csv` (**git 제외**) |

## 규칙
- 피처는 **인과적**이어야 한다 (t 시점 피처에 t 이후 정보 금지). 이 성질은 테스트로 보호되므로 `features.py`를 바꾸면 테스트를 먼저 돌린다
- 현재 결과는 **합성 데이터 기준**이며 우위는 미입증이다. 실데이터 결과로 오해할 문구를 쓰지 않는다
- 비용(편도 10bp)을 반영한 성과만 보고한다
- 설정값은 YAML에 두고 코드에 숫자를 박지 않는다

## 실행
```bash
python prediction/run_pipeline.py            # 합성 데이터
python prediction/run_pipeline.py --csv data/xxx.csv   # date, close[, volume, sentiment]
python -m pytest prediction                  # 테스트 (CI에서도 실행)
```
