# .github/ — CI

> 상위: [../CLAUDE.md](../CLAUDE.md)

`workflows/ci.yml` — main push·PR마다 실행 (Python 3.12):
1. `py_compile`: `impact/*.py prediction/*.py prediction/moe_regime/*.py collectors/*.py`
2. `python -m pytest prediction`
3. `data/raw` 1분봉 무결성 (소스별 읽기 + `time` 중복 없음)

## 규칙
- 폴더를 추가·이름 변경하면 1번의 `py_compile` 경로도 같이 고친다 (새 `.py` 폴더가 CI에서 빠지지 않게)
- CI에 호가창 데이터(`data/orderbook/`)는 없다 — 그것에 의존하는 검사를 넣지 않는다
