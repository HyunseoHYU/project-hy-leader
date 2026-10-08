"""
수집기 공통 도우미
================

모든 수집기(collectors/*.py)가 같이 쓰는 경로·환경변수·CSV 처리 함수.

- import 하는 순간 프로젝트 루트의 .env 파일을 읽어 환경변수로 등록한다 (API 키 등)
- DATA_DIR: 수집 결과를 저장하는 data/ 폴더의 절대경로
  → 어느 폴더에서 스크립트를 실행해도 같은 위치에 저장된다
"""

import csv
import os

from dotenv import load_dotenv

# 프로젝트 루트(HY_LEADER/) = 이 파일(collectors/common.py)의 두 단계 위
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# .env 파일의 KEY=VALUE 들을 환경변수로 등록 (이미 설정된 환경변수는 덮어쓰지 않음)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def require_env(name):
    """필수 환경변수를 읽는다. 없으면 어떤 키가 빠졌는지 알려 주며 중단."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name} (see .env.example)")
    return value


def append_rows_csv(path, columns, rows):
    """
    CSV 파일 끝에 행들을 덧붙인다.
    파일이 없거나 비어 있으면 머리글(컬럼 이름) 행을 먼저 쓴다.
    """
    is_existing_file = os.path.exists(path) and os.path.getsize(path) > 0
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not is_existing_file:
            writer.writerow(columns)
        writer.writerows(rows)


def load_seen_ids(path, id_column):
    """
    이미 저장된 CSV에서 id 컬럼 값을 모두 읽어 집합(set)으로 돌려준다.
    새로 수집한 항목 중 이미 저장된 것을 거르는 데 쓴다 (중복 저장 방지, 증분 수집).
    """
    if not os.path.exists(path):
        return set()

    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        has_id_column = id_column in (reader.fieldnames or [])
        if not has_id_column:
            return set()
        return {row[id_column] for row in reader}
