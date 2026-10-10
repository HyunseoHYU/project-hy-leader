"""
마크다운 표용 숫자 서식 (summary.md 생성 시 공통 사용).

기존 descriptive_stats.py(탐색 단계, archive/pre-restructure 브랜치로 보관)에 있던 함수를
impact 트랙의 분석 스크립트들이 계속 쓸 수 있도록 분리한 것.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def format_number(value, float_format: str = ".4g"):
    """
    표 안의 숫자를 읽기 쉽게 문자열로 바꾼다.
        정수 값         → 천 단위 쉼표   (5258880 → "5,258,880")
        1000 이상 실수  → 쉼표 + 정수    (13880.2 → "13,880")
        그 외           → 유효숫자 4자리 (0.0010937 → "0.001094")
    """
    is_number = isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))
    if not is_number:
        return value
    if pd.isna(value):
        return ""
    if float(value).is_integer() and abs(value) >= 1:
        return f"{int(value):,}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return format(value, float_format)


def to_markdown_table(table: pd.DataFrame, float_format: str = ".4g") -> str:
    """DataFrame → 마크다운 표 문자열 (숫자는 format_number로 서식화)."""
    formatted = table.map(lambda value: format_number(value, float_format))
    # disable_numparse: 이미 만든 문자열("13,880")을 tabulate가 다시 숫자로 바꾸지 않게
    return formatted.to_markdown(stralign="right", disable_numparse=True)
