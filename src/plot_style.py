"""
공통 차트 스타일 & 저장 도우미
============================

모든 분석 스크립트(descriptive_stats.py, eda_*.py, liquidity_*.py)가 같은 색·서식을
쓰도록 한곳에 모아 둔 모듈. import 하는 순간 matplotlib 기본 설정(rcParams)이 적용된다.

사용 예:
    from plot_style import BLUE, ORANGE, save_figure
    fig, ax = plt.subplots()
    ax.plot(x, y, color=BLUE)
    save_figure(fig, output_dir / "chart.png")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# 화면이 없는 서버(VPS)에서도 그림을 파일로 저장할 수 있도록 비대화형 백엔드 사용
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# 색상 팔레트 (dataviz 스킬 기본값)
#   - 계열(series)을 구분할 때는 BLUE → ORANGE → AQUA 순서로 고정해서 사용
#   - 글자·축·격자는 회색 계열(INK, MUTED, GRID)로 눈에 덜 띄게
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"  # 1순위 계열 색
ORANGE = "#eb6834"  # 2순위 계열 색 (비교 대상, 기준선 등)
AQUA = "#1baf7a"  # 3순위 계열 색

INK = "#0b0b0b"  # 제목·본문 글자
INK_SECONDARY = "#52514e"  # 눈금 글자
MUTED = "#898781"  # 축 테두리, 보조선
GRID = "#e1e0d9"  # 격자선
SURFACE = "#fcfcfb"  # 배경

# ---------------------------------------------------------------------------
# matplotlib 기본 설정
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        # 배경색
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        # 글자·축 색
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        # 격자는 얇고 옅게
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        # 위·오른쪽 테두리 제거 (데이터에 집중)
        "axes.spines.top": False,
        "axes.spines.right": False,
        # 크기·해상도
        "font.size": 10,
        "lines.linewidth": 1.5,
        "savefig.dpi": 150,
    }
)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """그림을 PNG로 저장하고 메모리에서 닫는다 (반복 생성 시 메모리 누수 방지)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  saved: {output_path}")


def save_table_as_image(table: pd.DataFrame, title: str, output_path: Path) -> None:
    """
    통계표(DataFrame)를 이미지로 저장한다. 발표 자료에 바로 붙이기 위한 용도.
    숫자는 소수점 4자리로 반올림해 표시한다.
    """
    # --- 1. 표 크기에 맞춰 그림 크기 결정 (열·행 수에 비례) ---
    fig_width = 1.6 * (len(table.columns) + 1.2)
    fig_height = 0.42 * len(table) + 1.4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # --- 2. 표 그리기 ---
    cell_text = table.round(4).astype(str).values
    drawn_table = ax.table(
        cellText=cell_text,
        rowLabels=table.index,
        colLabels=table.columns,
        cellLoc="right",
        loc="center",
    )
    drawn_table.auto_set_font_size(False)
    drawn_table.set_fontsize(9)
    drawn_table.scale(1, 1.5)

    # --- 3. 서식: 머리글 행은 파란 배경 + 흰 글자, 나머지는 옅은 테두리 ---
    for (row_index, _column_index), cell in drawn_table.get_celld().items():
        cell.set_edgecolor(GRID)
        is_header_row = row_index == 0
        if is_header_row:
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_facecolor(BLUE)

    ax.set_title(title, fontsize=13, color=INK, fontweight="bold", loc="left", pad=18)
    save_figure(fig, output_path)
