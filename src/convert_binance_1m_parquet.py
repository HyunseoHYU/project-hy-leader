"""
Binance BTCUSDT 1분봉 CSV → 연도별 Parquet 변환기
================================================

목적
    856MB 원본 CSV를 연도별 Parquet으로 나눠 저장한다.
    (GitHub 파일당 100MB 제한을 피하고, 읽기 속도도 CSV보다 훨씬 빠르다)

입력
    data/binance/btcusdt_1m_max.csv        collectors/collect_binance_1m.py 결과 (git 제외)

출력
    data/raw/binance_btcusdt_1m_{YYYY}.parquet
    - Bitstamp 파일과 같은 규칙: `time` 컬럼 = 캔들 시작 시각(UTC)
    - close_time(캔들 종료 시각)도 timestamp로 변환해 함께 보관
    - 같은 시각 캔들이 두 번 있으면 하나만 남김 (이어받기 수집 시 경계에서 생길 수 있음)

실행
    python src/convert_binance_1m_parquet.py

구현 메모
    약 480만 행이라 pandas보다 메모리를 적게 쓰는 pyarrow로 직접 처리한다.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# 경로 & 컬럼 형식
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV_PATH = PROJECT_ROOT / "data" / "binance" / "btcusdt_1m_max.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

FLOAT_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]
MILLISECOND_UTC = pa.timestamp("ms", tz="UTC")


def read_klines_csv() -> pa.Table:
    """CSV를 컬럼 형식을 지정해 읽는다 (ISO 시각 문자열 → UTC timestamp)."""
    column_types = {column: pa.float64() for column in FLOAT_COLUMNS}
    column_types["open_time"] = MILLISECOND_UTC
    # close_time은 ".999000"처럼 마이크로초까지 있어서 µs로 읽은 뒤 ms로 바꾼다
    column_types["close_time"] = pa.timestamp("us", tz="UTC")
    column_types["num_trades"] = pa.int64()

    table = pacsv.read_csv(INPUT_CSV_PATH, convert_options=pacsv.ConvertOptions(column_types=column_types))

    close_time_index = table.column_names.index("close_time")
    close_time_in_ms = table["close_time"].cast(MILLISECOND_UTC)
    table = table.set_column(close_time_index, "close_time", close_time_in_ms)
    return table


def drop_duplicate_candles(table: pa.Table) -> tuple[pa.Table, int]:
    """
    시간순 정렬 후, 바로 앞 행과 시각이 같은 행을 버린다.
    반환: (중복 제거된 표, 버린 행 수)
    """
    table = table.sort_by("time")

    # is_same_as_previous[i] = (i+1번째 행 시각 == i번째 행 시각)
    is_same_as_previous = pc.equal(table["time"].slice(1), table["time"].slice(0, table.num_rows - 1))

    # 첫 행은 항상 남기고, 나머지는 '앞 행과 다를 때만' 남긴다
    keep_mask = pa.concat_arrays([pa.array([True]), pc.invert(is_same_as_previous.combine_chunks())])
    dropped_count = table.num_rows - pc.sum(keep_mask).as_py()
    return table.filter(keep_mask), dropped_count


def write_yearly_parquet(table: pa.Table) -> None:
    """달력 연도별로 나눠 zstd 압축 Parquet으로 저장한다 (임시 파일 → 이름 변경으로 안전하게)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    year_of_candle = pc.year(table["time"])

    for year in sorted(set(year_of_candle.to_pylist())):
        candles_in_year = table.filter(pc.equal(year_of_candle, year))

        output_path = OUTPUT_DIR / f"binance_btcusdt_1m_{year}.parquet"
        temporary_path = output_path.with_suffix(".parquet.tmp")
        pq.write_table(candles_in_year, temporary_path, compression="zstd")
        temporary_path.replace(output_path)

        print(f"saved {output_path.relative_to(PROJECT_ROOT)}: {candles_in_year.num_rows:,} rows")


def main() -> None:
    # --- 1. CSV 읽기 ---
    table = read_klines_csv()
    print(f"read {table.num_rows:,} rows from {INPUT_CSV_PATH.relative_to(PROJECT_ROOT)}")

    # --- 2. open_time → time 으로 이름 통일 (Bitstamp 파일과 같은 규칙) ---
    renamed_columns = ["time" if name == "open_time" else name for name in table.column_names]
    table = table.rename_columns(renamed_columns)

    # --- 3. 중복 캔들 제거 ---
    table, dropped_count = drop_duplicate_candles(table)
    print(f"dropped {dropped_count} duplicate rows")

    # --- 4. 연도별 저장 ---
    write_yearly_parquet(table)


if __name__ == "__main__":
    main()
