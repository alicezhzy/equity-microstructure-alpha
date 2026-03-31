from __future__ import annotations

from typing import Iterable
import polars as pl


REQUIRED_COLUMNS = [
    "ts_event",
    "price",
    "size",
    "symbol",
]


def validate_required_columns(
    df: pl.DataFrame,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def sort_by_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(["symbol", "ts_event"])


def remove_duplicate_ticks(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove exact duplicate rows.
    """
    return df.unique(maintain_order=True)


def filter_invalid_prices(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove rows with non-positive or null prices.
    """
    return df.filter(
        pl.col("price").is_not_null() &
        (pl.col("price") > 0)
    )


def filter_invalid_sizes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove rows with non-positive or null sizes.
    """
    return df.filter(
        pl.col("size").is_not_null() &
        (pl.col("size") > 0)
    )


def filter_suspicious_price_jumps(
    df: pl.DataFrame,
    max_abs_return: float = 0.20,
) -> pl.DataFrame:
    """
    Remove rows where the event-to-event price jump is implausibly large.
    This is a conservative sanity filter for bad ticks.
    """
    df = sort_by_timestamp(df)

    df = df.with_columns(
        pl.col("price")
        .pct_change()
        .over("symbol")
        .alias("_event_return")
    )

    df = df.filter(
        pl.col("_event_return").is_null() |
        (pl.col("_event_return").abs() <= max_abs_return)
    ).drop("_event_return")

    return df


def filter_regular_trading_hours(
    df: pl.DataFrame,
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
) -> pl.DataFrame:
    """
    Keep only rows inside regular trading hours.
    Assumes ts_event is timezone-aware and already in the intended timezone.
    For now this is optional because your sample appears premarket.
    """
    return df.filter(
        pl.col("ts_event").dt.strftime("%H:%M:%S") >= start_time
    ).filter(
        pl.col("ts_event").dt.strftime("%H:%M:%S") <= end_time
    )


def summarize_cleaning(
    before_df: pl.DataFrame,
    after_df: pl.DataFrame,
) -> pl.DataFrame:
    before_rows = before_df.height
    after_rows = after_df.height
    removed_rows = before_rows - after_rows
    removed_pct = 100 * removed_rows / before_rows if before_rows > 0 else 0.0

    return pl.DataFrame(
        {
            "before_rows": [before_rows],
            "after_rows": [after_rows],
            "removed_rows": [removed_rows],
            "removed_pct": [removed_pct],
        }
    )


def clean_trade_data(
    df: pl.DataFrame,
    remove_dupes: bool = True,
    filter_bad_prices: bool = True,
    filter_bad_sizes: bool = True,
    filter_bad_jumps: bool = False,
    regular_trading_hours_only: bool = False,
) -> pl.DataFrame:
    """
    End-to-end cleaner for event-level trade data.
    """
    validate_required_columns(df)

    cleaned = df

    if remove_dupes:
        cleaned = remove_duplicate_ticks(cleaned)

    cleaned = sort_by_timestamp(cleaned)

    if filter_bad_prices:
        cleaned = filter_invalid_prices(cleaned)

    if filter_bad_sizes:
        cleaned = filter_invalid_sizes(cleaned)

    if filter_bad_jumps:
        cleaned = filter_suspicious_price_jumps(cleaned)

    if regular_trading_hours_only:
        cleaned = filter_regular_trading_hours(cleaned)

    return cleaned