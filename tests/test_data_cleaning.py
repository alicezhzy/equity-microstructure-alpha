import polars as pl

from src.data_cleaning import (
    remove_duplicate_ticks,
    filter_invalid_prices,
    filter_invalid_sizes,
    sort_by_timestamp,
)


def test_remove_duplicate_ticks():
    df = pl.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ts_event": [1, 1],
            "price": [100.0, 100.0],
            "size": [10, 10],
        }
    )
    out = remove_duplicate_ticks(df)
    assert out.height == 1


def test_filter_invalid_prices():
    df = pl.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ts_event": [1, 2],
            "price": [100.0, -1.0],
            "size": [10, 10],
        }
    )
    out = filter_invalid_prices(df)
    assert out.height == 1
    assert out["price"][0] == 100.0


def test_filter_invalid_sizes():
    df = pl.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ts_event": [1, 2],
            "price": [100.0, 101.0],
            "size": [10, 0],
        }
    )
    out = filter_invalid_sizes(df)
    assert out.height == 1
    assert out["size"][0] == 10


def test_sort_by_timestamp():
    df = pl.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ts_event": [2, 1],
            "price": [101.0, 100.0],
            "size": [10, 10],
        }
    )
    out = sort_by_timestamp(df)
    assert out["ts_event"].to_list() == [1, 2]