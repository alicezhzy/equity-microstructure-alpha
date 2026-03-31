import polars as pl


def add_returns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add past 1-second return feature.
    """
    df = df.with_columns(
        pl.col("close").pct_change().alias("return_1s")
    )
    return df


def add_future_return(df: pl.DataFrame, horizon: int = 1) -> pl.DataFrame:
    """
    Add future return target for a given horizon in seconds.
    """
    df = df.with_columns(
        (
            pl.col("close").shift(-horizon) / pl.col("close") - 1
        ).alias(f"future_return_{horizon}s")
    )
    return df


def add_direction_label(df: pl.DataFrame, horizon: int = 1) -> pl.DataFrame:
    """
    Convert future return into binary direction label.
    1 = positive return, 0 = zero or negative return
    """
    df = df.with_columns(
        pl.when(pl.col(f"future_return_{horizon}s") > 0)
        .then(1)
        .otherwise(0)
        .alias(f"direction_{horizon}s")
    )
    return df