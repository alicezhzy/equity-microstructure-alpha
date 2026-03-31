import polars as pl


def add_second_bucket(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("ts_event").dt.truncate("1s").alias("second")
    )


def build_one_second_bars(df: pl.DataFrame) -> pl.DataFrame:
    df = add_second_bucket(df)

    bars = (
        df.group_by("second")
        .agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
            pl.len().alias("trade_count"),
            (
                (pl.col("price") * pl.col("size")).sum()
                / pl.col("size").sum()
            ).alias("vwap"),
            pl.col("symbol").first().alias("symbol"),
        ])
        .sort("second")
    )

    return bars