import polars as pl
from src.bar_construction import build_one_second_bars


def add_second_bucket(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("ts_event").dt.truncate("1s").alias("second")
    )


def build_order_flow_features(df: pl.DataFrame) -> pl.DataFrame:
    df = add_second_bucket(df)

    features = (
        df.group_by("second")
        .agg([
            pl.when(pl.col("side") == "B").then(pl.col("size")).otherwise(0).sum().alias("buy_volume"),
            pl.when(pl.col("side") == "A").then(pl.col("size")).otherwise(0).sum().alias("sell_volume"),
            pl.when(pl.col("side") == "N").then(pl.col("size")).otherwise(0).sum().alias("neutral_volume"),
        ])
        .with_columns([
            pl.col("buy_volume").cast(pl.Float64),
            pl.col("sell_volume").cast(pl.Float64),
            pl.col("neutral_volume").cast(pl.Float64),
        ])
        .with_columns([
            (
                (pl.col("buy_volume") - pl.col("sell_volume")) /
                (pl.col("buy_volume") + pl.col("sell_volume") + 1e-9)
            ).alias("order_flow_imbalance")
        ])
        .sort("second")
    )

    return features


def build_feature_table(df: pl.DataFrame) -> pl.DataFrame:
    bars = build_one_second_bars(df)
    flow = build_order_flow_features(df)

    features = bars.join(flow, on="second", how="left").sort("second")
    return features


def add_lagged_returns(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("return_1s").shift(1).alias("return_lag1"),
        pl.col("return_1s").shift(5).alias("return_lag5"),
    ])
    return df


def add_rolling_volatility(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("return_1s").rolling_std(window_size=10).alias("volatility_10s"),
        pl.col("return_1s").rolling_std(window_size=30).alias("volatility_30s"),
    ])
    return df


def add_trade_intensity(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("trade_count").rolling_mean(window_size=10).alias("trade_intensity_10s"),
    ])
    return df


def add_volume_zscore(df: pl.DataFrame) -> pl.DataFrame:

    df = df.with_columns([
        pl.col("volume")
        .rolling_mean(window_size=30)
        .alias("volume_mean_30"),

        pl.col("volume")
        .rolling_std(window_size=30)
        .alias("volume_std_30"),
    ])

    df = df.with_columns([
        (
            (pl.col("volume") - pl.col("volume_mean_30")) /
            (pl.col("volume_std_30") + 1e-9)
        ).alias("volume_zscore")
    ])

    return df