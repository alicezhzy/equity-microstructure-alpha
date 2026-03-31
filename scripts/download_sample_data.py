import databento as db
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

client = db.Historical()  # uses DATABENTO_API_KEY from environment

data = client.timeseries.get_range(
    dataset="XNAS.ITCH",
    schema="trades",
    symbols=["AAPL"],
    start="2022-09-20",
    end="2022-09-21",
)

data.to_parquet(RAW_DIR / "aapl_trades_2022-09-20.parquet")
print("Saved:", RAW_DIR / "aapl_trades_2022-09-20.parquet")