# Equity Microstructure Alpha Research

This project explores whether microstructure signals derived from tick-level equity data can predict short-horizon price movements.

The goal is to build a complete quantitative research pipeline including data processing, feature engineering, machine learning modeling, and transaction-cost-aware backtesting.

## Research Objectives

- Investigate predictive power of microstructure features such as bid-ask spread, order imbalance, and trade intensity.
- Evaluate signal quality using statistical diagnostics such as information coefficient (IC) and decile analysis.
- Train machine learning models to forecast short-horizon returns.
- Perform walk-forward validation to simulate realistic research conditions.
- Backtest trading signals with transaction cost and execution lag assumptions.
- Build a scalable data pipeline capable of handling large tick-level datasets.

## Pipeline Overview

Raw Tick Data  
→ Data Cleaning  
→ One-Second Bar Construction  
→ Feature Engineering  
→ Target Construction  
→ Signal Diagnostics (IC / Decile Analysis)  
→ Machine Learning Models  
→ Walk-Forward Validation  
→ Transaction-Cost-Aware Backtesting

## Technologies

- Python
- NumPy / Pandas / Polars
- Scikit-learn
- LightGBM
- Parquet data storage
