"""
data.py - Data Loading & Preprocessing
========================================
Handles data loading, cleaning, and preprocessing for the backtesting framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings("ignore")


def load_data(
    filepath: str,
    timestamp_col: str = "Timestamp",
    ticker_col: Optional[str] = "Ticker",
    price_col: str = "Close",
    volume_col: Optional[str] = "Volume",
    parse_dates: bool = True,
    handle_duplicates: str = "last",  # 'first', 'last', or 'drop'
) -> pd.DataFrame:
    """
    Load and perform initial data cleaning.
    
    Parameters
    ----------
    filepath : path to CSV file
    timestamp_col : name of timestamp column
    ticker_col : name of ticker column (None if single asset)
    price_col : name of price column
    volume_col : name of volume column (optional)
    parse_dates : whether to parse timestamps
    handle_duplicates : how to handle duplicate timestamps
    
    Returns
    -------
    df : cleaned DataFrame with datetime index
    """
    print(f"[data] Loading {filepath} ...")
    
    # Load data
    if parse_dates:
        df = pd.read_csv(filepath, parse_dates=[timestamp_col])
    else:
        df = pd.read_csv(filepath)
    
    print(f"[data] Loaded {len(df):,} rows × {len(df.columns)} columns.")
    
    # Convert timestamp to datetime if not already
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Handle duplicate timestamps
    if handle_duplicates == "last":
        dup_count = df.duplicated(subset=[timestamp_col]).sum()
        if dup_count > 0:
            print(f"[data] WARNING: {dup_count:,} duplicated timestamps detected – keeping {handle_duplicates}.")
            df = df.drop_duplicates(subset=[timestamp_col], keep=handle_duplicates)
    elif handle_duplicates == "first":
        df = df.drop_duplicates(subset=[timestamp_col], keep="first")
    elif handle_duplicates == "drop":
        df = df.drop_duplicates(subset=[timestamp_col], keep=False)
    
    # Set timestamp as index
    df.set_index(timestamp_col, inplace=True)
    
    # Sort by index
    df.sort_index(inplace=True)
    
    # Handle missing values
    print("\nMissing Value Summary:")
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'missing_count': missing_counts,
        'missing_pct': missing_pcts
    })
    missing_df = missing_df[missing_df['missing_count'] > 0]
    
    if len(missing_df) > 0:
        print(missing_df)
        
        # Drop columns with > 50% missing
        high_missing = missing_df[missing_df['missing_pct'] > 50].index.tolist()
        if high_missing:
            print(f"[data] Dropping {len(high_missing)} features > 50% missing: {high_missing}")
            df = df.drop(columns=high_missing)
        
        # Forward fill remaining missing values
        df = df.ffill()
        
        # Back fill any remaining at the start
        df = df.bfill()
        
        print(f"[data] After imputation: {df.isnull().sum().sum()} NaNs remain.")
    else:
        print("[data] No missing values detected.")
    
    return df


def create_target_variable(
    df: pd.DataFrame,
    price_col: str = "Close",
    horizon: int = 1,
    target_type: str = "return",  # 'return', 'log_return', or 'directional'
    group_by_ticker: bool = True,
    ticker_col: Optional[str] = "Ticker",
) -> pd.DataFrame:
    """
    Create target variable 'y' for prediction.
    
    Parameters
    ----------
    df : input DataFrame
    price_col : column name for price
    horizon : forward-looking periods (1 = next period)
    target_type : type of target
        - 'return': simple return (p(t+h)/p(t) - 1)
        - 'log_return': log return
        - 'directional': binary sign (1 for up, 0 for down)
    group_by_ticker : whether to handle multiple tickers separately
    ticker_col : column name for ticker identifier
    
    Returns
    -------
    df : DataFrame with target column 'y' added
    """
    df = df.copy()
    
    if group_by_ticker and ticker_col and ticker_col in df.columns:
        # Handle multiple tickers separately
        def create_target_for_group(group):
            if target_type == "return":
                group['y'] = group[price_col].shift(-horizon) / group[price_col] - 1
            elif target_type == "log_return":
                group['y'] = np.log(group[price_col].shift(-horizon) / group[price_col])
            elif target_type == "directional":
                returns = group[price_col].shift(-horizon) / group[price_col] - 1
                group['y'] = (returns > 0).astype(int)
            else:
                raise ValueError(f"Unknown target_type: {target_type}")
            return group
        
        df = df.groupby(ticker_col, group_keys=False).apply(create_target_for_group)
    else:
        # Single asset
        if target_type == "return":
            df['y'] = df[price_col].shift(-horizon) / df[price_col] - 1
        elif target_type == "log_return":
            df['y'] = np.log(df[price_col].shift(-horizon) / df[price_col])
        elif target_type == "directional":
            returns = df[price_col].shift(-horizon) / df[price_col] - 1
            df['y'] = (returns > 0).astype(int)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
    
    # Drop rows where target is NaN (end of series)
    df = df.dropna(subset=['y'])
    
    print(f"[data] Created target 'y' (type: {target_type}, horizon: {horizon})")
    print(f"[data] Target range: [{df['y'].min():.6f}, {df['y'].max():.6f}]")
    
    return df


def create_features(
    df: pd.DataFrame,
    price_col: str = "Close",
    volume_col: Optional[str] = "Volume",
    windows: List[int] = [5, 10, 20, 50],
    group_by_ticker: bool = True,
    ticker_col: Optional[str] = "Ticker",
) -> pd.DataFrame:
    """
    Create feature columns from price and volume data.
    
    Parameters
    ----------
    df : input DataFrame with target already created
    price_col : column name for price
    volume_col : column name for volume (optional)
    windows : list of rolling window sizes
    group_by_ticker : whether to handle multiple tickers separately
    ticker_col : column name for ticker identifier
    
    Returns
    -------
    df : DataFrame with additional feature columns
    """
    df = df.copy()
    
    def create_features_for_group(group):
        # Price-based features
        # Returns
        for window in windows:
            group[f'return_{window}'] = group[price_col].pct_change(window)
        
        # Rolling statistics
        for window in windows:
            group[f'close_ma_{window}'] = group[price_col].rolling(window).mean()
            group[f'close_std_{window}'] = group[price_col].rolling(window).std()
            group[f'close_max_{window}'] = group[price_col].rolling(window).max()
            group[f'close_min_{window}'] = group[price_col].rolling(window).min()
            
            # Price position relative to rolling range
            range_high = group[f'close_max_{window}']
            range_low = group[f'close_min_{window}']
            group[f'price_position_{window}'] = (group[price_col] - range_low) / (range_high - range_low + 1e-12)
        
        # Momentum indicators
        for window in windows:
            group[f'momentum_{window}'] = group[price_col] / group[price_col].shift(window) - 1
        
        # Volume-based features (if available)
        if volume_col and volume_col in group.columns:
            for window in windows:
                group[f'volume_ma_{window}'] = group[volume_col].rolling(window).mean()
                group[f'volume_ratio_{window}'] = group[volume_col] / (group[f'volume_ma_{window}'] + 1e-12)
        
        # Price-volume correlation (if volume available)
        if volume_col and volume_col in group.columns:
            group['price_volume_corr'] = (
                group[price_col].pct_change()
                .rolling(20)
                .corr(group[volume_col].pct_change())
            )
        
        return group
    
    if group_by_ticker and ticker_col and ticker_col in df.columns:
        df = df.groupby(ticker_col, group_keys=False).apply(create_features_for_group)
    else:
        df = create_features_for_group(df)
    
    # Drop rows with NaN from rolling calculations
    max_window = max(windows) if windows else 50
    df = df.iloc[max_window:]
    
    print(f"[data] Created {len([c for c in df.columns if c not in ['y', price_col, volume_col, ticker_col]])} features")
    
    return df


def describe_data(df: pd.DataFrame, target_col: str = "y") -> Dict:
    """
    Describe dataset statistics.
    
    Parameters
    ----------
    df : DataFrame with target column
    target_col : name of target column
    
    Returns
    -------
    description : dictionary of statistics
    """
    # Ensure target column exists
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in data. Run create_target_variable() first.")
    
    description = {
        "rows": len(df),
        "columns": len(df.columns),
        "date_range_start": str(df.index[0]),
        "date_range_end": str(df.index[-1]),
        "target_mean": round(float(df[target_col].mean()), 8),
        "target_std": round(float(df[target_col].std()), 8),
        "target_min": round(float(df[target_col].min()), 8),
        "target_max": round(float(df[target_col].max()), 8),
        "target_skew": round(float(df[target_col].skew()), 4),
        "target_kurtosis": round(float(df[target_col].kurtosis()), 4),
        "target_positive_pct": round((df[target_col] > 0).mean() * 100, 2),
        "target_negative_pct": round((df[target_col] < 0).mean() * 100, 2),
    }
    
    # Add information about tickers if present
    if 'Ticker' in df.columns:
        description["unique_tickers"] = df['Ticker'].nunique()
        description["tickers"] = df['Ticker'].unique().tolist()[:10]  # First 10
    
    # Add feature information
    feature_cols = [c for c in df.columns if c != target_col]
    description["feature_count"] = len(feature_cols)
    description["feature_names"] = feature_cols[:20]  # First 20 features
    
    # Data quality metrics
    description["missing_count"] = int(df.isnull().sum().sum())
    description["missing_pct"] = round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 4)
    
    return description


def prepare_full_pipeline(
    filepath: str,
    horizon: int = 1,
    target_type: str = "return",
    windows: List[int] = [5, 10, 20, 50],
    price_col: str = "Close",
    volume_col: Optional[str] = "Volume",
    timestamp_col: str = "Timestamp",
    ticker_col: Optional[str] = "Ticker",
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Complete data preparation pipeline.
    
    Parameters
    ----------
    filepath : path to CSV file
    horizon : forward prediction horizon
    target_type : 'return', 'log_return', or 'directional'
    windows : rolling windows for feature creation
    price_col : price column name
    volume_col : volume column name (optional)
    timestamp_col : timestamp column name
    ticker_col : ticker column name (None for single asset)
    test_size : proportion for test set
    val_size : proportion for validation set
    
    Returns
    -------
    train_df : training data
    val_df : validation data
    test_df : test data
    metadata : dictionary with dataset information
    """
    print("\n" + "=" * 70)
    print("DATA PREPARATION PIPELINE".center(70))
    print("=" * 70 + "\n")
    
    # Step 1: Load data
    df = load_data(
        filepath=filepath,
        timestamp_col=timestamp_col,
        ticker_col=ticker_col,
        price_col=price_col,
        volume_col=volume_col,
    )
    
    # Step 2: Create target
    df = create_target_variable(
        df=df,
        price_col=price_col,
        horizon=horizon,
        target_type=target_type,
        group_by_ticker=(ticker_col is not None),
        ticker_col=ticker_col,
    )
    
    # Step 3: Create features
    df = create_features(
        df=df,
        price_col=price_col,
        volume_col=volume_col,
        windows=windows,
        group_by_ticker=(ticker_col is not None),
        ticker_col=ticker_col,
    )
    
    # Step 4: Describe data
    metadata = describe_data(df)
    
    # Step 5: Split data (time-based split, not random)
    n = len(df)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    metadata.update({
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_start": str(train_df.index[0]),
        "train_end": str(train_df.index[-1]),
        "val_start": str(val_df.index[0]),
        "val_end": str(val_df.index[-1]),
        "test_start": str(test_df.index[0]),
        "test_end": str(test_df.index[-1]),
    })
    
    print(f"\n[data] Split complete:")
    print(f"  Train: {len(train_df):,} rows ({metadata['train_start']} → {metadata['train_end']})")
    print(f"  Valid: {len(val_df):,} rows ({metadata['val_start']} → {metadata['val_end']})")
    print(f"  Test:  {len(test_df):,} rows ({metadata['test_start']} → {metadata['test_end']})")
    
    return train_df, val_df, test_df, metadata


def print_data_summary(metadata: Dict):
    """Pretty-print data summary."""
    print("\n" + "=" * 70)
    print("DATA SUMMARY".center(70))
    print("=" * 70)
    
    print(f"\n[Dataset Overview]")
    print(f"  Rows:               {metadata['rows']:,}")
    print(f"  Features:           {metadata['feature_count']}")
    print(f"  Date Range:         {metadata['date_range_start']} → {metadata['date_range_end']}")
    
    if 'unique_tickers' in metadata:
        print(f"  Unique Tickers:     {metadata['unique_tickers']}")
    
    print(f"\n[Target Variable (y)]")
    print(f"  Mean:               {metadata['target_mean']:.8f}")
    print(f"  Std Dev:            {metadata['target_std']:.8f}")
    print(f"  Min:                {metadata['target_min']:.8f}")
    print(f"  Max:                {metadata['target_max']:.8f}")
    print(f"  Skewness:           {metadata['target_skew']:.4f}")
    print(f"  Kurtosis:           {metadata['target_kurtosis']:.4f}")
    print(f"  Positive:           {metadata['target_positive_pct']:.2f}%")
    print(f"  Negative:           {metadata['target_negative_pct']:.2f}%")
    
    print(f"\n[Data Quality]")
    print(f"  Missing Values:     {metadata['missing_count']} ({metadata['missing_pct']}%)")
    
    print(f"\n[Data Split]")
    print(f"  Train:              {metadata['train_rows']:,} rows")
    print(f"  Validation:         {metadata['val_rows']:,} rows")
    print(f"  Test:               {metadata['test_rows']:,} rows")
    
    print("\n[Sample Features]")
    for feat in metadata['feature_names'][:10]:
        print(f"  • {feat}")
    
    if len(metadata['feature_names']) > 10:
        print(f"  ... and {len(metadata['feature_names']) - 10} more")
    
    print("=" * 70 + "\n")

