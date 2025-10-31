"""
GPS utilities for parsing and validating Geolife trajectory data.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_plt_file(filepath: Path) -> pd.DataFrame:
    """
    Load a single .plt GPS trajectory file from Geolife dataset.

    Args:
        filepath: Path to the .plt file

    Returns:
        DataFrame with columns: lat, lon, altitude, date, timestamp, raw
    """
    # Skip first 6 lines which are metadata
    # Format: lat,lon,zero,altitude,number,date,time
    df = pd.read_csv(
        filepath,
        skiprows=6,
        names=["lat", "lon", "_zero", "altitude", "_number", "date", "time"],
        index_col=False,  # Don't use first column as index
    )

    # Combine date and time into timestamp
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df["raw"] = False  # Flag for raw data before cleaning

    # Drop unnecessary columns
    df = df.drop(columns=["_zero", "_number", "date", "time"])

    # Reset index to avoid index becoming the lat column after concat
    df = df.reset_index(drop=True)

    logger.info(f"Loaded {len(df)} points from {filepath.name}")
    return df


def load_all_trajectories(data_dir: Path) -> pd.DataFrame:
    """
    Load all .plt files from the Geolife dataset.

    Args:
        data_dir: Directory containing Geolife subdirectories

    Returns:
        DataFrame with all trajectories concatenated
    """
    all_data = []
    plt_files = list(data_dir.rglob("*.plt"))

    logger.info(f"Found {len(plt_files)} .plt files")

    for filepath in plt_files:
        try:
            df = load_plt_file(filepath)
            # Add metadata from file structure
            parts = filepath.parts
            df["user_id"] = parts[-4] if len(parts) >= 4 else "unknown"
            df["trajectory_id"] = filepath.stem
            df["filepath"] = str(filepath.relative_to(data_dir))
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            continue

    if not all_data:
        raise ValueError("No trajectory files found")

    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(df_all)} total points from {len(plt_files)} files")
    return df_all


def calculate_speed_acceleration(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Calculate speed (km/h) and acceleration (m/s²) from GPS coordinates.

    Args:
        df: DataFrame with GPS coordinates
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with added speed and acceleration columns
    """
    df = df.copy()
    df = df.sort_values([timestamp_col]).reset_index(drop=True)

    # Calculate time delta in seconds
    time_delta = (df[timestamp_col] - df[timestamp_col].shift(1)).dt.total_seconds()
    df["dt"] = time_delta

    # Calculate distance using Haversine formula
    from geopy.distance import geodesic

    def calculate_distance(row, prev_row):
        if pd.isna(row["dt"]) or pd.isna(prev_row["lat"]):
            return 0.0
        try:
            # geodesic expects (latitude, longitude) tuple
            point1 = (float(row[lat_col]), float(row[lon_col]))
            point2 = (float(prev_row[lat_col]), float(prev_row[lon_col]))
            return geodesic(point1, point2).meters
        except (ValueError, TypeError) as e:
            logger.debug(f"Distance calculation failed: {e}")
            return 0.0
        except Exception as e:
            logger.warning(f"Unexpected error in distance calculation: {e}")
            return 0.0

    # Calculate distances (optimized with vectorization where possible)
    df["distance_m"] = 0.0

    # Process in chunks to avoid memory issues
    chunk_size = 100000
    total_chunks = (len(df) + chunk_size - 1) // chunk_size

    logger.info(f"Calculating distances for {len(df):,} points in {total_chunks} chunks...")
    for i in range(1, len(df)):
        if i % chunk_size == 0:
            logger.info(f"  Processing point {i:,} / {len(df):,}")
        df.loc[i, "distance_m"] = calculate_distance(df.iloc[i], df.iloc[i - 1])

    # Speed in m/s, then convert to km/h
    df["speed_mps"] = df["distance_m"] / df["dt"]
    df["speed_kmh"] = df["speed_mps"] * 3.6

    # Acceleration in m/s²
    speed_change = df["speed_mps"] - df["speed_mps"].shift(1)
    df["acceleration_ms2"] = speed_change / df["dt"]

    # Replace infinite and NaN values with 0
    df["speed_kmh"] = df["speed_kmh"].replace([np.inf, -np.inf], 0).fillna(0)
    df["acceleration_ms2"] = df["acceleration_ms2"].replace([np.inf, -np.inf], 0).fillna(0)

    return df


def filter_unrealistic_data(
    df: pd.DataFrame,
    max_speed_kmh: float = 200.0,
    max_accel_ms2: float = 10.0,
    min_time_gap_sec: float = 0.0,
    max_time_gap_sec: float = 300.0,
) -> pd.DataFrame:
    """
    Filter out unrealistic GPS points (teleporting, impossible speeds).

    Args:
        df: DataFrame with speed and acceleration data
        max_speed_kmh: Maximum reasonable speed in km/h
        max_accel_ms2: Maximum reasonable acceleration in m/s²
        min_time_gap_sec: Minimum time gap between points (seconds)
        max_time_gap_sec: Maximum time gap between points (seconds)

    Returns:
        Filtered DataFrame
    """
    n_before = len(df)

    # Filter unrealistic speeds
    df = df[df["speed_kmh"] <= max_speed_kmh].copy()

    # Filter unrealistic accelerations
    df = df[df["acceleration_ms2"].abs() <= max_accel_ms2].copy()

    # Filter time gaps
    if "dt" in df.columns:
        df = df[(df["dt"] >= min_time_gap_sec) & (df["dt"] <= max_time_gap_sec)].copy()

    # Filter altitude outliers (more than 3 std away from mean)
    if "altitude" in df.columns:
        mean_alt = df["altitude"].mean()
        std_alt = df["altitude"].std()
        df = df[abs(df["altitude"] - mean_alt) <= 3 * std_alt].copy()

    n_after = len(df)
    removed = n_before - n_after
    logger.info(f"Filtered {removed} unrealistic points ({removed/n_before*100:.1f}%)")

    return df.reset_index(drop=True)


def compute_trip_duration_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trip duration and total distance for each trajectory.

    Args:
        df: DataFrame with trajectory data

    Returns:
        DataFrame with added duration_min and distance_km columns
    """
    df = df.copy()

    # Group by trajectory if trajectory_id exists
    group_cols = []
    if "trajectory_id" in df.columns:
        group_cols.append("trajectory_id")

    if group_cols:
        df_grouped = df.groupby(group_cols)
    else:
        # Single trip
        df_grouped = [("all", df)]

    result_dfs = []
    # Handle both DataFrameGroupBy objects and iterables
    if hasattr(df_grouped, "groups"):
        # DataFrameGroupBy object
        for name, group_df in df_grouped:
            group_df = group_df.sort_values("timestamp")
            max_ts = group_df["timestamp"].max()
            min_ts = group_df["timestamp"].min()
            duration_sec = (max_ts - min_ts).total_seconds()
            distance_km = group_df["distance_m"].sum() / 1000

            group_df["duration_min"] = duration_sec / 60
            group_df["distance_km"] = distance_km

            result_dfs.append(group_df)
    else:
        # Iterable of tuples
        for name, group_df in df_grouped:
            group_df = group_df.sort_values("timestamp")
            max_ts = group_df["timestamp"].max()
            min_ts = group_df["timestamp"].min()
            duration_sec = (max_ts - min_ts).total_seconds()
            distance_km = group_df["distance_m"].sum() / 1000

            group_df["duration_min"] = duration_sec / 60
            group_df["distance_km"] = distance_km

            result_dfs.append(group_df)

    return pd.concat(result_dfs, ignore_index=True)


def clean_geolife_data(
    data_dir: Path,
    max_speed_kmh: float = 200.0,
    max_accel_ms2: float = 10.0,
) -> pd.DataFrame:
    """
    Complete pipeline: load, calculate, and clean Geolife data.

    Args:
        data_dir: Directory containing Geolife data
        max_speed_kmh: Maximum speed threshold
        max_accel_ms2: Maximum acceleration threshold

    Returns:
        Clean DataFrame with all computed features
    """
    logger.info("Loading all trajectories...")
    df = load_all_trajectories(data_dir)

    logger.info("Calculating speed and acceleration...")
    df = calculate_speed_acceleration(df)

    logger.info("Filtering unrealistic data...")
    df = filter_unrealistic_data(df, max_speed_kmh, max_accel_ms2)

    logger.info("Computing trip statistics...")
    df = compute_trip_duration_distance(df)

    logger.info(f"Final dataset: {len(df)} points")
    return df
