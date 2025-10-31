"""
Feature extraction for trip behavior analysis.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract aggregate features per trip/trajectory.

    Args:
        df: DataFrame with trajectory data including speed, acceleration, etc.

    Returns:
        DataFrame with one row per trip and aggregate features
    """
    if "trajectory_id" not in df.columns:
        logger.warning("No trajectory_id column found, treating as single trip")
        df["trajectory_id"] = "trip_0"

    # Group by trajectory
    grouped = df.groupby("trajectory_id")

    features = []

    for traj_id, group_df in grouped:
        # Basic statistics
        mean_speed = group_df["speed_kmh"].mean()
        std_speed = group_df["speed_kmh"].std()
        max_speed = group_df["speed_kmh"].max()

        mean_accel = group_df["acceleration_ms2"].mean()
        std_accel = group_df["acceleration_ms2"].std()
        max_accel = group_df["acceleration_ms2"].max()
        min_accel = group_df["acceleration_ms2"].min()

        # Percent of time stopped (< 1 km/h)
        stopped_count = (group_df["speed_kmh"] < 1.0).sum()
        pct_stopped = stopped_count / len(group_df) * 100

        # Trip characteristics
        trip_duration_min = (
            group_df["duration_min"].iloc[0] if "duration_min" in group_df.columns else np.nan
        )
        trip_distance_km = (
            group_df["distance_km"].iloc[0] if "distance_km" in group_df.columns else np.nan
        )

        # Average speed (distance/time)
        if trip_duration_min > 0:
            avg_speed_distance = trip_distance_km / (trip_duration_min / 60)
        else:
            avg_speed_distance = np.nan

        # Jerk (rate of change of acceleration) - indicator of smoothness
        jerk = group_df["acceleration_ms2"].diff().abs().mean()
        std_jerk = group_df["acceleration_ms2"].diff().abs().std()

        # Speed consistency (coefficient of variation)
        speed_cv = std_speed / mean_speed if mean_speed > 0 else np.nan

        # Metadata
        user_id = group_df["user_id"].iloc[0] if "user_id" in group_df.columns else "unknown"
        n_points = len(group_df)

        feat_dict = {
            "trajectory_id": traj_id,
            "user_id": user_id,
            "n_points": n_points,
            "trip_duration_min": trip_duration_min,
            "trip_distance_km": trip_distance_km,
            "mean_speed_kmh": mean_speed,
            "std_speed_kmh": std_speed,
            "max_speed_kmh": max_speed,
            "speed_cv": speed_cv,
            "avg_speed_distance_kmh": avg_speed_distance,
            "mean_accel_ms2": mean_accel,
            "std_accel_ms2": std_accel,
            "max_accel_ms2": max_accel,
            "min_accel_ms2": min_accel,
            "pct_time_stopped": pct_stopped,
            "mean_jerk": jerk,
            "std_jerk": std_jerk,
        }

        features.append(feat_dict)

    features_df = pd.DataFrame(features)
    logger.info(f"Extracted features for {len(features_df)} trips")
    return features_df


def standardize_features(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Standardize features for clustering (z-score normalization).

    Args:
        df: DataFrame with features
        feature_cols: List of columns to standardize. If None, uses all numeric columns.

    Returns:
        DataFrame with standardized features added as *_std columns
    """
    if feature_cols is None:
        # Exclude metadata columns
        exclude_cols = ["trajectory_id", "user_id"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    df_scaled = df.copy()

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[feature_cols])

    # Add scaled features
    for i, col in enumerate(feature_cols):
        df_scaled[f"{col}_std"] = features_scaled[:, i]

    return df_scaled


def select_features_for_clustering(df: pd.DataFrame) -> list[str]:
    """
    Select which features to use for behavior clustering.

    Args:
        df: DataFrame with features

    Returns:
        List of feature column names
    """
    # Key behavioral features
    base_features = [
        "mean_speed_kmh",
        "std_speed_kmh",
        "max_speed_kmh",
        "mean_accel_ms2",
        "std_accel_ms2",
        "pct_time_stopped",
        "mean_jerk",
    ]

    # Check which features exist
    available_features = [f for f in base_features if f in df.columns]

    # Add standardized versions if they exist
    std_features = [f"{f}_std" for f in base_features if f"{f}_std" in df.columns]

    # Prefer standardized if available
    if std_features:
        return std_features

    return available_features


def filter_trips(
    df: pd.DataFrame, min_duration_min: float = 1.0, min_distance_km: float = 0.1
) -> pd.DataFrame:
    """
    Filter out trips that are too short to be meaningful.

    Args:
        df: DataFrame with trip features
        min_duration_min: Minimum trip duration in minutes
        min_distance_km: Minimum trip distance in kilometers

    Returns:
        Filtered DataFrame
    """
    n_before = len(df)

    # Filter short trips
    if "trip_duration_min" in df.columns:
        df = df[df["trip_duration_min"] >= min_duration_min].copy()

    if "trip_distance_km" in df.columns:
        df = df[df["trip_distance_km"] >= min_distance_km].copy()

    n_after = len(df)
    logger.info(f"Filtered {n_before - n_after} short trips")

    return df.reset_index(drop=True)
