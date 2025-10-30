"""Tests for feature_engineering module."""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import (
    extract_trip_features,
    standardize_features,
    select_features_for_clustering,
    filter_trips,
)


@pytest.fixture
def sample_trajectory_data():
    """Create sample trajectory data with features."""
    timestamps = pd.date_range("2023-01-01 10:00:00", periods=100, freq="1min")
    
    return pd.DataFrame({
        "lat": np.linspace(39.9, 40.1, 100),
        "lon": np.full(100, 116.4),
        "altitude": np.full(100, 50.0),
        "timestamp": timestamps,
        "user_id": "test_user",
        "trajectory_id": "test_trip",
        "speed_kmh": np.full(100, 30.0),  # Constant speed
        "acceleration_ms2": np.zeros(100),  # No acceleration
        "distance_m": np.full(100, 500.0),
        "dt": np.full(100, 60.0),
        "duration_min": np.full(100, 60.0),
        "distance_km": np.full(100, 1.0),
    })


def test_extract_trip_features(sample_trajectory_data):
    """Test feature extraction."""
    features_df = extract_trip_features(sample_trajectory_data)
    
    # Should return one row per trajectory
    assert len(features_df) == 1
    
    # Check key features exist
    assert "trajectory_id" in features_df.columns
    assert "mean_speed_kmh" in features_df.columns
    assert "mean_accel_ms2" in features_df.columns
    assert "trip_duration_min" in features_df.columns
    assert "trip_distance_km" in features_df.columns
    
    # Values should match input
    assert abs(features_df["mean_speed_kmh"].iloc[0] - 30.0) < 0.1
    assert abs(features_df["mean_accel_ms2"].iloc[0]) < 0.1


def test_standardize_features(sample_trajectory_data):
    """Test feature standardization."""
    features_df = extract_trip_features(sample_trajectory_data)
    features_scaled = standardize_features(features_df)
    
    # Check that standardized columns exist
    assert any("_std" in col for col in features_scaled.columns)


def test_select_features_for_clustering(sample_trajectory_data):
    """Test feature selection for clustering."""
    features_df = extract_trip_features(sample_trajectory_data)
    features_scaled = standardize_features(features_df)
    
    feature_cols = select_features_for_clustering(features_scaled)
    
    # Should return non-empty list
    assert len(feature_cols) > 0
    
    # Should prefer standardized features if available
    if "_std" in " ".join(feature_cols):
        assert all("_std" in col for col in feature_cols)


def test_filter_trips():
    """Test filtering short trips."""
    features_df = pd.DataFrame({
        "trajectory_id": ["trip1", "trip2", "trip3"],
        "trip_duration_min": [0.5, 5.0, 10.0],
        "trip_distance_km": [0.05, 2.0, 5.0],
    })
    
    filtered = filter_trips(
        features_df, min_duration_min=1.0, min_distance_km=0.1
    )
    
    # Should remove trip1 (too short)
    assert len(filtered) == 2
    assert "trip1" not in filtered["trajectory_id"].values


def test_extract_trip_features_multiple_trips(sample_trajectory_data):
    """Test feature extraction with multiple trips."""
    # Create two trips
    trip1 = sample_trajectory_data.copy()
    trip2 = sample_trajectory_data.copy()
    trip2["trajectory_id"] = "test_trip_2"
    
    multi_trip_df = pd.concat([trip1, trip2])
    
    features_df = extract_trip_features(multi_trip_df)
    
    # Should return features for both trips
    assert len(features_df) == 2
    assert set(features_df["trajectory_id"]) == {"test_trip", "test_trip_2"}

