"""Tests for geo_utils module."""

import numpy as np
import pandas as pd
import pytest

from src.geo_utils import (
    calculate_speed_acceleration,
    compute_trip_duration_distance,
    filter_unrealistic_data,
)


@pytest.fixture
def sample_gps_data():
    """Create sample GPS trajectory data."""
    # Create a simple trajectory with known properties
    timestamps = pd.date_range("2023-01-01 10:00:00", periods=10, freq="5min")

    # Move north at constant speed (approximately 30 km/h)
    # Each point moves about 2.5 km north
    lats = np.linspace(39.9, 40.1, 10)
    lons = np.full(10, 116.4)  # Constant longitude

    return pd.DataFrame(
        {
            "lat": lats,
            "lon": lons,
            "altitude": np.full(10, 50.0),
            "timestamp": timestamps,
            "user_id": "test_user",
            "trajectory_id": "test_trip",
        }
    )


def test_calculate_speed_acceleration(sample_gps_data):
    """Test speed and acceleration calculation."""
    df = calculate_speed_acceleration(sample_gps_data)

    # Check that speed and acceleration columns exist
    assert "speed_kmh" in df.columns
    assert "acceleration_ms2" in df.columns
    assert "distance_m" in df.columns

    # First point should have 0 speed (no previous point)
    assert df.iloc[0]["speed_kmh"] == 0

    # Speed should be approximately constant (30 km/h)
    non_zero_speeds = df["speed_kmh"][df["speed_kmh"] > 0]
    assert len(non_zero_speeds) > 0
    assert abs(non_zero_speeds.mean() - 30) < 10  # Within 10 km/h


def test_filter_unrealistic_data(sample_gps_data):
    """Test filtering unrealistic GPS data."""
    # Add some unrealistic points
    df_with_extremes = sample_gps_data.copy()
    df_with_extremes = calculate_speed_acceleration(df_with_extremes)

    # Add unrealistic speed
    df_with_extremes.loc[len(df_with_extremes)] = {
        "lat": 40.2,
        "lon": 116.4,
        "altitude": 50,
        "timestamp": pd.Timestamp("2023-01-01 10:50:00"),
        "user_id": "test_user",
        "trajectory_id": "test_trip",
        "speed_kmh": 500,  # Unrealistic
        "acceleration_ms2": 0,
        "distance_m": 0,
        "dt": 300,
    }

    df_filtered = filter_unrealistic_data(df_with_extremes, max_speed_kmh=200.0, max_accel_ms2=10.0)

    # Unrealistic point should be removed
    assert len(df_filtered) < len(df_with_extremes)
    assert df_filtered["speed_kmh"].max() <= 200.0


def test_compute_trip_duration_distance(sample_gps_data):
    """Test trip duration and distance calculation."""
    df = calculate_speed_acceleration(sample_gps_data)
    df = compute_trip_duration_distance(df)

    # Check that columns exist
    assert "duration_min" in df.columns
    assert "distance_km" in df.columns

    # Duration should be approximately 45 minutes (9 intervals * 5 min)
    assert abs(df["duration_min"].iloc[0] - 45) < 1


def test_filter_unrealistic_data_altitude(sample_gps_data):
    """Test altitude filtering."""
    df = calculate_speed_acceleration(sample_gps_data)

    # Make sure all altitudes are reasonable first, then add one outlier
    df["altitude"] = 50.0

    # Add outlier altitude with all required columns
    new_row = pd.DataFrame(
        {
            "lat": [40.2],
            "lon": [116.4],
            "altitude": [50000],  # Extreme altitude
            "timestamp": [pd.Timestamp("2023-01-01 10:50:00")],
            "user_id": ["test_user"],
            "trajectory_id": ["test_trip"],
            "dt": [300.0],
            "distance_m": [0.0],
            "speed_mps": [0.0],
            "speed_kmh": [0.0],
            "acceleration_ms2": [0.0],
        }
    )

    df_before = pd.concat([df, new_row], ignore_index=True)
    df_filtered = filter_unrealistic_data(df_before)

    # Extreme altitude should be removed or significantly reduced
    # Test that filtering occurred
    assert len(df_filtered) < len(df_before) or df_filtered["altitude"].max() < 50000
