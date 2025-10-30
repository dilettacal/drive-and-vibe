"""Tests for events module."""

import pytest
import pandas as pd
import numpy as np
from src.events import (
    detect_hard_braking,
    detect_sudden_acceleration,
    detect_stops,
    detect_all_events,
    get_event_summary,
)


@pytest.fixture
def sample_trajectory_data():
    """Create sample trajectory data."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01 10:00:00", periods=100, freq="1min"),
        "lat": np.linspace(39.9, 40.1, 100),
        "lon": np.full(100, 116.4),
        "speed_kmh": np.concatenate([
            np.full(50, 50.0),  # Normal driving
            [0.0],  # Stop
            np.full(49, 50.0),
        ]),
        "acceleration_ms2": np.concatenate([
            np.full(40, 0.0),  # Normal
            np.full(10, -5.0),  # Hard brakes
            [0.0],
            np.full(49, 0.0),
        ]),
        "trajectory_id": "test_trip",
        "user_id": "test_user",
    })


def test_detect_hard_braking(sample_trajectory_data):
    """Test hard braking detection."""
    df = detect_hard_braking(sample_trajectory_data, threshold_ms2=-3.0)
    
    # Check column exists
    assert "is_hard_brake" in df.columns
    
    # Should detect hard brakes where accel < -3 m/s²
    n_brakes = df["is_hard_brake"].sum()
    assert n_brakes >= 10  # We added 10 points with accel = -5


def test_detect_sudden_acceleration(sample_trajectory_data):
    """Test sudden acceleration detection."""
    # Add sudden acceleration points
    df = sample_trajectory_data.copy()
    df.loc[10:15, "acceleration_ms2"] = 5.0  # Above threshold
    
    df = detect_sudden_acceleration(df, threshold_ms2=3.0)
    
    # Check column exists
    assert "is_sudden_accel" in df.columns
    
    # Should detect sudden accelerations
    n_accels = df["is_sudden_accel"].sum()
    assert n_accels >= 6


def test_detect_stops(sample_trajectory_data):
    """Test stop detection."""
    df = detect_stops(sample_trajectory_data, speed_threshold_kmh=1.0)
    
    # Check column exists
    assert "is_stopped" in df.columns
    
    # Should detect the stop point
    n_stops = df["is_stopped"].sum()
    assert n_stops >= 1


def test_get_event_summary(sample_trajectory_data):
    """Test event summary generation."""
    df = detect_hard_braking(sample_trajectory_data)
    df = detect_sudden_acceleration(df)
    df = detect_stops(df)
    
    summary = get_event_summary(df)
    
    # Should have one row per trajectory
    assert len(summary) == 1
    
    # Check summary columns
    assert "trajectory_id" in summary.columns
    assert "n_hard_brakes" in summary.columns
    assert "n_stop_events" in summary.columns


def test_get_event_summary_multiple_trips(sample_trajectory_data):
    """Test event summary with multiple trips."""
    trip1 = sample_trajectory_data.copy()
    trip2 = sample_trajectory_data.copy()
    trip2["trajectory_id"] = "trip_2"
    
    multi_trip_df = pd.concat([trip1, trip2])
    multi_trip_df = detect_all_events(multi_trip_df)
    
    summary = get_event_summary(multi_trip_df)
    
    # Should have summary for both trips
    assert len(summary) == 2
    assert set(summary["trajectory_id"]) == {"test_trip", "trip_2"}


def test_detect_all_events(sample_trajectory_data):
    """Test detecting all events at once."""
    df = detect_all_events(sample_trajectory_data)
    
    # Check all event columns exist
    assert "is_hard_brake" in df.columns
    assert "is_sudden_accel" in df.columns
    assert "is_stopped" in df.columns
    assert "is_sharp_turn" in df.columns

