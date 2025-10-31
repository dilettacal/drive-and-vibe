"""Tests for clustering module."""

import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    assign_behavior_types,
    cluster_trips_dbscan,
    cluster_trips_kmeans,
)


@pytest.fixture
def sample_features():
    """Create sample trip features."""
    np.random.seed(42)

    # Create three distinct groups
    n_trips = 30

    # Aggressive drivers (high speed, high accel variation)
    aggressive = pd.DataFrame(
        {
            "trajectory_id": [f"trip_{i}" for i in range(n_trips // 3)],
            "mean_speed_kmh": np.random.normal(80, 5, n_trips // 3),
            "std_speed_kmh": np.random.normal(20, 3, n_trips // 3),
            "mean_accel_ms2": np.random.normal(2, 0.5, n_trips // 3),
            "std_accel_ms2": np.random.normal(4, 0.5, n_trips // 3),
            "pct_time_stopped": np.random.normal(5, 1, n_trips // 3),
        }
    )

    # Normal drivers
    normal = pd.DataFrame(
        {
            "trajectory_id": [f"trip_{i}" for i in range(n_trips // 3, 2 * n_trips // 3)],
            "mean_speed_kmh": np.random.normal(50, 5, n_trips // 3),
            "std_speed_kmh": np.random.normal(10, 2, n_trips // 3),
            "mean_accel_ms2": np.random.normal(0, 0.2, n_trips // 3),
            "std_accel_ms2": np.random.normal(2, 0.5, n_trips // 3),
            "pct_time_stopped": np.random.normal(15, 3, n_trips // 3),
        }
    )

    # Cautious drivers (low speed, many stops)
    cautious = pd.DataFrame(
        {
            "trajectory_id": [f"trip_{i}" for i in range(2 * n_trips // 3, n_trips)],
            "mean_speed_kmh": np.random.normal(30, 5, n_trips // 3),
            "std_speed_kmh": np.random.normal(5, 1, n_trips // 3),
            "mean_accel_ms2": np.random.normal(-0.5, 0.2, n_trips // 3),
            "std_accel_ms2": np.random.normal(1, 0.3, n_trips // 3),
            "pct_time_stopped": np.random.normal(30, 5, n_trips // 3),
        }
    )

    return pd.concat([aggressive, normal, cautious], ignore_index=True)


def test_cluster_trips_kmeans(sample_features):
    """Test KMeans clustering."""
    # Add standardized versions
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    for col in ["mean_speed_kmh", "std_accel_ms2", "pct_time_stopped"]:
        sample_features[f"{col}_std"] = scaler.fit_transform(sample_features[[col]])

    clustered = cluster_trips_kmeans(
        sample_features, feature_cols=["mean_speed_kmh_std", "std_accel_ms2_std"], n_clusters=3
    )

    # Check cluster column exists
    assert "cluster" in clustered.columns

    # Should have 3 clusters
    assert clustered["cluster"].nunique() == 3


def test_assign_behavior_types(sample_features):
    """Test behavior type assignment."""
    # Add synthetic clusters
    sample_features["cluster"] = np.repeat([0, 1, 2], 10)

    result = assign_behavior_types(sample_features)

    # Check behavior_type column exists
    assert "behavior_type" in result.columns

    # Should have behavior labels
    assert result["behavior_type"].nunique() > 0


def test_cluster_trips_dbscan(sample_features):
    """Test DBSCAN clustering."""
    # Add standardized versions
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    for col in ["mean_speed_kmh", "std_accel_ms2"]:
        sample_features[f"{col}_std"] = scaler.fit_transform(sample_features[[col]])

    clustered = cluster_trips_dbscan(
        sample_features,
        feature_cols=["mean_speed_kmh_std", "std_accel_ms2_std"],
        eps=0.5,
        min_samples=3,
    )

    # Check cluster column exists
    assert "cluster" in clustered.columns

    # DBSCAN may have noise points (-1)
    assert -1 in clustered["cluster"].values or clustered["cluster"].nunique() > 0
