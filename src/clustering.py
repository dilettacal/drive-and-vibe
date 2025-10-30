"""
Behavior clustering for classifying driving styles.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


def cluster_trips_kmeans(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster trips using KMeans.

    Args:
        df: DataFrame with trip features
        feature_cols: List of feature columns to use
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        DataFrame with cluster labels added
    """
    # Select features
    X = df[feature_cols].values

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Add labels
    df_labeled = df.copy()
    df_labeled["cluster"] = cluster_labels

    # Calculate silhouette score
    silhouette = silhouette_score(X, cluster_labels)
    logger.info(f"KMeans clustering: {n_clusters} clusters, silhouette score: {silhouette:.3f}")

    # Print cluster statistics
    for i in range(n_clusters):
        cluster_df = df_labeled[df_labeled["cluster"] == i]
        logger.info(f"Cluster {i}: {len(cluster_df)} trips")

    return df_labeled


def cluster_trips_dbscan(
    df: pd.DataFrame,
    feature_cols: list[str],
    eps: float = 0.5,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Cluster trips using DBSCAN.

    Args:
        df: DataFrame with trip features
        feature_cols: List of feature columns to use
        eps: Maximum distance between samples in the same cluster
        min_samples: Minimum number of samples in a cluster

    Returns:
        DataFrame with cluster labels added (-1 for noise)
    """
    # Select features
    X = df[feature_cols].values

    # Fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)

    # Add labels
    df_labeled = df.copy()
    df_labeled["cluster"] = cluster_labels

    # Print cluster statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    logger.info(f"DBSCAN clustering: {n_clusters} clusters, {n_noise} noise points")

    for i in range(n_clusters):
        cluster_df = df_labeled[df_labeled["cluster"] == i]
        logger.info(f"Cluster {i}: {len(cluster_df)} trips")

    if n_noise > 0:
        logger.info(f"Noise: {n_noise} trips")

    return df_labeled


def classify_driving_behavior(
    df: pd.DataFrame,
    method: str = "kmeans",
    n_clusters: int = 3,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Classify trips into behavior types (aggressive, normal, cautious).

    Args:
        df: DataFrame with trip features
        method: Clustering method ('kmeans' or 'dbscan')
        n_clusters: Number of clusters (for KMeans)
        feature_cols: Features to use. If None, auto-select.

    Returns:
        DataFrame with cluster labels and behavior type
    """
    if feature_cols is None:
        # Use standardized features if available
        std_cols = [col for col in df.columns if col.endswith("_std")]
        if std_cols:
            feature_cols = std_cols
        else:
            # Fallback to non-standardized
            from .feature_engineering import select_features_for_clustering

            feature_cols = select_features_for_clustering(df)

    logger.info(f"Clustering using {len(feature_cols)} features")

    # Perform clustering
    if method.lower() == "kmeans":
        df_labeled = cluster_trips_kmeans(df, feature_cols, n_clusters)
    elif method.lower() == "dbscan":
        df_labeled = cluster_trips_dbscan(df, feature_cols)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Assign behavior types based on cluster characteristics
    df_labeled = assign_behavior_types(df_labeled)

    return df_labeled


def assign_behavior_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable behavior types to clusters based on their characteristics.

    Args:
        df: DataFrame with cluster labels

    Returns:
        DataFrame with behavior_type column added
    """
    df_labeled = df.copy()

    # Calculate cluster centroids for key features
    cluster_stats = []
    for cluster_id in df_labeled["cluster"].unique():
        if cluster_id == -1:  # DBSCAN noise
            continue

        cluster_df = df_labeled[df_labeled["cluster"] == cluster_id]
        stats = {
            "cluster": cluster_id,
            "mean_speed": cluster_df["mean_speed_kmh"].mean(),
            "mean_accel": cluster_df["mean_accel_ms2"].mean(),
            "std_accel": cluster_df["std_accel_ms2"].mean(),
            "pct_stopped": cluster_df["pct_time_stopped"].mean(),
        }
        cluster_stats.append(stats)

    if not cluster_stats:
        # No valid clusters
        df_labeled["behavior_type"] = "unknown"
        return df_labeled

    stats_df = pd.DataFrame(cluster_stats)

    # Classify based on characteristics
    # Aggressive: high speed, high acceleration variation
    # Normal: medium characteristics
    # Cautious: low speed, high % stopped
    behavior_map = {}

    # Sort clusters by average speed
    stats_df = stats_df.sort_values("mean_speed", ascending=False)

    # Assign behaviors
    n_clusters = len(stats_df)
    if n_clusters == 3:
        behavior_map[stats_df.iloc[0]["cluster"]] = "Aggressive"  # Fastest
        behavior_map[stats_df.iloc[1]["cluster"]] = "Normal"
        behavior_map[stats_df.iloc[2]["cluster"]] = "Cautious"  # Slowest
    elif n_clusters == 2:
        behavior_map[stats_df.iloc[0]["cluster"]] = "Aggressive"
        behavior_map[stats_df.iloc[1]["cluster"]] = "Normal"
    else:
        # More or fewer clusters - use generic labels
        for i, row in stats_df.iterrows():
            behavior_map[row["cluster"]] = f"Style_{int(row['cluster'])}"

    # Add noise label for DBSCAN
    if -1 in df_labeled["cluster"].values:
        behavior_map[-1] = "Noise"

    # Assign behavior types
    df_labeled["behavior_type"] = df_labeled["cluster"].map(behavior_map)
    df_labeled["behavior_type"] = df_labeled["behavior_type"].fillna("Unknown")

    logger.info(f"Assigned behavior types: {df_labeled['behavior_type'].value_counts().to_dict()}")

    return df_labeled


def get_cluster_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for each behavior cluster.

    Args:
        df: DataFrame with clusters and features

    Returns:
        DataFrame with cluster characteristics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ["cluster"]]

    cluster_characteristics = df.groupby("behavior_type")[numeric_cols].agg(["mean", "std"])

    return cluster_characteristics

