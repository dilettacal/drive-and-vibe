"""
Visualization utilities for trajectories and behavior analysis.
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-darkgrid")


def plot_trajectory_map(
    df: pd.DataFrame,
    output_file: str | Path = "trajectory_map.html",
    color_by: str | None = None,
    show_events: bool = True,
) -> folium.Map:
    """
    Create an interactive Folium map showing trajectories.

    Args:
        df: DataFrame with trajectory data (lat, lon, timestamp)
        output_file: Path to save the HTML map
        color_by: Column to use for coloring (e.g., 'speed_kmh', 'behavior_type')
        show_events: Whether to mark event locations

    Returns:
        Folium map object
    """
    # Center map on data
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Group by trajectory if available
    if "trajectory_id" in df.columns:
        trajectories = df.groupby("trajectory_id")
    else:
        trajectories = [("single", df)]

    # Color scheme
    if color_by:
        if color_by == "behavior_type":
            colors = {"Aggressive": "red", "Normal": "blue", "Cautious": "green", "Unknown": "gray"}
        elif color_by in df.columns:
            # Use continuous color scale
            values = df[color_by]
            min_val = values.min()
            max_val = values.max()
            # Define color palette for continuous values (red to green)
            from matplotlib.cm import RdYlGn
            cmap = RdYlGn
        else:
            color_by = None

    # Plot each trajectory
    for traj_id, traj_df in trajectories:
        # Sort by timestamp
        traj_df = traj_df.sort_values("timestamp")

        # Get color
        if color_by == "behavior_type":
            behavior = traj_df[color_by].iloc[0] if color_by in traj_df.columns else "Unknown"
            color = colors.get(behavior, "gray")
        elif color_by:
            # Use continuous color scale based on mean value for this trajectory
            if max_val > min_val:  # Avoid division by zero
                # Normalize to 0-1 range
                normalized = (traj_df[color_by].mean() - min_val) / (max_val - min_val)
                # Get color from colormap
                rgba = cmap(normalized)
                # Convert to hex
                color = f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
            else:
                color = "blue"
        else:
            color = "blue"

        # Create polyline
        points = [[row["lat"], row["lon"]] for _, row in traj_df.iterrows()]

        if len(points) > 1:
            folium.PolyLine(
                points,
                color=color,
                weight=3,
                opacity=0.7,
                popup=f"Trajectory: {traj_id}",
            ).add_to(m)

        # Mark events
        if show_events:
            for event_col in ["is_hard_brake", "is_sudden_accel", "is_sharp_turn"]:
                if event_col in traj_df.columns:
                    event_points = traj_df[traj_df[event_col] == True]
                    for _, row in event_points.iterrows():
                        event_type = event_col.replace("is_", "").replace("_", " ").title()
                        folium.CircleMarker(
                            location=[row["lat"], row["lon"]],
                            radius=5,
                            popup=f"{event_type}<br>{traj_id}",
                            color="red",
                            fill=True,
                        ).add_to(m)

    # Save map
    m.save(str(output_file))
    logger.info(f"Saved trajectory map to {output_file}")

    return m


def plot_speed_timeline(df: pd.DataFrame, trajectory_id: str, output_file: str | Path | None = None) -> None:
    """
    Plot speed and acceleration timeline for a single trip.

    Args:
        df: DataFrame with trajectory data
        trajectory_id: ID of trajectory to plot
        output_file: Optional path to save figure
    """
    # Select trajectory
    if "trajectory_id" in df.columns:
        traj_df = df[df["trajectory_id"] == trajectory_id].sort_values("timestamp")
    else:
        traj_df = df.sort_values("timestamp")

    if len(traj_df) == 0:
        logger.warning(f"No data found for trajectory {trajectory_id}")
        return

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot speed
    axes[0].plot(traj_df["timestamp"], traj_df["speed_kmh"], linewidth=2, color="blue", label="Speed")
    axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[0].set_ylabel("Speed (km/h)", fontsize=12)
    axes[0].set_title(f"Speed Timeline - {trajectory_id}", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot acceleration
    axes[1].plot(traj_df["timestamp"], traj_df["acceleration_ms2"], linewidth=2, color="red", label="Acceleration")
    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    # Mark events
    if "is_hard_brake" in traj_df.columns:
        brakes = traj_df[traj_df["is_hard_brake"] == True]
        axes[1].scatter(brakes["timestamp"], brakes["acceleration_ms2"], color="red", s=100, label="Hard brake", marker="v")
    if "is_sudden_accel" in traj_df.columns:
        accels = traj_df[traj_df["is_sudden_accel"] == True]
        axes[1].scatter(accels["timestamp"], accels["acceleration_ms2"], color="green", s=100, label="Sudden accel", marker="^")

    axes[1].set_xlabel("Time", fontsize=12)
    axes[1].set_ylabel("Acceleration (m/s²)", fontsize=12)
    axes[1].set_title("Acceleration Timeline", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved timeline plot to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_behavior_clusters(df: pd.DataFrame, output_file: str | Path | None = None) -> None:
    """
    Visualize behavior clusters in feature space.

    Args:
        df: DataFrame with cluster labels
        output_file: Optional path to save figure
    """
    if "behavior_type" not in df.columns:
        logger.warning("No behavior_type column found")
        return

    # Select key features for visualization
    feature_pairs = [
        ("mean_speed_kmh", "mean_accel_ms2"),
        ("std_speed_kmh", "std_accel_ms2"),
        ("max_speed_kmh", "pct_time_stopped"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (feat_x, feat_y) in zip(axes, feature_pairs):
        if feat_x not in df.columns or feat_y not in df.columns:
            continue

        # Plot each behavior type
        for behavior in df["behavior_type"].unique():
            behavior_df = df[df["behavior_type"] == behavior]
            ax.scatter(behavior_df[feat_x], behavior_df[feat_y], label=behavior, alpha=0.6, s=50)

        ax.set_xlabel(feat_x.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel(feat_y.replace("_", " ").title(), fontsize=10)
        ax.set_title(f"{feat_x} vs {feat_y}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved cluster plot to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_cluster_comparison(df: pd.DataFrame, output_file: str | Path | None = None) -> None:
    """
    Compare cluster characteristics using KDE plots.

    Args:
        df: DataFrame with cluster labels and features
        output_file: Optional path to save figure
    """
    if "behavior_type" not in df.columns:
        logger.warning("No behavior_type column found")
        return

    # Key features to compare
    features = ["mean_speed_kmh", "std_accel_ms2", "pct_time_stopped", "mean_jerk"]
    features = [f for f in features if f in df.columns]

    if not features:
        logger.warning("No suitable features found for comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, feature in zip(axes, features[:4]):
        # KDE plot for each behavior type
        for behavior in df["behavior_type"].unique():
            behavior_df = df[df["behavior_type"] == behavior]
            ax.hist(
                behavior_df[feature],
                bins=20,
                alpha=0.6,
                label=behavior,
                density=True,
            )

        ax.set_xlabel(feature.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"Distribution of {feature}", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {output_file}")
    else:
        plt.show()

    plt.close()


def create_dashboard(df_trajectories: pd.DataFrame, df_features: pd.DataFrame, output_file: str | Path) -> None:
    """
    Create a combined dashboard with maps and statistics.

    Args:
        df_trajectories: DataFrame with raw trajectory data
        df_features: DataFrame with trip features and clusters
        output_file: Path to save HTML dashboard
    """
    # This would create a comprehensive HTML dashboard
    # For now, just create a trajectory map
    logger.info("Creating dashboard (trajectory map)...")
    plot_trajectory_map(df_trajectories, output_file=output_file, color_by="behavior_type", show_events=True)

