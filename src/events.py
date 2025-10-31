"""
Event detection for finding interesting driving events (hard braking, acceleration, etc.).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def detect_hard_braking(df: pd.DataFrame, threshold_ms2: float = -3.0) -> pd.DataFrame:
    """
    Detect hard braking events (sudden deceleration).

    Args:
        df: DataFrame with trajectory data
        threshold_ms2: Acceleration threshold (negative = braking)

    Returns:
        DataFrame with is_hard_brake column
    """
    df = df.copy()
    df["is_hard_brake"] = df["acceleration_ms2"] < threshold_ms2
    return df


def detect_sudden_acceleration(df: pd.DataFrame, threshold_ms2: float = 3.0) -> pd.DataFrame:
    """
    Detect sudden acceleration events.

    Args:
        df: DataFrame with trajectory data
        threshold_ms2: Acceleration threshold (positive = acceleration)

    Returns:
        DataFrame with is_sudden_accel column
    """
    df = df.copy()
    df["is_sudden_accel"] = df["acceleration_ms2"] > threshold_ms2
    return df


def detect_stops(df: pd.DataFrame, speed_threshold_kmh: float = 1.0) -> pd.DataFrame:
    """
    Detect stopped events (vehicle speed below threshold).

    Args:
        df: DataFrame with trajectory data
        speed_threshold_kmh: Speed threshold in km/h

    Returns:
        DataFrame with is_stopped column
    """
    df = df.copy()
    df["is_stopped"] = df["speed_kmh"] < speed_threshold_kmh
    return df


def detect_turns(df: pd.DataFrame, angle_threshold_deg: float = 30.0) -> pd.DataFrame:
    """
    Detect sharp turns based on heading changes.

    Args:
        df: DataFrame with trajectory data
        angle_threshold_deg: Minimum heading change in degrees

    Returns:
        DataFrame with is_sharp_turn column
    """
    df = df.copy()

    # Calculate heading from GPS points
    def calculate_bearing(lat1, lon1, lat2, lon2):
        """Calculate bearing between two GPS points."""
        from math import atan2, cos, degrees, radians, sin

        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        dlon_rad = radians(lon2 - lon1)

        y = sin(dlon_rad) * cos(lat2_rad)
        x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon_rad)

        bearing = degrees(atan2(y, x))
        return (bearing + 360) % 360

    # Calculate headings
    headings = []
    for i in range(len(df)):
        if i == 0:
            headings.append(0.0)
        else:
            heading = calculate_bearing(
                df.iloc[i - 1]["lat"],
                df.iloc[i - 1]["lon"],
                df.iloc[i]["lat"],
                df.iloc[i]["lon"],
            )
            headings.append(heading)

    df["heading_deg"] = headings

    # Calculate heading change
    df["heading_change"] = df["heading_deg"].diff().abs()
    # Handle wrap-around (359 -> 1 should be 2 degrees, not 358)
    df["heading_change"] = df["heading_change"].apply(lambda x: min(x, 360 - x))

    # Detect sharp turns
    df["is_sharp_turn"] = df["heading_change"] > angle_threshold_deg

    return df


def detect_all_events(
    df: pd.DataFrame,
    brake_threshold: float = -3.0,
    accel_threshold: float = 3.0,
    stop_threshold: float = 1.0,
    turn_threshold: float = 30.0,
) -> pd.DataFrame:
    """
    Detect all types of events.

    Args:
        df: DataFrame with trajectory data
        brake_threshold: Hard braking threshold (m/s²)
        accel_threshold: Sudden acceleration threshold (m/s²)
        stop_threshold: Stop speed threshold (km/h)
        turn_threshold: Sharp turn angle threshold (degrees)

    Returns:
        DataFrame with event detection columns
    """
    logger.info("Detecting driving events...")

    df = detect_hard_braking(df, brake_threshold)
    df = detect_sudden_acceleration(df, accel_threshold)
    df = detect_stops(df, stop_threshold)
    df = detect_turns(df, turn_threshold)

    # Summary statistics per trip
    if "trajectory_id" in df.columns:
        # Log summary statistics
        logger.info("Event detection complete:")
        logger.info(f"  Hard brakes: {df['is_hard_brake'].sum()}")
        logger.info(f"  Sudden accelerations: {df['is_sudden_accel'].sum()}")
        logger.info(f"  Stop events: {df['is_stopped'].sum()}")
        logger.info(f"  Sharp turns: {df['is_sharp_turn'].sum()}")

    return df


def get_event_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of events per trip.

    Args:
        df: DataFrame with event detection results

    Returns:
        DataFrame with event counts per trip
    """
    if "trajectory_id" not in df.columns:
        logger.warning("No trajectory_id column found")
        return pd.DataFrame()

    event_cols = ["is_hard_brake", "is_sudden_accel", "is_stopped", "is_sharp_turn"]
    event_cols = [col for col in event_cols if col in df.columns]

    summary = df.groupby("trajectory_id")[event_cols].sum()

    # Rename columns for readability
    summary = summary.rename(
        columns={
            "is_hard_brake": "n_hard_brakes",
            "is_sudden_accel": "n_sudden_accels",
            "is_stopped": "n_stop_events",
            "is_sharp_turn": "n_sharp_turns",
        }
    )

    return summary.reset_index()
