#!/usr/bin/env python3
"""
Standalone script to process Geolife GPS trajectory data.
This is more efficient than running in Jupyter notebooks for large datasets.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.geo_utils import (
    calculate_speed_acceleration,
    compute_trip_duration_distance,
    filter_unrealistic_data,
    load_all_trajectories,
    load_plt_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_sample_trajectories(data_dir: Path, n_files: int) -> pd.DataFrame:
    """Load only a sample of trajectory files for faster processing."""
    plt_files = list(data_dir.rglob("*.plt"))

    if n_files >= len(plt_files):
        return load_all_trajectories(data_dir)

    logger.info(f"Loading sample of {n_files} files from {len(plt_files)} total files...")
    sample_files = plt_files[:n_files]

    all_data = []
    for filepath in sample_files:
        try:
            df = load_plt_file(filepath)
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
    logger.info(f"Loaded {len(df_all)} total points from {len(all_data)} files")
    return df_all


def process_geolife_data(
    data_dir: Path,
    output_dir: Path,
    max_speed_kmh: float = 200.0,
    max_accel_ms2: float = 10.0,
    sample_size: int | None = None,
) -> None:
    """
    Process Geolife GPS trajectory data with progress tracking.

    Args:
        data_dir: Directory containing Geolife .plt files
        output_dir: Directory to save processed data
        max_speed_kmh: Maximum reasonable speed threshold
        max_accel_ms2: Maximum reasonable acceleration threshold
        sample_size: Optional number of trajectories to process (for testing)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Geolife GPS Trajectory Data Processing")
    logger.info("=" * 60)

    # Step 1: Load trajectories
    logger.info("Step 1/4: Loading GPS trajectories...")
    if sample_size:
        df = load_sample_trajectories(data_dir, sample_size)
    else:
        df = load_all_trajectories(data_dir)

    logger.info(f"Loaded {len(df):,} GPS points from {df['trajectory_id'].nunique()} trajectories")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Step 2: Calculate speed and acceleration
    logger.info("\nStep 2/4: Calculating speed and acceleration...")
    df = calculate_speed_acceleration(df)

    # Step 3: Filter unrealistic data
    logger.info("\nStep 3/4: Filtering unrealistic data...")
    df = filter_unrealistic_data(df, max_speed_kmh=max_speed_kmh, max_accel_ms2=max_accel_ms2)

    # Step 4: Compute trip statistics
    logger.info("\nStep 4/4: Computing trip duration and distance...")
    df = compute_trip_duration_distance(df)

    # Save results
    logger.info("\nSaving processed data...")
    output_file = output_dir / "trajectories_cleaned.parquet"
    df.to_parquet(output_file, index=False)
    logger.info(f"✅ Saved {len(df):,} points to {output_file}")

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"Total GPS points:     {len(df):,}")
    logger.info(f"Unique users:         {df['user_id'].nunique()}")
    logger.info(f"Unique trajectories:  {df['trajectory_id'].nunique()}")
    logger.info("\nSpeed statistics (km/h):")
    logger.info(f"  Mean:  {df['speed_kmh'].mean():.2f}")
    logger.info(f"  Std:   {df['speed_kmh'].std():.2f}")
    logger.info(f"  Max:   {df['speed_kmh'].max():.2f}")
    logger.info("\nAcceleration statistics (m/s²):")
    logger.info(f"  Mean:  {df['acceleration_ms2'].mean():.2f}")
    logger.info(f"  Std:   {df['acceleration_ms2'].std():.2f}")
    logger.info(
        f"  Range: [{df['acceleration_ms2'].min():.2f}, {df['acceleration_ms2'].max():.2f}]"
    )


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Process Geolife GPS trajectory data")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/geolife/Data"), help="Input data directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed"), help="Output directory"
    )
    parser.add_argument(
        "--max-speed", type=float, default=200.0, help="Maximum speed threshold (km/h)"
    )
    parser.add_argument(
        "--max-accel", type=float, default=10.0, help="Maximum acceleration threshold (m/s²)"
    )
    parser.add_argument(
        "--sample", type=int, help="Optional: process only N trajectory files (for testing)"
    )

    args = parser.parse_args()

    process_geolife_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_speed_kmh=args.max_speed,
        max_accel_ms2=args.max_accel,
        sample_size=args.sample,
    )


if __name__ == "__main__":
    main()
