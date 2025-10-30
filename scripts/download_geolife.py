#!/usr/bin/env python3
"""
Download and extract the Geolife GPS trajectory dataset.
"""

import subprocess
import zipfile
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kaggle dataset URL
GEOLIFE_URL = "https://www.kaggle.com/api/v1/datasets/download/arashnic/microsoft-geolife-gps-trajectory-dataset"
DATA_DIR = Path(__file__).parent.parent / "data"


def download_file_curl(url: str, output_path: Path) -> None:
    """Download a file from URL using curl."""
    logger.info(f"Downloading Geolife dataset from Kaggle...")
    logger.info(f"This may take several minutes (dataset is ~200MB)...")
    
    try:
        # Use curl to download
        result = subprocess.run(
            ["curl", "-L", "-o", str(output_path), url],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Download complete: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file."""
    logger.info(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info("Extraction complete")


def main():
    """Main function to download and extract Geolife dataset."""
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download zip file
    zip_path = DATA_DIR / "microsoft-geolife-gps-trajectory-dataset.zip"
    geolife_dir = DATA_DIR / "geolife"

    # Check if data already exists
    if geolife_dir.exists() and any(geolife_dir.glob("**/*.plt")):
        logger.info(f"Geolife data already exists at {geolife_dir}")
        logger.info(f"Found {len(list(geolife_dir.glob('**/*.plt')))} .plt files")
        return

    # Download if zip doesn't exist
    if not zip_path.exists():
        logger.info("Downloading dataset...")
        download_file_curl(GEOLIFE_URL, zip_path)
    else:
        logger.info(f"Zip file already exists: {zip_path}")

    # Extract
    logger.info("Extracting dataset...")
    extract_zip(zip_path, DATA_DIR)

    # Move to expected location if needed
    # The Kaggle dataset might extract to different names
    possible_names = [
        DATA_DIR / "Geolife Trajectories 1.3",
        DATA_DIR / "geolife-dataset",
        DATA_DIR / "microsoft-geolife-gps-trajectory-dataset",
    ]
    
    extracted = None
    for name in possible_names:
        if name.exists():
            extracted = name
            break
    
    if extracted and extracted != geolife_dir:
        logger.info(f"Moving extracted data from {extracted.name} to geolife/")
        if geolife_dir.exists():
            shutil.rmtree(geolife_dir)
        extracted.rename(geolife_dir)
    
    # Check if extraction was successful
    plt_files = list(geolife_dir.glob("**/*.plt"))
    if plt_files:
        logger.info(f"✅ Geolife dataset ready at {geolife_dir}")
        logger.info(f"   Found {len(plt_files)} .plt files")
    else:
        logger.warning("⚠️  Extraction may have failed - no .plt files found")
        logger.warning("   Please manually extract the dataset")


if __name__ == "__main__":
    main()

