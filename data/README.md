# Data Directory

This directory contains the Geolife GPS trajectory dataset.

## Structure

```
data/
├── geolife/          # Raw .plt files from Geolife dataset
└── processed/        # Cleaned data and features
```

## Downloading the Dataset

To download the Geolife dataset, run:

```bash
make data
```

Or manually:

```bash
python scripts/download_geolife.py
```

This will download the dataset from Microsoft Research and extract it to `geolife/`.

## Dataset Information

The Geolife dataset contains GPS trajectories from 182 users over a period of over five years (April 2007 to August 2012). The dataset has a total distance of 1,292,951 kilometers and a total duration of 50,176 hours.

For more information, see: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/

## License

The Geolife dataset is provided by Microsoft Research. Please refer to their terms of use.

