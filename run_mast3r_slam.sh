#!/bin/bash

# Initialize micromamba
eval "$(micromamba shell hook --shell bash)"

# Activate conda environment
micromamba activate multicam-mast3r-slam

# Run the Python script with config
python main.py --config config/multicam.yaml

# Keep terminal open
read -p 'Press Enter to close...'