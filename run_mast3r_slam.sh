#!/bin/bash

# Source conda
source /home/rtx4090/anaconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate mast3r-slam

# Run the Python script with config
python main.py --config config/multicam.yaml