import pathlib
import re
import cv2
from natsort import natsorted
import numpy as np
import torch
import pyrealsense2 as rs
import yaml

from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.config import config
from mast3r_slam.dataloader import (
    TUMDataset, EurocDataset, ETH3DDataset, 
    SevenScenesDataset, RealsenseDataset, Webcam,
    MP4Dataset, RGBFiles, load_dataset as load_single_dataset
)
HAS_TORCHCODEC = True
try:
    from torchcodec.decoders import VideoDecoder
except Exception as e:
    HAS_TORCHCODEC = False


class MultiCameraDataset:
    def __init__(self, dataset_paths):
        self.datasets = []
        for path in dataset_paths:
            self.datasets.append(load_single_dataset(path))
        
        self.save_results = True
        self.img_size = self.datasets[0].img_size
        self.use_calibration = all(dataset.use_calibration for dataset in self.datasets)
        self.camera_intrinsics = [d.camera_intrinsics for d in self.datasets]
        
    def __len__(self):
        return min(len(dataset) for dataset in self.datasets)
    
    def __getitem__(self, idx):
        # Return images from all cameras at given index
        images = []
        timestamps = []
        for dataset in self.datasets:
            if idx < len(dataset):
                timestamp, img = dataset[idx]
                timestamps.append(timestamp)
                images.append(img)
        return timestamps, images
    
    def subsample(self, subsample):
        for dataset in self.datasets:
            dataset.subsample(subsample)
            
    def get_img_shape(self):
        shapes = []
        for dataset in self.datasets:
            shapes.append(dataset.get_img_shape())
        return shapes[0]  # Return the first one for compatibility
        
    def has_calib(self):
        return all(dataset.has_calib() for dataset in self.datasets)


def load_multi_dataset(dataset_paths):
    # Special case for webcams
    if all("webcam" in path for path in dataset_paths):
        try:
            from mast3r_slam.webcam_multi import MultiWebcam
            # Extract camera IDs from paths like "webcam:0", "webcam:1"
            camera_ids = []
            for path in dataset_paths:
                if ":" in path:
                    cam_id = int(path.split(":")[-1])
                    camera_ids.append(cam_id)
                else:
                    # Default to sequential IDs if not specified
                    camera_ids = list(range(len(dataset_paths)))
                    break
            return MultiWebcam(camera_ids)
        except Exception as e:
            print(f"Error initializing MultiWebcam: {e}")
            print("Falling back to standard dataset loading")
    
    # Standard dataset loading
    return MultiCameraDataset(dataset_paths)