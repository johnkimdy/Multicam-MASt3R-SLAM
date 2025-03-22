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
    MonocularDataset, TUMDataset, EurocDataset, ETH3DDataset, 
    SevenScenesDataset, RealsenseDataset, Webcam,
    MP4Dataset, RGBFiles, load_dataset as load_single_dataset
)
HAS_TORCHCODEC = True
try:
    from torchcodec.decoders import VideoDecoder
except Exception as e:
    HAS_TORCHCODEC = False


def parse_webcam_spec(spec):
    """Parse webcam specification string into camera IDs"""
    if ":" in spec:
        return int(spec.split(":")[-1])
    return 0  # Default camera ID

class WebcamDataset:
    """A picklable webcam dataset class that initializes camera in the process that needs it"""
    def __init__(self, camera_ids):
        """Initialize webcam dataset
        
        Args:
            camera_ids: List of camera IDs to use
        """
        self.camera_ids = camera_ids if isinstance(camera_ids, list) else [camera_ids]
        self.caps = [None] * len(self.camera_ids)  # Initialize cameras later
        self.img_size = 512
        self.use_calibration = False
        self.camera_intrinsics = [None] * len(self.camera_ids)
        self.save_results = False
        self.timestamps = []
        
    def ensure_cameras_initialized(self):
        """Initialize cameras if not already done"""
        for i, camera_id in enumerate(self.camera_ids):
            if self.caps[i] is None:
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open camera {camera_id}")
                self.caps[i] = cap
                
    def __len__(self):
        return 999999  # Continuous stream
        
    def __getitem__(self, idx):
        self.ensure_cameras_initialized()
        
        timestamps = []
        images = []
        
        for i, cap in enumerate(self.caps):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read from camera {self.camera_ids[i]}")
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            
            # Approximate timestamp
            timestamp = idx / 30.0  # Assuming 30fps
            
            timestamps.append(timestamp)
            images.append(frame)
            
        return timestamps, images
        
    def get_img_shape(self):
        return [(self.img_size, self.img_size)]
        
    def has_calib(self):
        return False
        
    def close(self):
        """Clean up camera resources"""
        for cap in self.caps:
            if cap is not None:
                cap.release()
        self.caps = [None] * len(self.camera_ids)


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