import cv2
import numpy as np
from mast3r_slam.dataloader import MonocularDataset
from mast3r_slam.mast3r_utils import resize_img

class MultiWebcam:
    def __init__(self, camera_ids=[0, 1]):
        """Initialize multiple webcams for parallel capture
        
        Args:
            camera_ids: List of camera IDs to use (typically 0, 1, etc.)
        """
        self.cameras = []
        for cam_id in camera_ids:
            self.cameras.append(WebcamSingle(cam_id))
        
        self.save_results = False
        self.img_size = 512
        self.use_calibration = False
        self.camera_intrinsics = [None] * len(camera_ids)
        
    def __len__(self):
        return 999999
    
    def __getitem__(self, idx):
        # Return images from all cameras at given index
        images = []
        timestamps = []
        for camera in self.cameras:
            timestamp, img = camera[idx]
            timestamps.append(timestamp)
            images.append(img)
        return timestamps, images
    
    def subsample(self, subsample):
        # No subsampling for webcam feed
        pass
            
    def get_img_shape(self):
        # Return the first camera's shape (they should all be the same)
        return self.cameras[0].get_img_shape()
        
    def has_calib(self):
        return all(camera.has_calib() for camera in self.cameras)
    
    def set_calibration(self, intrinsics_list):
        """Set calibration parameters for each camera
        
        Args:
            intrinsics_list: List of calibration parameters, one for each camera
        """
        if len(intrinsics_list) != len(self.cameras):
            raise ValueError(f"Expected {len(self.cameras)} calibration parameters, got {len(intrinsics_list)}")
        
        for i, intrinsics in enumerate(intrinsics_list):
            self.cameras[i].camera_intrinsics = intrinsics
            self.camera_intrinsics[i] = intrinsics
        
        self.use_calibration = True


class WebcamSingle(MonocularDataset):
    def __init__(self, camera_id=0):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = f"webcam:{camera_id}"
        self.camera_id = camera_id
        self.img_size = 512
        
        # Don't initialize camera here - do it in the process that needs it
        self.cap = None
        self.save_results = False
        self.timestamps = []
    
    def ensure_camera_initialized(self):
        """Ensure camera is initialized in the current process"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open camera {self.camera_id}")
    
    def __len__(self):
        return 999999
    
    def read_img(self, idx):
        self.ensure_camera_initialized()
        ret, img = self.cap.read()
        if not ret:
            raise ValueError(f"Failed to read image from camera {self.camera_id}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        timestamp = idx / 30.0  # Approximate timestamp at 30fps
        self.timestamps.append(timestamp)
        
        return img.astype(np.float32) / 255.0
        
    def close(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
    def get_image(self, idx):
        img = self.read_img(idx)
        if self.use_calibration and self.camera_intrinsics is not None:
            img = self.camera_intrinsics.remap(img)
        return img