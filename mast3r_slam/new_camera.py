import cv2
import numpy as np
import pathlib
from dataloader import MonocularDataset

class NewCameraDataset(MonocularDataset):
    def __init__(self):
        super().__init__()
        self.pipeline = cv2.VideoCapture(0)  # Initialize the new camera (e.g., webcam)
        self.save_results = False

    def __len__(self):
        return 999999  # Or implement a proper length based on your needs

    def get_timestamp(self, idx):
        return self.timestamps[idx]  # Implement timestamp logic as needed

    def read_img(self, idx):
        ret, img = self.pipeline.read()
        if not ret:
            raise ValueError("Failed to read image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.timestamps.append(idx / 30)  # Update timestamp logic
        return img 