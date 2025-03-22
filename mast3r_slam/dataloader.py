import pathlib
import re
import cv2
from natsort import natsorted
import numpy as np
import torch
import pyrealsense2 as rs
import yaml
import av
import time
import traceback
import threading
import os

from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.config import config

HAS_TORCHCODEC = True
try:
    from torchcodec.decoders import VideoDecoder
except Exception as e:
    HAS_TORCHCODEC = False
def initialize_camera(max_index=5):
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Webcam initialized at index {index}")
                return cap
            cap.release()
    raise RuntimeError("No available webcam found")


class MonocularDataset(torch.utils.data.Dataset):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        self.rgb_files = []
        self.timestamps = []
        self.img_size = 512
        self.camera_intrinsics = None
        self.use_calibration = config["use_calib"]
        self.save_results = True

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Call get_image before timestamp for realsense camera
        img = self.get_image(idx)
        timestamp = self.get_timestamp(idx)
        return timestamp, img

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        img = cv2.imread(self.rgb_files[idx])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_image(self, idx):
        img = self.read_img(idx)
        if self.use_calibration:
            img = self.camera_intrinsics.remap(img)
        return img.astype(self.dtype) / 255.0

    def get_img_shape(self):
        img = self.read_img(0)
        raw_img_shape = img.shape
        img = resize_img(img, self.img_size)
        # 3XHxW, HxWx3 -> HxW, HxW
        return img["img"][0].shape[1:], raw_img_shape[:2]

    def subsample(self, subsample):
        self.rgb_files = self.rgb_files[::subsample]
        self.timestamps = self.timestamps[::subsample]

    def has_calib(self):
        return self.camera_intrinsics is not None


class TUMDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        rgb_list = self.dataset_path / "rgb.txt"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=" ", dtype=np.unicode_, skiprows=0)
        self.rgb_files = [self.dataset_path / f for f in tstamp_rgb[:, 1]]
        self.timestamps = tstamp_rgb[:, 0]

        match = re.search(r"freiburg(\d+)", dataset_path)
        idx = int(match.group(1))
        if idx == 1:
            calib = np.array(
                [517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633]
            )
        if idx == 2:
            calib = np.array(
                [520.9, 521.0, 325.1, 249.7, 0.2312, -0.7849, -0.0033, -0.0001, 0.9172]
            )
        if idx == 3:
            calib = np.array([535.4, 539.2, 320.1, 247.6])
        W, H = 640, 480
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calib)


class EurocDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        # For Euroc dataset, the distortion is too much to handle for MASt3R.
        # So we always undistort the images, but the calibration will not be used for any later optimization unless specified.
        self.use_calibration = True
        self.dataset_path = pathlib.Path(dataset_path)
        rgb_list = self.dataset_path / "mav0/cam0/data.csv"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=",", dtype=np.unicode_, skiprows=0)
        self.rgb_files = [
            self.dataset_path / "mav0/cam0/data" / f for f in tstamp_rgb[:, 1]
        ]
        self.timestamps = tstamp_rgb[:, 0]
        with open(self.dataset_path / "mav0/cam0/sensor.yaml") as f:
            self.cam0 = yaml.load(f, Loader=yaml.FullLoader)
        W, H = self.cam0["resolution"]
        intrinsics = self.cam0["intrinsics"]
        distortion = np.array(self.cam0["distortion_coefficients"])
        self.camera_intrinsics = Intrinsics.from_calib(
            self.img_size, W, H, [*intrinsics, *distortion], always_undistort=True
        )

    def read_img(self, idx):
        img = cv2.imread(self.rgb_files[idx], cv2.IMREAD_GRAYSCALE)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class ETH3DDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        rgb_list = self.dataset_path / "rgb.txt"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=" ", dtype=np.unicode_, skiprows=0)
        self.rgb_files = [self.dataset_path / f for f in tstamp_rgb[:, 1]]
        self.timestamps = tstamp_rgb[:, 0]
        calibration = np.loadtxt(
            self.dataset_path / "calibration.txt",
            delimiter=" ",
            dtype=np.float32,
            skiprows=0,
        )
        _, (H, W) = self.get_img_shape()
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calibration)


class SevenScenesDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        self.rgb_files = natsorted(
            list((self.dataset_path / "seq-01").glob("*.color.png"))
        )
        self.timestamps = np.arange(0, len(self.rgb_files)).astype(self.dtype)
        fx, fy, cx, cy = 585.0, 585.0, 320.0, 240.0
        self.camera_intrinsics = Intrinsics.from_calib(
            self.img_size, 640, 480, [fx, fy, cx, cy]
        )


class RealsenseDataset(MonocularDataset):
    def __init__(self):
        super().__init__()
        self.dataset_path = None
        self.pipeline = rs.pipeline()
        # self.h, self.w = 720, 1280
        self.h, self.w = 480, 640
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color, self.w, self.h, rs.format.bgr8, 30
        )
        self.profile = self.pipeline.start(self.rs_config)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        # self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        # self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        # self.rgb_sensor.set_option(rs.option.exposure, 200)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.save_results = False

        if self.use_calibration:
            rgb_intrinsics = self.rgb_profile.get_intrinsics()
            self.camera_intrinsics = Intrinsics.from_calib(
                self.img_size,
                self.w,
                self.h,
                [
                    rgb_intrinsics.fx,
                    rgb_intrinsics.fy,
                    rgb_intrinsics.ppx,
                    rgb_intrinsics.ppy,
                ],
            )

    def __len__(self):
        return 999999

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        frameset = self.pipeline.wait_for_frames()
        timestamp = frameset.get_timestamp()
        timestamp /= 1000
        self.timestamps.append(timestamp)

        rgb_frame = frameset.get_color_frame()
        img = np.asanyarray(rgb_frame.get_data())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(self.dtype)
        return img


class Webcam(MonocularDataset):
    def __init__(self):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = None
        # load webcam using opencv
        self.cap = initialize_camera() # 동적으로 인덱스 탐색
        self.save_results = False

    def __len__(self):
        return 999999

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        ret, img = self.cap.read()
        if not ret:
            raise ValueError("Failed to read image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.timestamps.append(idx / 30)

        return img
    
class iPhone(MonocularDataset):
    def __init__(self, stream_url = "http://143.248.162.187:4747/video"):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = None
        # load webcam using opencv
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(self.stream_url) 
        self.save_results = False

    def __len__(self):
        return 999999

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        ret, img = self.cap.read()
        if not ret:
            raise ValueError("Failed to read image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.timestamps.append(idx / 30)

        return img

class Android(MonocularDataset):
    def __init__(self, stream_url = "http://143.248.162.187:4747/video"):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = None
        # load webcam using opencv
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(self.stream_url) 
        self.save_results = False

    def __len__(self):
        return 999999

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        ret, img = self.cap.read()
        if not ret:
            raise ValueError("Failed to read image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.timestamps.append(idx / 30)

        return img


class MP4Dataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = pathlib.Path(dataset_path)
        if HAS_TORCHCODEC:
            self.decoder = VideoDecoder(str(self.dataset_path))
            self.fps = self.decoder.metadata.average_fps
            self.total_frames = self.decoder.metadata.num_frames
        else:
            print("torchcodec is not installed. This may slow down the dataloader")
            self.cap = cv2.VideoCapture(str(self.dataset_path))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.stride = config["dataset"]["subsample"]

    def __len__(self):
        return self.total_frames // self.stride

    def read_img(self, idx):
        if HAS_TORCHCODEC:
            img = self.decoder[idx * self.stride]  # c,h,w
            img = img.permute(1, 2, 0)
            img = img.numpy()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx * self.stride)
            ret, img = self.cap.read()
            if not ret:
                raise ValueError("Failed to read image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(self.dtype)
        timestamp = idx / self.fps
        self.timestamps.append(timestamp)
        return img


class RGBFiles(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = pathlib.Path(dataset_path)
        self.rgb_files = natsorted(list((self.dataset_path).glob("*.png")))
        self.timestamps = np.arange(0, len(self.rgb_files)).astype(self.dtype) / 30.0


class Intrinsics:
    def __init__(self, img_size, W, H, K_orig, K, distortion, mapx, mapy):
        self.img_size = img_size
        self.W, self.H = W, H
        self.K_orig = K_orig
        self.K = K
        self.distortion = distortion
        self.mapx = mapx
        self.mapy = mapy
        _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
            np.zeros((H, W, 3)), self.img_size, return_transformation=True
        )
        self.K_frame = self.K.copy()
        self.K_frame[0, 0] = self.K[0, 0] / scale_w
        self.K_frame[1, 1] = self.K[1, 1] / scale_h
        self.K_frame[0, 2] = self.K[0, 2] / scale_w - half_crop_w
        self.K_frame[1, 2] = self.K[1, 2] / scale_h - half_crop_h

    def remap(self, img):
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)

    @staticmethod
    def from_calib(img_size, W, H, calib, always_undistort=False):
        if not config["use_calib"] and not always_undistort:
            return None
        fx, fy, cx, cy = calib[:4]
        distortion = np.zeros(4)
        if len(calib) > 4:
            distortion = np.array(calib[4:])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        K_opt = K.copy()
        mapx, mapy = None, None
        center = config["dataset"]["center_principle_point"]
        K_opt, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion, (W, H), 0, (W, H), centerPrincipalPoint=center
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, distortion, None, K_opt, (W, H), cv2.CV_32FC1
        )

        return Intrinsics(img_size, W, H, K, K_opt, distortion, mapx, mapy)


class RTMPStreamDataset(MonocularDataset):
    def __init__(self, rtmp_url):
        super().__init__()
        self.rtmp_url = rtmp_url
        self.dataset_path = pathlib.Path("rtmp_stream")
        self.use_calibration = config["use_calib"]
        self.save_results = True
        self.timestamps = []
        self.cap = None
        self.last_frame = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.fps = 30.0  # Default FPS
        
        # Get configuration from rtmp section
        rtmp_config = config.get("rtmp", {})
        self.crop_factor = rtmp_config.get("crop_factor", 0.7)
        self.buffer_size = rtmp_config.get("buffer_size", 1)
        self.frame_skip = rtmp_config.get("frame_skip", 1)
        self.use_threading = rtmp_config.get("use_threading", True)
        
        # For threaded capture
        self.frame_ready = False
        self.current_frame = None
        self.capture_thread = None
        self.stop_thread = False
        self.frame_lock = threading.Lock()
        
        print(f"RTMP Stream Config: crop={self.crop_factor}, buffer={self.buffer_size}, skip={self.frame_skip}, threading={self.use_threading}")
        
        # Try to connect to the stream
        self.connect_to_stream()
        
        # Initialize with a valid frame
        if not self.get_test_frame():
            print("WARNING: Could not get initial frame, creating dummy frame")
            self.last_frame = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Start capture thread if enabled
        if self.use_threading:
            self.start_capture_thread()
    
    def start_capture_thread(self):
        """Start background thread for frame capture"""
        self.stop_thread = False
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()
        
    def capture_frames(self):
        """Background thread function to capture frames"""
        consecutive_errors = 0
        max_errors = 5
        
        while not self.stop_thread:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if not self.connect_to_stream():
                        time.sleep(0.5)
                        consecutive_errors += 1
                        if consecutive_errors >= max_errors:
                            print("Too many consecutive errors, stopping capture thread")
                            break
                        continue
                
                ret, frame = self.cap.read()
                if ret:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Apply center crop
                    img = self.apply_center_crop(img)
                    
                    # Update the current frame
                    with self.frame_lock:
                        self.current_frame = img.copy()  # Create a copy to avoid reference issues
                        self.frame_ready = True
                    
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        print("Failed to read frame multiple times, reconnecting...")
                        self.connect_to_stream()
                        consecutive_errors = 0
                
                # Sleep a small amount to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in capture thread: {e}")
                consecutive_errors += 1
                time.sleep(0.5)
    
    def apply_center_crop(self, img):
        """Crop the center portion of the image based on crop_factor"""
        if img is None or self.crop_factor >= 1.0:
            return img
            
        h, w = img.shape[:2]
        
        # Calculate new dimensions
        new_h = int(h * self.crop_factor)
        new_w = int(w * self.crop_factor)
        
        # Calculate starting coordinates for the crop
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        
        # Crop the image
        cropped_img = img[start_y:start_y+new_h, start_x:start_x+new_w]
        
        return cropped_img
    
    def connect_to_stream(self):
        """Connect to the RTMP stream using OpenCV"""
        try:
            # Close existing capture if any
            if self.cap is not None:
                self.cap.release()
                
            # Open new connection with optimized settings for RTMP
            self.cap = cv2.VideoCapture(self.rtmp_url)
            
            # Set buffer size to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Additional optimizations for network streams
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;0|fflags;nobuffer|fflags;flush_packets"
            
            # Get stream properties
            if self.cap.isOpened():
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0 or self.fps > 100:  # Unrealistic FPS values
                    self.fps = 30.0  # Fallback to default
                print(f"Connected to RTMP stream with FPS: {self.fps}")
                return True
            else:
                print(f"Failed to open RTMP stream: {self.rtmp_url}")
                return False
        except Exception as e:
            print(f"Error connecting to stream: {e}")
            return False
            
    def get_test_frame(self):
        """Get a test frame to initialize the system"""
        try:
            if self.cap is None or not self.cap.isOpened():
                if not self.connect_to_stream():
                    return False
                    
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.last_frame_time = time.time()
                return True
            return False
        except Exception as e:
            print(f"Error getting test frame: {e}")
            return False
    
    def __del__(self):
        """Cleanup resources"""
        self.stop_thread = True
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            
    def __len__(self):
        return 1000000  # Just return a large number for streaming
        
    def get_timestamp(self, idx):
        if idx < len(self.timestamps):
            return self.timestamps[idx]
        else:
            timestamp = self.frame_count / self.fps
            return timestamp
    
    def read_img(self, idx):
        """Override read_img to follow the same pattern as other classes, adding center crop"""
        # Skip frames if system is falling behind (if idx % frame_skip != 0)
        if self.frame_skip > 1 and idx % self.frame_skip != 0 and self.last_frame is not None:
            return self.last_frame
            
        try:
            # If threading is enabled, get frame from the thread
            if self.use_threading:
                with self.frame_lock:
                    if self.frame_ready:
                        img = self.current_frame.copy()
                        self.last_frame = img
                        self.last_frame_time = time.time()
                        timestamp = self.frame_count / self.fps
                        self.timestamps.append(timestamp)
                        self.frame_count += 1
                        return img
            
            # Fallback to direct capture if threading is disabled or no frame is ready
            if self.cap is None or not self.cap.isOpened():
                if not self.connect_to_stream():
                    if self.last_frame is not None:
                        return self.last_frame
                    else:
                        return np.zeros((512, 512, 3), dtype=np.uint8)
            
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply center crop
                img = self.apply_center_crop(img)
                
                self.last_frame = img
                self.last_frame_time = time.time()
                timestamp = self.frame_count / self.fps
                self.timestamps.append(timestamp)
                self.frame_count += 1
                return img
            else:
                # Try to reconnect if read failed
                print("Failed to read frame, reconnecting...")
                self.connect_to_stream()
                if self.last_frame is not None:
                    return self.last_frame
                else:
                    return np.zeros((512, 512, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Error in read_img: {e}")
            if self.last_frame is not None:
                return self.last_frame
            else:
                return np.zeros((512, 512, 3), dtype=np.uint8)
            
    def get_img_shape(self):
        """Get the shape of the images"""
        if self.last_frame is None:
            dummy_frame = np.zeros((512, 512, 3), dtype=np.uint8)
            img = resize_img(dummy_frame, self.img_size)
            return img["img"][0].shape[1:], (512, 512)
            
        raw_img_shape = self.last_frame.shape[:2]
        img = resize_img(self.last_frame, self.img_size)
        return img["img"][0].shape[1:], raw_img_shape


def load_dataset(dataset_path):
    # Handle RTMP URLs first
    if isinstance(dataset_path, str) and (
        dataset_path.startswith("rtmp://") or 
        dataset_path.startswith("rtsp://") or
        dataset_path.startswith("http://") and dataset_path.endswith(".m3u8")
    ):
        try:
            return RTMPStreamDataset(dataset_path)
        except Exception as e:
            print(f"Error creating RTMP dataset: {e}")
            print("Falling back to dummy RTMP dataset")
            # Create a dummy dataset that will return black frames
            return RTMPStreamDataset("dummy://stream")
        
    # Handle other dataset types
    split_dataset_type = str(dataset_path).split("/")
    if "tum" in split_dataset_type:
        return TUMDataset(dataset_path)
    if "euroc" in split_dataset_type:
        return EurocDataset(dataset_path)
    if "eth3d" in split_dataset_type:
        return ETH3DDataset(dataset_path)
    if "7-scenes" in split_dataset_type:
        return SevenScenesDataset(dataset_path)
    if "realsense" in split_dataset_type:
        return RealsenseDataset()
    if "webcam" in split_dataset_type:
        return Webcam()
    if "iphone" in split_dataset_type:
        stream_url = config["stream_ip"]["iphone_ip"]
        return iPhone(stream_url)
    if "android" in split_dataset_type:
        stream_url = config["stream_ip"]["android_ip"]
        return Android(stream_url)
    if "rtmp" in split_dataset_type:
        stream_url = config["stream_ip"]["rtmp_ip"]
        key = config["stream_ip"]["key"]
        return RTMPStreamDataset(f"{stream_url}/{key}")
    ext = split_dataset_type[-1].split(".")[-1]
    if ext in ["mp4", "avi", "MOV", "mov"]:
        return MP4Dataset(dataset_path)
    return RGBFiles(dataset_path)
