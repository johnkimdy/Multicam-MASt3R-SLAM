import torch
from mast3r_slam.frame import SharedStates, SharedKeyframes, Mode, Frame, create_frame


class MultiCameraStates:
    def __init__(self, manager, num_cameras, h, w, dtype=torch.float32, device="cuda"):
        self.states = []
        for _ in range(num_cameras):
            self.states.append(SharedStates(manager, h, w, dtype, device))
        
        self.num_cameras = num_cameras
        self.device = device
        
    def get_state(self, camera_id):
        if camera_id >= self.num_cameras:
            raise ValueError(f"Camera ID {camera_id} out of range (0-{self.num_cameras-1})")
        return self.states[camera_id]
    
    def set_mode_all(self, mode):
        for state in self.states:
            state.set_mode(mode)
    
    def get_modes(self):
        return [state.get_mode() for state in self.states]
    
    def is_any_tracking(self):
        return any(state.get_mode() == Mode.TRACKING for state in self.states)
    
    def are_all_terminated(self):
        return all(state.get_mode() == Mode.TERMINATED for state in self.states)


def create_frame_for_camera(camera_id, i, imgs, T_WCs, img_size=512, device="cuda:0"):
    """Create a frame for a specific camera"""
    img = imgs[camera_id]
    T_WC = T_WCs[camera_id] if T_WCs else None
    frame = create_frame(i, img, T_WC, img_size, device)
    return frame