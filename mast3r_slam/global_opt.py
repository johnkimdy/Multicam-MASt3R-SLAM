import lietorch
import torch
import numpy as np
from mast3r_slam.config import config
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.geometry import (
    constrain_points_to_ray,
)
from mast3r_slam.mast3r_utils import mast3r_match_symmetric
import mast3r_slam_backends


def compute_dynamic_match_threshold(kf_i_idx, kf_j_idx, keyframes, base_threshold=0.1):
    """
    Dynamically adjust matching threshold based on camera ID and pose distance
    
    Args:
        kf_i_idx, kf_j_idx: Keyframe indices
        keyframes: SharedKeyframes object
        base_threshold: Base min_match_frac for same camera
    
    Returns:
        Adaptive threshold value
    """

    # Same camera - use base threshold
    if keyframes.cam_id[kf_i_idx] == keyframes.cam_id[kf_j_idx]:
        return base_threshold
    
    # Different cameras - compute pose-dependent threshold
    T_i_data = keyframes.T_WC[kf_i_idx]  # lietorch.Sim3
    T_j_data = keyframes.T_WC[kf_j_idx]

    # Handle different possible tensor shapes
    if T_i_data.dim() == 1:
        T_i_data = T_i_data.unsqueeze(0)  # Add batch dimension
    if T_j_data.dim() == 1:
        T_j_data = T_j_data.unsqueeze(0)
        
    # Create lietorch.Sim3 objects from the tensor data
    T_i = lietorch.Sim3(T_i_data)
    T_j = lietorch.Sim3(T_j_data)
    
    # Relative transformation T_ij = T_i^-1 * T_j
    T_rel = T_i.inv() * T_j
    
    # Extract pose components
    translation = T_rel.translation()  # [3]
    # rotation_log = T_rel.rotation().log()  # [3] 
    # scale = T_rel.scale()  # [1]
    scale_i, scale_j = float(T_i_data.view(-1)[0].item()), float(T_j_data.view(-1)[0].item())

    # For rotation, try different approaches
    try:
        # Method 1: Check if quaternion() method exists
        quat = T_rel.vec().squeeze()[1:5]  # [x, y, z, w] or [w, x, y, z]
        # Normalize quaternion first
        quat_norm = quat / torch.norm(quat)
        # For unit quaternion, rotation angle θ = 2 * arccos(|qw|)
        qw = quat_norm[-1]  # Assuming [qx, qy, qz, qw] format
        rot_angle = 2 * torch.acos(torch.clamp(torch.abs(qw), 0, 1)).item()
        
    except AttributeError:
        try:
            # Method 2: Extract quaternion from raw data
            # Sim3 data format: [scale, qx, qy, qz, qw, tx, ty, tz]
            T_rel_data = T_rel.data.squeeze()
            quat = T_rel_data[1:5]  # [qx, qy, qz, qw]
            qw = quat[3]  # Real part
            rot_angle = 2 * torch.acos(torch.clamp(torch.abs(qw), 0, 1)).item()
            
        except:
            # Method 3: Fallback - compute from original poses
            T_i_data_flat = T_i_data.squeeze()
            T_j_data_flat = T_j_data.squeeze()
            
            # Extract quaternions from original poses
            qi = T_i_data_flat[1:5]  # [qx, qy, qz, qw]
            qj = T_j_data_flat[1:5]
            
            # Compute relative rotation using quaternion dot product
            dot_product = torch.abs(torch.dot(qi, qj))
            rot_angle = 2 * torch.acos(torch.clamp(dot_product, 0, 1)).item()
    
    # Compute distance metrics
    trans_dist = torch.norm(translation).item()
    # rot_angle = torch.norm(rotation_log).item()  # Radians
    if scale_i > 0 and scale_j > 0:
        scale_diff = abs(torch.log(scale_j / scale_i).item())
    else:
        scale_diff = 0.0

    # scale_diff = abs(torch.log(scale).item())
    
    # Normalized distance (0-1 range)
    # You may need to tune these normalization factors based on your scene scale
    trans_norm = min(trans_dist / 5.0, 1.0)  # Normalize by 5m baseline
    rot_norm = min(rot_angle / (np.pi/2), 1.0)  # Normalize by 90 degrees  
    scale_norm = min(scale_diff / 0.5, 1.0)  # Normalize by 50% scale change
    
    # Combined pose distance [0-1]
    w_t, w_r, w_s = 0.2, 0.6, 0.2  # Weights for translation, rotation, scale
    pose_distance = w_t * trans_norm + w_r * rot_norm + w_s * scale_norm
    
    # Adaptive threshold formula
    inter_camera_penalty = 0.05  # Base penalty for different cameras
    distance_penalty = 0.07 * pose_distance  # Additional penalty based on distance
    
    adaptive_threshold = base_threshold + inter_camera_penalty + distance_penalty
    
    # Clamp to reasonable range
    print("Adaptive Threshold: ", min(adaptive_threshold, 0.2))
    return min(adaptive_threshold, 0.13)  # Max threshold of 60%

class FactorGraph:
    def __init__(self, model, frames: SharedKeyframes, K=None, device="cuda"):
        self.model = model
        self.frames = frames
        self.device = device
        self.cfg = config["local_opt"]
        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_ii2jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_jj2ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.valid_match_j = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.valid_match_i = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.Q_ii2jj = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.Q_jj2ii = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.window_size = self.cfg["window_size"]

        self.K = K

    

    def add_factors(self, ii, jj, min_match_frac, is_reloc=False):
        kf_ii = [self.frames[idx] for idx in ii]
        kf_jj = [self.frames[idx] for idx in jj]
        feat_i = torch.cat([kf_i.feat for kf_i in kf_ii])
        feat_j = torch.cat([kf_j.feat for kf_j in kf_jj])
        pos_i = torch.cat([kf_i.pos for kf_i in kf_ii])
        pos_j = torch.cat([kf_j.pos for kf_j in kf_jj])
        shape_i = [kf_i.img_true_shape for kf_i in kf_ii]
        shape_j = [kf_j.img_true_shape for kf_j in kf_jj]

        (
            idx_i2j,
            idx_j2i,
            valid_match_j,
            valid_match_i,
            Qii,
            Qjj,
            Qji,
            Qij,
        ) = mast3r_match_symmetric(
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )

        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni

        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)

        # Dynamic threshold calculation
        adaptive_thresholds = []
        for i_idx, j_idx in zip(ii_tensor, jj_tensor):
            # thresh = compute_dynamic_match_threshold(
            #     i_idx.item(), j_idx.item(), self.frames, min_match_frac
            # )

            # Just read the data, don't do any lietorch operations
            T_i_data = self.frames.T_WC[i_idx.item()]
            T_j_data = self.frames.T_WC[j_idx.item()]

            # Simple heuristic without lietorch
            if self.frames.cam_id[i_idx.item()] == self.frames.cam_id[j_idx.item()]:
                thresh = min_match_frac
            else:
                thresh = min_match_frac + 0.03  # Just add penalty for different cameras
            adaptive_thresholds.append(thresh)

        adaptive_thresholds = torch.tensor(adaptive_thresholds, device=self.device) #FIXME: adaptive min_match_frac seems to be very sensitive. 1E-2 sensitive.

        # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < adaptive_thresholds # min_match_frac
        # invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges

        if invalid_edges.any() and is_reloc:
            return False

        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])

        added_new_edges = valid_edges.sum() > 0
        return added_new_edges

    def get_unique_kf_idx(self):
        return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)

    def prep_two_way_edges(self):
        ii = torch.cat((self.ii, self.jj), dim=0)
        jj = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def get_poses_points(self, unique_kf_idx):
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))

        Cs = torch.stack([kf.get_average_conf() for kf in kfs])

        return Xs, T_WCs, Cs

    def solve_GN_rays(self):
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        max_iter = self.cfg["max_iters"]
        sigma_ray = self.cfg["sigma_ray"]
        sigma_dist = self.cfg["sigma_dist"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]
        mast3r_slam_backends.gauss_newton_rays(
            pose_data,
            Xs,
            Cs,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])

    def solve_GN_calib(self):
        K = self.K
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        # Constrain points to ray
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]

        img_size = self.frames[0].img.shape[-2:]
        height, width = img_size

        mast3r_slam_backends.gauss_newton_calib(
            pose_data,
            Xs,
            Cs,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])
