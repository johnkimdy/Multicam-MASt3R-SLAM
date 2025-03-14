import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
import torch.multiprocessing as mp

from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
from mast3r_slam.multicam_dataloader import load_multi_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.multicam_frame import MultiCameraStates, create_frame_for_camera
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization


def relocalization(frame, keyframes, factor_graph, retrieval_database, camera_id=0):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print(f"CAMERA {camera_id} RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print(f"Success! Camera {camera_id} Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print(f"Camera {camera_id} Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    # claude suggestion 
    # mode = states.get_mode()
    if config["multicamera"]["enabled"]:
        while not states.are_all_terminated():
            time.sleep(0.01)
            
            # Handle relocalization for any camera
            for camera_id, state in enumerate(states.states):
                if state.get_mode() == Mode.RELOC:
                    frame = state.get_frame()
                    success = relocalization(frame, keyframes, factor_graph, retrieval_database, camera_id)
                    if success:
                        state.set_mode(Mode.TRACKING)
                    state.dequeue_reloc()
                
            # Process any optimization tasks
            idx = -1
            for state in states.states:
                with state.lock:
                    if len(state.global_optimizer_tasks) > 0:
                        idx = state.global_optimizer_tasks[0]
                        break
                        
                if idx == -1:
                    continue

            # Graph Construction
            kf_idx = []
            # k to previous consecutive keyframes
            n_consec = 1
            for j in range(min(n_consec, idx)):
                kf_idx.append(idx - 1 - j)
            frame = keyframes[idx]
            retrieval_inds = retrieval_database.update(
                frame,
                add_after_query=True,
                k=config["retrieval"]["k"],
                min_thresh=config["retrieval"]["min_thresh"],
            )
            kf_idx += retrieval_inds

            lc_inds = set(retrieval_inds)
            lc_inds.discard(idx - 1)
            if len(lc_inds) > 0:
                print("Database retrieval", idx, ": ", lc_inds)

            kf_idx = set(kf_idx)  # Remove duplicates by using set
            kf_idx.discard(idx)  # Remove current kf idx if included
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [idx] * len(kf_idx)
            if kf_idx:
                factor_graph.add_factors(
                    kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
                )

            # Update edge information for all states for visualization
            for state in states.states:
                with state.lock:
                    state.edges_ii[:] = factor_graph.ii.cpu().tolist()
                    state.edges_jj[:] = factor_graph.jj.cpu().tolist()

            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()

            # Remove completed task
            for state in states.states:
                with state.lock:
                    if len(state.global_optimizer_tasks) > 0 and state.global_optimizer_tasks[0] == idx:
                        state.global_optimizer_tasks.pop(0)
                        break
    
    # # original
    else:
        mode = states.get_mode()
        while mode is not Mode.TERMINATED:
            mode = states.get_mode()
            if mode == Mode.INIT or states.is_paused():
                time.sleep(0.01)
                continue
            if mode == Mode.RELOC:
                frame = states.get_frame()
                success = relocalization(frame, keyframes, factor_graph, retrieval_database)
                if success:
                    states.set_mode(Mode.TRACKING)
                states.dequeue_reloc()
                continue
            idx = -1
            with states.lock:
                if len(states.global_optimizer_tasks) > 0:
                    idx = states.global_optimizer_tasks[0]
            if idx == -1:
                time.sleep(0.01)
                continue
                
            print("hello")
            # Graph Construction
            kf_idx = []
            # k to previous consecutive keyframes
            n_consec = 1
            for j in range(min(n_consec, idx)):
                kf_idx.append(idx - 1 - j)
            frame = keyframes[idx]
            retrieval_inds = retrieval_database.update(
                frame,
                add_after_query=True,
                k=config["retrieval"]["k"],
                min_thresh=config["retrieval"]["min_thresh"],
            )
            kf_idx += retrieval_inds

            lc_inds = set(retrieval_inds)
            lc_inds.discard(idx - 1)
            if len(lc_inds) > 0:
                print("Database retrieval", idx, ": ", lc_inds)

            kf_idx = set(kf_idx)  # Remove duplicates by using set
            kf_idx.discard(idx)  # Remove current kf idx if included
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [idx] * len(kf_idx)
            if kf_idx:
                factor_graph.add_factors(
                    kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
                )

            with states.lock:
                states.edges_ii[:] = factor_graph.ii.cpu().tolist()
                states.edges_jj[:] = factor_graph.jj.cpu().tolist()

            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()

            with states.lock:
                if len(states.global_optimizer_tasks) > 0:
                    idx = states.global_optimizer_tasks.pop(0)
        


def run_camera_process(camera_id, dataset, config, model, states, keyframes, K, main2viz, viz2main):
    """Run the tracking process for a single camera"""
    state = states.get_state(camera_id)
    tracker = FrameTracker(model, keyframes, device=keyframes.device)
    
    i = 0
    fps_timer = time.time()
    
    print(f"Camera {camera_id} process started")
    
    while True:
        # Handle termination
        if state.get_mode() == Mode.TERMINATED:
            print(f"Camera {camera_id} process terminated")
            break
            
        # Handle pause
        if state.is_paused():
            time.sleep(0.01)
            continue
            
        # Handle viewer messages
        msg = try_get_msg(viz2main)
        if msg is not None and msg.is_terminated:
            state.set_mode(Mode.TERMINATED)
            break
        
        # End of dataset
        if i >= len(dataset):
            state.set_mode(Mode.TERMINATED)
            break
            
        # Get next frame
        timestamps, imgs = dataset[i]
        img = imgs[camera_id]
        timestamp = timestamps[camera_id]
        
        # Get camera pose
        if i == 0 and camera_id == 0:
            # First camera initializes the world
            T_WC = lietorch.Sim3.Identity(1, device=keyframes.device)
        elif i == 0 and not config["multicamera"]["initialize_all"]:
            # Other cameras start in RELOC mode to find their pose relative to camera 0
            T_WC = lietorch.Sim3.Identity(1, device=keyframes.device)
            state.set_mode(Mode.RELOC)
        else:
            # Otherwise use the last camera pose
            T_WC = state.get_frame().T_WC
            
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=keyframes.device)
        
        mode = state.get_mode()
        if mode == Mode.INIT:
            # Initialize via mono inference
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            state.queue_global_optimization(len(keyframes) - 1)
            state.set_mode(Mode.TRACKING)
            state.set_frame(frame)
            
        elif mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                state.set_mode(Mode.RELOC)
            state.set_frame(frame)
            
            if add_new_kf:
                print(f"Camera {camera_id}: Adding new keyframe at frame {i}")
                keyframes.append(frame)
                state.queue_global_optimization(len(keyframes) - 1)
                
        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            state.set_frame(frame)
            state.queue_reloc()
            
        # Log time occasionally
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"Camera {camera_id} FPS: {FPS}")
            
        i += 1


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["datasets/tum/rgbd_dataset_freiburg1_desk"])
    parser.add_argument("--config", default="config/multicam.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")

    args = parser.parse_args()

    load_config(args.config)
    print("Datasets:", args.datasets)
    print(config)

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    # Load datasets
    if len(args.datasets) == 1 and not config["multicamera"]["enabled"]:
        print(args.datasets[0].split(":")[0])
        dataset = load_dataset(args.datasets[0])
        dataset.subsample(config["dataset"]["subsample"])
        h, w = dataset.get_img_shape()[0]
        num_cameras = 1
    else:
        # Multiple cameras
        config["multicamera"]["enabled"] = True
        config["multicamera"]["cameras"] = len(args.datasets)
        dataset = load_multi_dataset(args.datasets)
        dataset.subsample(config["dataset"]["subsample"])
        h, w = dataset.get_img_shape()[0]
        num_cameras = len(args.datasets)

    # Handle calibration
    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        if isinstance(dataset.camera_intrinsics, list):
            # Multi-camera case
            for cam_id, intrinsic in enumerate(dataset.camera_intrinsics):
                dataset.camera_intrinsics[cam_id] = Intrinsics.from_calib(
                    dataset.img_size,
                    intrinsics["width"],
                    intrinsics["height"],
                    intrinsics["calibration"],
                )
        else:
            # Single camera case
            dataset.camera_intrinsics = Intrinsics.from_calib(
                dataset.img_size,
                intrinsics["width"],
                intrinsics["height"],
                intrinsics["calibration"],
            )

    # Shared memory structures
    keyframes = SharedKeyframes(manager, h, w)
    
    if config["multicamera"]["enabled"]:
        states = MultiCameraStates(manager, num_cameras, h, w)
    else:
        # For backward compatibility
        states = SharedStates(manager, h, w)

    # Visualization process
    if not args.no_viz:
        # For now, visualization still uses only the first camera's state
        first_state = states.get_state(0) if config["multicamera"]["enabled"] else states
        viz = mp.Process(
            target=run_visualization,
            args=(config, first_state, keyframes, main2viz, viz2main),
        )
        viz.start()

    # Load model
    model = load_mast3r(device=device)
    model.share_memory()

    # Get calibration
    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]
    
    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
        
    K = None
    if use_calib:
        if isinstance(dataset.camera_intrinsics, list):
            # For now, just use the first camera's intrinsics
            K = torch.from_numpy(dataset.camera_intrinsics[0].K_frame).to(
                device, dtype=torch.float32
            )
        else:
            K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
                device, dtype=torch.float32
            )
        keyframes.set_intrinsics(K)

    
    # Run backend process
    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    print("wow")
    # Run camera processes
    camera_processes = []
    for camera_id in range(num_cameras):
        if config["multicamera"]["enabled"]:
            print("disabled?")
            cam_process = mp.Process(
                target=run_camera_process,
                args=(camera_id, dataset, config, model, states, keyframes, K, main2viz, viz2main)
            )
            cam_process.start()
            camera_processes.append(cam_process)
        else:
            # Run the original main loop for backward compatibility
            # from main import run_backend as original_run_backend
            print("debug1")
            run_backend(config, model, states, keyframes, K)
            break

    print("wow1")
    # Wait for processes to finish
    if config["multicamera"]["enabled"]:
        for p in camera_processes:
            p.join()
    
    backend.join()
    
    print("wow2")
    if not args.no_viz:
        viz.join()
        
    print("All processes completed")