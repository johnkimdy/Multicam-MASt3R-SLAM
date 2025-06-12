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
import traceback
import numpy as np
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import load_multi_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization_multi import WindowMsg, run_visualization
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

def relocalization(frame, keyframes, factor_graph, retrieval_database):
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
        camera_id = frame.camera_ID
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ): # Relocalization Successful!
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

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

    mode = states.get_mode()
    num_cams = states.get_num_cams()
    reference_camera_id = states.get_reference_camera_id()
    print(f"[Backend] Number of cameras: {num_cams}")
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
        
        for i in range(num_cams):
            if i == reference_camera_id:
                continue
            cammode = states.get_camera_mode(i)
            if cammode == Mode.RELOC:
                frame = states.get_frame(i)
                success = relocalization(frame, keyframes, factor_graph, retrieval_database)
                if success:
                    states.set_camera_mode(Mode.TRACKING, i)
                states.dequeue_reloc()
                continue

        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction # LOOK AT ME!!!!
        kf_idx = []

        frame = keyframes[idx] # frame.cam_id
        # print("keyframe cam_id: ", frame.camera_ID)
        target_cam_id = frame.camera_ID

        '''
        Graph construction priority is to add edges to the previous keyframe of the same camera ID, 
        then the GN solver will optimize the pose with the rest of the graph regardless of camera ID.
        '''
        # k to previous consecutive keyframes
        n_consec = 1 #TODO it was 1
        # Find the previous n_consec keyframes with the same camera ID to add it to kf_idx first
        for j in range(idx - 1, -1, -1):
            if keyframes.cam_id[j] == target_cam_id:
                # If the camera ID of the keyframe matches the target camera ID, add it to kf_idx
                 kf_idx.append(j)
            if len(kf_idx) >= n_consec:
                break
        if len(kf_idx) == 0:
            print("No previous keyframes found for camera ID", target_cam_id)

        retrieval_inds = retrieval_database.update( # LOOK AT ME!!!!
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
        print("Global optimization for kf ", idx, " with ", len(kf_idx), " kfs: ", kf_idx)
        if kf_idx:
            factor_graph.add_factors( #TODO johnk
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


def test_per_camera_images(multi_dataset, num_frames=10):
    """
    Check a subset of frames from each camera (up to num_frames) using OpenCV
    to verify that the image data is valid.

    Args:
        multi_dataset: A MultiDataset instance.
        num_frames (int): The number of frames to check per camera.
    """
    import numpy as np

    datasets_by_camera = multi_dataset.datasets_by_camera
    for cam_id, ds in datasets_by_camera.items():
        print(f"\n[Camera ID: {cam_id}]")
        total_frames = len(ds)
        frames_to_check = min(num_frames, total_frames)

        for idx in range(frames_to_check):
            timestamp, image = ds[idx]
            if image is None:
                print(f"  Frame {idx} at {timestamp}: No image data!")
            else:
                print(f"  Frame {idx} at {timestamp}: Image shape: {image.shape}")

                # Check whether the entire image is made up of zeros
                if np.all(image == 0):
                    print("    -> This image is entirely zero pixels (black or transparent).")
                else:
                    print("    -> This image contains non-zero pixels.")

                # Optional: Display the image briefly (uncomment if needed)
                # import cv2
                # cv2.imshow(f"Camera {cam_id}", image)
                # cv2.waitKey(1)

    # (Optional) Close any OpenCV windows after the loop
    # cv2.destroyAllWindows()

def run_backend_init_nonrefcam(cfg, model, states, keyframes, K):
    """
    이 backend는 non-reference 카메라의 초기화(INIT 모드)를 담당합니다.
    reference 카메라는 메인 루프에서 tracking을 수행하므로, 여기서는 제외합니다.
    """
    set_global_config(cfg)
    device = keyframes.device
    retrieval_database = load_retriever(model)
    factor_graph = FactorGraph(model, keyframes, K, device)
    
    # 주기적으로 non-reference 카메라의 상태를 체크
    while states.get_mode() is not Mode.TERMINATED:
        # states.camera_modes는 각 카메라의 현재 모드를 저장합니다.
        # reference 카메라는 여기서 제외해야 합니다.
        for cam_id in list(states.camera_modes.keys()):
            # reference 카메라는 이미 메인 루프에서 처리되므로 건너뜁니다.
            if cam_id == states.reference_camera_id:
                continue
            
            if states.get_camera_mode(cam_id) == Mode.NONREF_INIT_RELOC:
                try:
                    # non-reference 카메라의 최신 프레임을 가져옵니다.
                    frame = states.get_frame(cam_id)
                    if frame is None:
                        continue
                    states.set_frame(frame, cam_id)
                    states.set_camera_mode(Mode.NONREF_INIT_COMPLETE, cam_id)




                    print(f"[Backend Non-Ref] Camera {cam_id} 초기화 성공!")
                except Exception as e:
                    print(f"[Backend Non-Ref] Camera {cam_id} 초기화 중 에러: {e}")


        time.sleep(0.01)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/replicamultiagent.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")
    parser.add_argument("--debug", default="False")
    parser.add_argument("--save-results", default="False", help="Save results to disk")
    
    
    
    args = parser.parse_args()

        
    save_results = args.save_results =="True" or args.save_results.lower() == "true"
    debug = args.debug == "True" or args.debug.lower() == "true"
    load_config(args.config)
    print(args.dataset)
    print(config)
    multicam_config = config.get("multidataset", {})
    
    # Extract dataset identifiers (paths) and camera IDs
    dataset_paths = []
    camera_ids = []
    for ds_conf in multicam_config['datasets']:
        # Use the 'path' if provided; otherwise, use the 'id'
        if 'path' in ds_conf:
            if ds_conf.get('id') == 'MultiAgentJPG':
                dataset_identifier = ds_conf.get('path', ds_conf.get('id'))
            else:
                dataset_identifier = ds_conf.get('path') #FIXME
        # if 'path' in ds_conf and ds_conf.get('id') == 'MultiAgentJPG':
        #     dataset_identifier = ds_conf.get('path', ds_conf.get('id'))
        else:
            dataset_identifier = ds_conf.get('id')
        print("dataset_identifier: ", dataset_identifier)
        dataset_paths.append(dataset_identifier)
        camera_ids.append(ds_conf['camera_id'])

    # Use the camera_id of the first dataset as the reference (or set it differently if needed)
    reference_camera_id = multicam_config['datasets'][0]['camera_id']

    # Now call load_multi_dataset() with these arguments:
    print("Dataset Paths: ", f"{dataset_paths}")
    datasets = load_multi_dataset(dataset_paths, camera_ids, reference_camera_id)
    # 각 카메라별로 이미지 데이터 점검 (예: 각 카메라에서 첫 10프레임 확인)
    test_per_camera_images(datasets, num_frames=10)


    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    datasets.reference.subsample(config["dataset"]["subsample"])  # Use the reference dataset property
    h, w = datasets.reference.get_img_shape()[0]  # Retrieve image shape from the reference dataset

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w, num_cams=len(multicam_config['datasets']),reference_camera_id=datasets.reference_camera_id)

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        print("Starting visualizer...")
        viz.start()

    model = load_mast3r(device=device)
    model.share_memory()

    K = None

    dataset = datasets.reference
    if save_results:
        # Create a directory to save results
        save_dir, seq_name = eval.prepare_savedir(args, dataset)  
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()  
    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()
    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    i = 0
    fps_timer = time.time()
    frames = []
    # Print the length of the datasets
    print(f"Total number of frames in the reference dataset: {len(datasets)}")
    
    # Print length of each individual dataset
    for cam_id, dataset in datasets.datasets_by_camera.items():
        print(f"Camera {cam_id} dataset length: {len(dataset)}")
    
    # Print reference camera ID for confirmation
    print(f"Reference camera ID: {datasets.reference_camera_id}")

    # 새로운 backend: non-reference 카메라 초기화 전용
    backend_init_nonrefcam = mp.Process(
        target=run_backend_init_nonrefcam, args=(config, model, states, keyframes, K)
    )
    backend_init_nonrefcam.start()

    while True:
        # The following variables are shared memory:
        # - keyframes: SharedKeyframes object that stores keyframe data
        # - states: SharedStates object that manages shared states and synchronization

        # print(f"Current frame index: {i}")
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(datasets):
            states.set_mode(Mode.TERMINATED)
            break

        _, refrenceimg = datasets.reference[i]

        if save_frames:
            frames.append(refrenceimg)

        # get frames last camera pose
        referenceT_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame(reference_camera_id).T_WC
        )
        reference_frame = create_frame(i, refrenceimg, referenceT_WC, img_size=datasets.reference.img_size, device=device)
        reference_camera_id = datasets.reference.camera_id

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, reference_frame)
            reference_frame.update_pointmap(X_init, C_init)
            keyframes.append(reference_frame)
            # Print camera IDs in a more informative way
            print(f"Camera IDs in keyframes: {keyframes.cam_id[:len(keyframes)]}")
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(reference_frame,reference_camera_id)

            i += 1
            continue

            
        # camera_mode = states.get_mode(reference_camera_id)
        if mode == Mode.TRACKING:
            # print(f"(New Key frame Tracking Befor Tracking) Camera IDs in keyframes: {keyframes.cam_id[:len(keyframes)]}")
            add_new_kf, match_info, try_reloc = tracker.track(reference_frame) # frame.update_pointmap()
            # print(f"(New Key frame Tracking) Camera IDs in keyframes: {keyframes.cam_id[:len(keyframes)]}")

            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(reference_frame, reference_camera_id)
            if add_new_kf:
                keyframes.append(reference_frame)
                states.queue_global_optimization(len(keyframes) - 1)
                # In single threaded mode, wait for the backend to finish - written by GPT-40
                while config["single_thread"]:
                    with states.lock:
                        if len(states.global_optimizer_tasks) == 0:
                            break
                    time.sleep(0.01)

        elif mode == Mode.RELOC:
            # Process reference camera relocalization
            X, C = mast3r_inference_mono(model, reference_frame)
            reference_frame.update_pointmap(X, C)
            states.set_frame(reference_frame, reference_camera_id)
            states.queue_reloc()

        else:
            raise Exception(f"Invalid mode for camera")   
        
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
               
        # Process other cameras in tracking mode
        for cam_id, ds in datasets.datasets_by_camera.items():
            if cam_id == reference_camera_id:
                continue  # Skip reference camera as it's already processed
            
            # Print camera ID and check against number of cameras

            # Check if this camera is in tracking mode
            cam_mode = states.get_camera_mode(cam_id)
                                # First create a frame for this camera
            cam_T_WC = lietorch.Sim3.Identity(1, device=device)
            # Get the image size for this camera


            # Create frame for this camera
            _, cam_img = ds[i]
            cam_frame = create_frame(
                i,
                cam_img,
                cam_T_WC,
                img_size=ds.img_size,
                device=device,
                cameraID=cam_id
            )

            if cam_mode == Mode.INIT:
                try:
                    # Get the frame for this camera
                   
                    
                    # Check if we have at least one keyframe from the reference camera

                    keyframe,_ = keyframes.last_keyframe(camid=reference_camera_id)# The first image

                    X_init, C_init = mast3r_inference_mono(model, cam_frame)
                    cam_frame.update_pointmap(X_init, C_init)
                    # Now match the camera frame with the reference keyframe
                    idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
                        model, cam_frame, keyframe
                    )
                    use_calib = config["use_calib"]
                    Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

                    # Get rid of batch dim
                    
                    idx_f2k = idx_f2k[0]
                    valid_match_k = valid_match_k[0]
    
                    Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = tracker.get_points_poses(cam_frame, keyframe, idx_f2k, ds.img_size, use_calib, K=None)

                    # Use canonical confidence average
                    
                    # Fix tensor shapes based on debug output
                    # From debug: valid_match_k shape: torch.Size([147456, 1])
                    valid_match_k = valid_match_k.squeeze(-1)  # Convert from [147456, 1] to [147456]
                    
                    # Ensure confidence tensors are properly shaped
                    valid_Cf = Cf > config["tracking"]["C_conf"]
                    valid_Ck = Ck > config["tracking"]["C_conf"]
                    valid_Q = Qk > config["tracking"]["Q_conf"]
                    
                    # Squeeze all tensors to ensure they're 1D for boolean operations
                    valid_Cf = valid_Cf.squeeze(-1)
                    valid_Ck = valid_Ck.squeeze(-1)
                    valid_Q = valid_Q.squeeze(-1)
                    
                    # Combine all validity criteria
                    valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
                    
                    # Calculate match fraction for debugging
                    match_frac = valid_opt.sum() / valid_opt.numel()
                    # Looking at the debug output, valid_opt might need reshaping
                    # [Debug] valid_opt shape: torch.Size([1, 147456])
                    # [Debug] valid_opt sum: 30660/147456 (20.79%)
                    
                    # Ensure valid_opt has the right shape for the optimization function
                    # Based on the tracker.opt_pose_ray_dist_sim3 function, valid_opt should be properly shaped
                    if valid_opt.dim() > 1:
                        # Reshape valid_opt to match expected shape
                        valid_opt = valid_opt.view(-1, 1)  # Reshape to ensure it's [n_points, 1]
                    elif valid_opt.dim() == 1:
                        # If it's already 1D, add the second dimension
                        valid_opt = valid_opt.view(-1, 1)  # Reshape from [n_points] to [n_points, 1]
                    

                    match_frac = valid_opt.sum() / valid_opt.numel() # if match % is low, no match, try reloc
                    if match_frac < config["tracking"]["min_match_frac"]:
                        if len(keyframes) > 20:
                            # Set the camera mode to RELOC since we have keyframes to relocalize against
                            states.set_camera_mode(Mode.RELOC, cam_id)
                            print(f"[Init] Camera {cam_id} set to RELOC mode")
                        else:
                            print(f"[Init] Camera {cam_id} waiting for reference keyframes")
                    # # Track
                    if not use_calib:
                        Qk = Qk[0] # Remove batch dimension
                        
                        try:
                            T_WCf, T_CkCf = tracker.opt_pose_ray_dist_sim3(
                                Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                            )
                            cam_frame.T_WC = T_WCf
                            cam_frame.update_pointmap(Xff, Cff)

                            # Store the frame in states
                            states.set_frame(cam_frame, cam_id)
                            states.set_camera_mode(Mode.TRACKING,cam_id)


                            keyframes.append(cam_frame)
                            print(f"Camera IDs in keyframes: {keyframes.cam_id[:len(keyframes)]}")
                        except Exception as e:
                            print(f"Error in opt_pose_ray_dist_sim3: {e}")

                    # Update camera frame pointmap

                    

                except Exception as e:
                    traceback_str = traceback.format_exc()
                    print(f"[Init Error] Failed to initialize camera {cam_id}: {e}")
                    print(f"[Init Error] Traceback:\n{traceback_str}")
                
                continue


            if cam_mode == Mode.TRACKING:
                try:
                    _, cam_img = ds[i]
                    
                    # Track this camera
                    cam_add_new_kf, cam_match_info, cam_try_reloc = tracker.track(cam_frame)
                    # if cam_try_reloc:
                    #     states.set_camera_mode(Mode.RELOC, cam_id)
                    states.set_frame(cam_frame, cam_id)
                    
                    # If this camera needs a new keyframe, add it
                    if cam_add_new_kf:
                        keyframes.append(cam_frame)
                        print(f"(Tracking non ref)Camera IDs in keyframes: {keyframes.cam_id[:len(keyframes)] + 1}")
                        states.queue_global_optimization(len(keyframes) - 1)
                except Exception as e:
                    #print(f"[Tracking Error] Failed to track camera {cam_id}: {e}")
                    traceback_str = traceback.format_exc()
                    #print(f"[Tracking Error] Traceback:\n{traceback_str}")
        # Process other cameras in relocalization mode
            # Check if this camera is in relocalization mode
            if cam_mode == Mode.RELOC:
                try:
                    # Get the frame for this camera
                    cam_frame = states.get_frame(cam_id)
                    if cam_frame is not None:
                        # Perform relocalization for this camera
                        cam_X, cam_C = mast3r_inference_mono(model, cam_frame)
                        cam_frame.update_pointmap(cam_X, cam_C)
                        states.set_frame(cam_frame, cam_id)
                        states.queue_reloc()
                        print(f"[Relocalization] Processed camera {cam_id + 1}")
                except Exception as e:
                    print(f"[Relocalization Error] Failed to relocalize camera {cam_id + 1}: {e}")
            
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)
        i += 1# log time

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        # eval.save_reconstruction(
        #     save_dir,
        #     f"{seq_name}.ply",
        #     keyframes,
        #     last_msg.C_conf_threshold,
        # )
        # eval.save_keyframes(
        #     save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        # )

    if save_results and not dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        # eval.save_reconstruction(
        #     save_dir,
        #     f"{seq_name}.ply",
        #     keyframes,
        #     last_msg.C_conf_threshold,
        # )
        # eval.save_keyframes(
        #     save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        # )
    print("done")
    backend.join()
    backend_init_nonrefcam.join()
    if not args.no_viz:
        viz.join()
