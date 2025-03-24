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
from mast3r_slam.global_opt import FactorGraph

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
        # Handle relocalization for each camera
        # for camera_id in range(config["multicamera"]["cameras"]):
        #     camera_mode = states.get_mode(camera_id)
        #     if camera_mode == Mode.RELOC:
        #         frame = states.get_frame(camera_id)
        #         success = relocalization(frame, keyframes, factor_graph, retrieval_database)
        #         if success:
        #             states.set_mode(Mode.TRACKING, camera_id)
        #         states.dequeue_reloc()
        #         continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
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



if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/multicam.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")
    parser.add_argument("--debug", default="False")

    
    
    args = parser.parse_args()
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
        dataset_identifier = ds_conf.get('path', ds_conf.get('id'))
        dataset_paths.append(dataset_identifier)
        camera_ids.append(ds_conf['camera_id'])

    # Use the camera_id of the first dataset as the reference (or set it differently if needed)
    reference_camera_id = multicam_config['datasets'][0]['camera_id']

    # Now call load_multi_dataset() with these arguments:
    datasets = load_multi_dataset(dataset_paths, camera_ids, reference_camera_id)

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    datasets.reference.subsample(config["dataset"]["subsample"])  # Use the reference dataset property
    h, w = datasets.reference.get_img_shape()[0]  # Retrieve image shape from the reference dataset

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    model = load_mast3r(device=device)
    model.share_memory()

    K = None


    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    i = 0
    fps_timer = time.time()
    frames = []

    while True:
        # The following variables are shared memory:
        # - keyframes: SharedKeyframes object that stores keyframe data
        # - states: SharedStates object that manages shared states and synchronization
        
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
            else states.get_frame().T_WC
        )
        reference_frame = create_frame(i, refrenceimg, referenceT_WC, img_size=datasets.reference.img_size, device=device)
        reference_camera_id = datasets.reference.camera_id

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, reference_frame)
            reference_frame.update_pointmap(X_init, C_init)
            keyframes.append(reference_frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(reference_frame)
            i += 1
            continue

            
        # camera_mode = states.get_mode(reference_camera_id)
        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(reference_frame) # frame.update_pointmap()
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(reference_frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, reference_frame)
            reference_frame.update_pointmap(X, C)
            states.set_frame(reference_frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception(f"Invalid mode for camera")

        if add_new_kf:
            keyframes.append(reference_frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish - written by GPT-40
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1


    print("done")
    backend.join()
    if not args.no_viz:
        viz.join()
