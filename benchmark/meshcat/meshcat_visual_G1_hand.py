#!/usr/bin/env python3
import time
import socket
import subprocess
import os
import signal
import atexit
import numpy as np

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

import meshcat
import meshcat.transformations as tf
import meshcat.geometry as g


# ---------------------------
# Meshcat connection helpers
# ---------------------------
ZMQ_HOST = "127.0.0.1"
ZMQ_PORT = 6000
ZMQ_URL  = f"tcp://{ZMQ_HOST}:{ZMQ_PORT}"
WEB_URL  = "http://127.0.0.1:7000/static/"
HAND_MARKER_RADIUS = 0.035
FK_MARKER_RADIUS = 0.025
BASE_X_OFFSET = -0.083
BASE_Y_OFFSET = -0.040
BASE_Z_OFFSET = -0.042
RMSE_HORIZON = 300

# BASE_X_OFFSET = 0.0
# BASE_Y_OFFSET = 0.0
# BASE_Z_OFFSET = 0.0

def _port_open(host, port, timeout=0.2):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _terminate_process(proc: subprocess.Popen, timeout=1.0):
    """Terminate a Popen process (and its process group if start_new_session=True)."""
    if proc is None:
        return
    if proc.poll() is not None:
        return  # already dead

    # Try graceful terminate (process group first)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return

    # Wait a bit
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return
        time.sleep(0.02)

    # Hard kill if still alive
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def connect_or_start_meshcat(zmq_url=ZMQ_URL):
    """
    Use external meshcat-server (persistent) if available; otherwise start it.

    Returns
    -------
    viewer : meshcat.Visualizer
    proc   : subprocess.Popen or None
        If we started meshcat-server, proc is the handle; otherwise None.
    """
    proc = None
    if not _port_open(ZMQ_HOST, ZMQ_PORT):
        proc = subprocess.Popen(
            ["meshcat-server", "--zmq-url", zmq_url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # lets us kill the whole group
        )
        for _ in range(200):
            if _port_open(ZMQ_HOST, ZMQ_PORT):
                break
            time.sleep(0.02)

    viewer = meshcat.Visualizer(zmq_url=zmq_url)

    # Ensure we kill only the server we started, on normal exit too
    atexit.register(lambda: _terminate_process(proc))

    return viewer, proc


def print_joint_indices(model: pin.Model):
    print("---- joints ----")
    for jid in range(model.njoints):
        j = model.joints[jid]
        print(
            f"[{jid:02d}] {model.names[jid]:30s}  nq={j.nq} nv={j.nv}  "
            f"idx_q={j.idx_q} idx_v={j.idx_v}"
        )


def load_position_csv(csv_path: str) -> np.ndarray:
    data = np.loadtxt(csv_path, delimiter=",")
    data = np.atleast_2d(data)
    if data.shape[1] < 4:
        raise RuntimeError(f"{csv_path} must have at least 4 columns, got {data.shape[1]}")
    return data[:, 1:4]


def compute_frame_positions(model: pin.Model, q_traj: np.ndarray, frame_name: str) -> np.ndarray:
    frame_id = model.getFrameId(frame_name)
    if frame_id >= model.nframes:
        raise RuntimeError(f"Frame '{frame_name}' not found in model.")

    data = model.createData()
    positions = np.zeros((q_traj.shape[0], 3))
    for i, q in enumerate(q_traj):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        positions[i] = data.oMf[frame_id].translation.copy()
    return positions


def print_z_offset_stats(reference_name: str, target_name: str, z_offset: np.ndarray):
    print(
        f"{target_name}.z - {reference_name}.z: "
        f"mean={np.mean(z_offset):.6f}, min={np.min(z_offset):.6f}, "
        f"max={np.max(z_offset):.6f}, first={z_offset[0]:.6f}"
    )


def print_rmse_stats(label: str, reference_positions: np.ndarray, target_positions: np.ndarray):
    error = target_positions - reference_positions
    rmse_xyz = np.sqrt(np.mean(np.square(error), axis=0))
    rmse_3d = np.sqrt(np.mean(np.sum(np.square(error), axis=1)))
    print(
        f"{label} RMSE: total={rmse_3d:.6f}, "
        f"x={rmse_xyz[0]:.6f}, y={rmse_xyz[1]:.6f}, z={rmse_xyz[2]:.6f}"
    )


def apply_base_xyz_offset(q: np.ndarray, x_offset: float, y_offset: float, z_offset: float) -> np.ndarray:
    q_shifted = np.array(q, copy=True)
    q_shifted[0] += x_offset
    q_shifted[1] += y_offset
    q_shifted[2] += z_offset
    return q_shifted


def apply_position_xyz_offset(position: np.ndarray, x_offset: float, y_offset: float, z_offset: float) -> np.ndarray:
    shifted = np.array(position, copy=True)
    shifted[0] += x_offset
    shifted[1] += y_offset
    shifted[2] += z_offset
    return shifted


def main():
    # ---------------------------
    # 1) Load G1 model from URDF
    # ---------------------------
    urdf_path = "/home/jkang/third_party/PRIME/data_g1_anitescu/unitree_description/urdf/g1/main.urdf"

    # IMPORTANT: this should point to the folder that makes `package://...` meshes resolvable.
    # If your URDF uses `package://unitree_description/...`, include the parent folder that contains `unitree_description/`.
    package_dirs = ["/home/jkang/third_party/PRIME/data_g1_anitescu/"]

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_path,
        package_dirs=package_dirs,
        root_joint=pin.JointModelFreeFlyer()
    )

    print("Model has nq =", model.nq, " nv =", model.nv)
    print_joint_indices(model)

    # ---------------------------
    # 2) Meshcat viewer + scene
    # ---------------------------
    viewer, meshcat_proc = connect_or_start_meshcat()
    print("Meshcat URL (leave this open):", WEB_URL)

    # Clear any old objects on the server
    viewer.delete()

    # Ground plane
    plane_z = -0.04
    plane_size = (20.0, 20.0, 0.01)
    viewer["ground"].set_object(g.Box(plane_size), g.MeshLambertMaterial(opacity=0.4))
    viewer["ground"].set_transform(
        tf.translation_matrix([0.0, 0.0, plane_z - plane_size[2] / 2.0])
    )
    viewer["hand_left"].set_object(
        g.Sphere(HAND_MARKER_RADIUS),
        g.MeshLambertMaterial(color=0x1F77B4, opacity=0.9),
    )
    viewer["hand_right"].set_object(
        g.Sphere(HAND_MARKER_RADIUS),
        g.MeshLambertMaterial(color=0xD62728, opacity=0.9),
    )
    viewer["hand_left_fk"].set_object(
        g.Sphere(FK_MARKER_RADIUS),
        g.MeshLambertMaterial(color=0x17BECF, opacity=0.9),
    )
    viewer["hand_right_fk"].set_object(
        g.Sphere(FK_MARKER_RADIUS),
        g.MeshLambertMaterial(color=0xFF7F0E, opacity=0.9),
    )

    # Two visualizers sharing the same viewer
    viz_fddp = MeshcatVisualizer(model, collision_model, visual_model)
    viz_fddp.initViewer(viewer, open=False)
    viz_fddp.loadViewerModel("g1_fddp")

    viz_sim = MeshcatVisualizer(model, collision_model, visual_model)
    viz_sim.initViewer(viewer, open=False)
    viz_sim.loadViewerModel("g1_sim")

    # Show neutral pose once
    q0 = pin.neutral(model)
    viz_fddp.display(q0)
    viz_sim.display(q0)

    # ---------------------------
    # 3) Load trajectories
    # ---------------------------
    csv_fddp = "/home/jkang/third_party/PRIME/data_g1_anitescu/xs_results_fddp.csv"
    csv_sim = "/home/jkang/third_party/PRIME/data_g1_anitescu/xs_results_fddp.csv"
    # csv_fddp = "/home/jkang/third_party/PRIME/data_g1_anitescu/xs_log.csv"
    # csv_sim  = "/home/jkang/third_party/PRIME/data_g1_anitescu/xs_log.csv"
    csv_hand_left = "/home/jkang/third_party/PRIME/data_g1_anitescu/hls_log.csv"
    csv_hand_right = "/home/jkang/third_party/PRIME/data_g1_anitescu/hrs_log.csv"

    q_fddp = np.loadtxt(csv_fddp, delimiter=",")
    q_sim  = np.loadtxt(csv_sim,  delimiter=",")
    hand_left = load_position_csv(csv_hand_left)
    hand_right = load_position_csv(csv_hand_right)

    q_fddp = np.atleast_2d(q_fddp)
    q_sim  = np.atleast_2d(q_sim)

    # Keep only the first nq columns (Pinocchio expects full configuration)
    if q_fddp.shape[1] < model.nq:
        raise RuntimeError(f"FDDP CSV has {q_fddp.shape[1]} cols but model.nq={model.nq}")
    if q_sim.shape[1] < model.nq:
        raise RuntimeError(f"SIM CSV has {q_sim.shape[1]} cols but model.nq={model.nq}")

    q_fddp = q_fddp[:, :model.nq]
    q_sim  = q_sim[:,  :model.nq]
    left_foot_fk = compute_frame_positions(model, q_sim, "LL_FOOT")
    torso_fk = compute_frame_positions(model, q_sim, "torso_link")
    hand_left_fk = compute_frame_positions(model, q_sim, "left_rubber_hand")
    hand_right_fk = compute_frame_positions(model, q_sim, "right_rubber_hand")
    left_hand_minus_foot_z = hand_left_fk[:, 2] - left_foot_fk[:, 2]
    left_hand_minus_torso_z = hand_left_fk[:, 2] - torso_fk[:, 2]

    print("Loaded FDDP length:", q_fddp.shape[0])
    print("Loaded SIM  length:", q_sim.shape[0])
    print("Loaded left hand positions:", hand_left.shape[0])
    print("Loaded right hand positions:", hand_right.shape[0])
    print("Computed LL_FOOT FK samples:", left_foot_fk.shape[0])
    print("Computed torso_link FK samples:", torso_fk.shape[0])
    print("Computed left_rubber_hand FK samples:", hand_left_fk.shape[0])
    print("Computed right_rubber_hand FK samples:", hand_right_fk.shape[0])
    print(f"Applying base XYZ offset: x={BASE_X_OFFSET}, y={BASE_Y_OFFSET}, z={BASE_Z_OFFSET}")
    print_z_offset_stats("LL_FOOT", "left_rubber_hand", left_hand_minus_foot_z)
    print_z_offset_stats("torso_link", "left_rubber_hand", left_hand_minus_torso_z)

    # ---------------------------
    # 4) Playback (aligned / downsampled)
    # ---------------------------
    start_idx = 0
    n_knots   = min(2000, q_sim.shape[0] - start_idx)
    n_scale   = 1
    end_idx   = start_idx + n_knots

    q_sim_sampled = q_sim[start_idx:end_idx:n_scale]
    hand_left_sampled = hand_left[start_idx:end_idx:n_scale]
    hand_right_sampled = hand_right[start_idx:end_idx:n_scale]
    hand_left_fk_sampled = hand_left_fk[start_idx:end_idx:n_scale]
    hand_right_fk_sampled = hand_right_fk[start_idx:end_idx:n_scale]

    playback_len = min(
        q_fddp.shape[0],
        q_sim_sampled.shape[0],
        hand_left_sampled.shape[0],
        hand_right_sampled.shape[0],
        hand_left_fk_sampled.shape[0],
        hand_right_fk_sampled.shape[0],
    )
    if playback_len == 0:
        raise RuntimeError("No overlapping samples found for robot and hand playback.")

    rmse_horizon = playback_len if RMSE_HORIZON is None else min(RMSE_HORIZON, playback_len)
    hand_left_fk_rmse = np.array(
        [
            apply_position_xyz_offset(p, BASE_X_OFFSET, BASE_Y_OFFSET, BASE_Z_OFFSET)
            for p in hand_left_fk_sampled[:rmse_horizon]
        ]
    )
    hand_right_fk_rmse = np.array(
        [
            apply_position_xyz_offset(p, BASE_X_OFFSET, BASE_Y_OFFSET, BASE_Z_OFFSET)
            for p in hand_right_fk_sampled[:rmse_horizon]
        ]
    )
    print(f"RMSE horizon: {rmse_horizon}")
    print_rmse_stats("left_rubber_hand vs left CSV hand", hand_left_sampled[:rmse_horizon], hand_left_fk_rmse)
    print_rmse_stats("right_rubber_hand vs right CSV hand", hand_right_sampled[:rmse_horizon], hand_right_fk_rmse)

    sleep_dt = 0.01  # playback speed

    try:
        while True:
            for i in range(playback_len):
                q1 = apply_base_xyz_offset(q_fddp[i], BASE_X_OFFSET, BASE_Y_OFFSET, BASE_Z_OFFSET)
                q2 = apply_base_xyz_offset(q_sim_sampled[i], BASE_X_OFFSET, BASE_Y_OFFSET, BASE_Z_OFFSET)
                viz_fddp.display(q1)
                viz_sim.display(q2)
                viewer["hand_left"].set_transform(tf.translation_matrix(hand_left_sampled[i]))
                viewer["hand_right"].set_transform(tf.translation_matrix(hand_right_sampled[i]))
                viewer["hand_left_fk"].set_transform(
                    tf.translation_matrix(
                        apply_position_xyz_offset(
                            hand_left_fk_sampled[i], BASE_X_OFFSET, BASE_Y_OFFSET, BASE_Z_OFFSET
                        )
                    )
                )
                viewer["hand_right_fk"].set_transform(
                    tf.translation_matrix(
                        apply_position_xyz_offset(
                            hand_right_fk_sampled[i], BASE_X_OFFSET, BASE_Y_OFFSET, BASE_Z_OFFSET
                        )
                    )
                )
                time.sleep(sleep_dt)
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nCtrl-C: shutting down...")

    finally:
        # Optional: clear scene
        try:
            viewer.delete()
        except Exception:
            pass

        # Kill only if we started meshcat-server
        if meshcat_proc is not None:
            _terminate_process(meshcat_proc)
            print("meshcat-server terminated.")
        else:
            print("meshcat-server was external; left running.")


if __name__ == "__main__":
    main()
