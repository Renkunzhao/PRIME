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
BASE_X_OFFSET = -0.043
BASE_Y_OFFSET = 0.010
BASE_Z_OFFSET = -0.022

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


def apply_base_xyz_offset(q: np.ndarray, x_offset: float, y_offset: float, z_offset: float) -> np.ndarray:
    q_shifted = np.array(q, copy=True)
    q_shifted[0] += x_offset
    q_shifted[1] += y_offset
    q_shifted[2] += z_offset
    return q_shifted


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
    # csv_fddp = "/home/jkang/third_party/PRIME/data_g1_anitescu/xs_log.csv"
    csv_sim  = "/home/jkang/third_party/PRIME/data_g1_anitescu/xs_log.csv"

    q_fddp = np.loadtxt(csv_fddp, delimiter=",")
    q_sim  = np.loadtxt(csv_sim,  delimiter=",")

    q_fddp = np.atleast_2d(q_fddp)
    q_sim  = np.atleast_2d(q_sim)

    # Keep only the first nq columns (Pinocchio expects full configuration)
    if q_fddp.shape[1] < model.nq:
        raise RuntimeError(f"FDDP CSV has {q_fddp.shape[1]} cols but model.nq={model.nq}")
    if q_sim.shape[1] < model.nq:
        raise RuntimeError(f"SIM CSV has {q_sim.shape[1]} cols but model.nq={model.nq}")

    q_fddp = q_fddp[:, :model.nq]
    q_sim  = q_sim[:,  :model.nq]

    print("Loaded FDDP length:", q_fddp.shape[0])
    print("Loaded SIM  length:", q_sim.shape[0])
    print(f"Applying base XYZ offset: x={BASE_X_OFFSET}, y={BASE_Y_OFFSET}, z={BASE_Z_OFFSET}")

    # ---------------------------
    # 4) Playback (aligned / downsampled)
    # ---------------------------
    start_idx = 0
    n_knots   = min(3000, q_sim.shape[0] - start_idx)
    n_scale   = 1
    end_idx   = start_idx + n_knots

    q_sim_sampled = q_sim[start_idx:end_idx:n_scale]

    sleep_dt = 0.01  # playback speed

    try:
        while True:
            for q1, q2 in zip(q_fddp, q_sim_sampled):
                viz_fddp.display(apply_base_xyz_offset(q1, BASE_X_OFFSET, BASE_Y_OFFSET, BASE_Z_OFFSET))
                viz_sim.display(apply_base_xyz_offset(q2, BASE_X_OFFSET, BASE_Y_OFFSET, BASE_Z_OFFSET))
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
