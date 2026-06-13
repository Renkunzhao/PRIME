import time
import socket
import subprocess
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


def _port_open(host, port, timeout=0.2):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def connect_or_start_meshcat(zmq_url=ZMQ_URL):
    """Use external meshcat-server (persistent) if available; otherwise start it."""
    if not _port_open(ZMQ_HOST, ZMQ_PORT):
        subprocess.Popen(
            ["meshcat-server", "--zmq-url", zmq_url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(200):
            if _port_open(ZMQ_HOST, ZMQ_PORT):
                break
            time.sleep(0.02)
    return meshcat.Visualizer(zmq_url=zmq_url)


def print_joint_indices(model: pin.Model):
    print("---- joints ----")
    for jid in range(model.njoints):
        j = model.joints[jid]
        print(f"[{jid:02d}] {model.names[jid]:30s}  nq={j.nq} nv={j.nv}  idx_q={j.idx_q} idx_v={j.idx_v}")


def se3_to_tf(M: pin.SE3):
    """Pinocchio SE3 -> 4x4 homogeneous for meshcat.transformations"""
    T = np.eye(4)
    T[:3, :3] = np.asarray(M.rotation)
    T[:3,  3] = np.asarray(M.translation).reshape(3)
    return T


def ensure_joint_frame_triad(viewer, path, axis_len=0.12, axis_radius=0.004):
    """
    Create a triad under viewer[path] using cylinders.
    IMPORTANT: meshcat.geometry.Cylinder is along +Y by default.
    """
    mat_x = g.MeshLambertMaterial(color=0xFF0000, opacity=1.0)
    mat_y = g.MeshLambertMaterial(color=0x00FF00, opacity=1.0)
    mat_z = g.MeshLambertMaterial(color=0x0000FF, opacity=1.0)

    cyl = g.Cylinder(axis_len, axis_radius)

    viewer[path]["x"].set_object(cyl, mat_x)
    viewer[path]["y"].set_object(cyl, mat_y)
    viewer[path]["z"].set_object(cyl, mat_z)

    # Cylinder axis is +Y. Build transforms so each axis is a "ray" from origin.
    # Rightmost transform applied first.
    # X axis: rotate Y->X via Rz(+90deg), then translate +X by L/2
    Tx = tf.translation_matrix([axis_len / 2.0, 0.0, 0.0]) @ tf.rotation_matrix(+np.pi / 2.0, [0, 0, 1])
    # Y axis: no rotation, translate +Y by L/2
    Ty = tf.translation_matrix([0.0, axis_len / 2.0, 0.0])
    # Z axis: rotate Y->Z via Rx(+90deg), then translate +Z by L/2
    Tz = tf.translation_matrix([0.0, 0.0, axis_len / 2.0]) @ tf.rotation_matrix(+np.pi / 2.0, [1, 0, 0])

    viewer[path]["x"].set_transform(Tx)
    viewer[path]["y"].set_transform(Ty)
    viewer[path]["z"].set_transform(Tz)


def update_joint_frame_triad(model, data, viewer, jid: int, name: str):
    """
    Update triad pose to joint frame placement in world.
    Uses Pinocchio joint placement data.oMi[jid].
    """
    if jid < 0 or jid >= model.njoints:
        raise ValueError(f"jid {jid} out of range [0, {model.njoints-1}]")

    # data.oMi is updated by forwardKinematics(model, data, q)
    T_world_joint = se3_to_tf(data.oMi[jid])
    viewer[name].set_transform(T_world_joint)


# ---------------------------
# 1) Load G1 model from URDF
# ---------------------------
# urdf_path = "/home/jkang/third_party/PRIME/data_g1_anitescu/unitree_description/urdf/g1/main.urdf"
urdf_path = "/home/jkang/third_party/PRIME/data_g1_anitescu/unitree_description/urdf/g1/g1_29dof.urdf"

# IMPORTANT: folder that makes `package://...` meshes resolvable.
package_dirs = ["/home/jkang/third_party/PRIME/data_g1_anitescu/"]

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_path,
    package_dirs=package_dirs,
    root_joint=pin.JointModelFreeFlyer()
)
data = model.createData()

print("Model has nq =", model.nq, " nv =", model.nv)
print_joint_indices(model)


# ---------------------------
# 2) Meshcat viewer + scene
# ---------------------------
viewer = connect_or_start_meshcat()
print("Meshcat URL (leave this open):", WEB_URL)

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
# 2.5) Joint frame triad config
# ---------------------------
# Pick the joint you want to visualize
jid_to_plot = 16  # <-- change this

triad_path = "joint_frame"
ensure_joint_frame_triad(viewer, triad_path, axis_len=0.18, axis_radius=0.005)


# ---------------------------
# 3) Load trajectories
# ---------------------------
csv_fddp = "/home/jkang/third_party/PRIME/data_g1_anitescu/high_kappa/xs_results_fddp.csv"
csv_sim  = "/home/jkang/third_party/PRIME/data_g1_anitescu/high_kappa/xs_log.csv"

q_fddp = np.loadtxt(csv_fddp, delimiter=",")
q_sim  = np.loadtxt(csv_sim,  delimiter=",")

q_fddp = np.atleast_2d(q_fddp)
q_sim  = np.atleast_2d(q_sim)

if q_fddp.shape[1] < model.nq:
    raise RuntimeError(f"FDDP CSV has {q_fddp.shape[1]} cols but model.nq={model.nq}")
if q_sim.shape[1] < model.nq:
    raise RuntimeError(f"SIM CSV has {q_sim.shape[1]} cols but model.nq={model.nq}")

q_fddp = q_fddp[:, :model.nq]
q_sim  = q_sim[:,  :model.nq]

print("Loaded FDDP length:", q_fddp.shape[0])
print("Loaded SIM  length:", q_sim.shape[0])


# ---------------------------
# 4) Playback (aligned / downsampled)
# ---------------------------
start_idx = 0
n_knots   = min(3000, q_sim.shape[0] - start_idx)
n_scale   = 1
end_idx   = start_idx + n_knots

q_sim_sampled = q_sim[start_idx:end_idx:n_scale]

sleep_dt = 0.005

while True:
    for q1, q2 in zip(q_fddp, q_sim_sampled):
        viz_fddp.display(q1)
        viz_sim.display(q2)

        # Update triad using one of the trajectories (pick sim here)
        pin.forwardKinematics(model, data, q2)
        # (optional but fine)
        pin.updateFramePlacements(model, data)
        update_joint_frame_triad(model, data, viewer, jid_to_plot, triad_path)

        time.sleep(sleep_dt)
    time.sleep(1.0)
