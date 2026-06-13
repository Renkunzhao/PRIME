#!/usr/bin/env python3
import time
import socket
import subprocess
import numpy as np

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

import meshcat
import meshcat.transformations as tf
import meshcat.geometry as g


# ============================================================
# Meshcat connection helpers
# ============================================================
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


# ============================================================
# Scene helpers
# ============================================================
def _set_prop_safe(node, key, value):
    try:
        node.set_property(key, value)
        return True
    except Exception:
        return False


def _delete_path_safe(viewer, path):
    try:
        viewer[path].delete()
        return True
    except Exception:
        return False


def setup_mjlab_studio(viewer, z0=-0.04, floor_size=30.0, grid=1.0):
    """
    IMPORTANT: Do NOT call viewer.delete() here.
    That can remove/reset /Cameras and controls in some meshcat builds,
    which often causes the "zoom stuck at fixed distance" behavior.

    Instead, delete only our groups.
    """
    # Clear only our stuff, keep cameras intact
    _delete_path_safe(viewer, "ground")
    _delete_path_safe(viewer, "g1_fddp")
    _delete_path_safe(viewer, "g1_sim")
    _delete_path_safe(viewer, "g1_fddp/sim_joint_points")

    # bright studio background (best-effort; harmless if unsupported)
    _set_prop_safe(viewer["/Background"], "visible", True)
    _set_prop_safe(viewer["/Background"], "top_color",    [0.98, 0.98, 0.99])
    _set_prop_safe(viewer["/Background"], "bottom_color", [0.98, 0.98, 0.99])
    _set_prop_safe(viewer["/Background"], "color",        [0.98, 0.98, 0.99])

    # optional fog
    _set_prop_safe(viewer["/Fog"], "visible", True)
    _set_prop_safe(viewer["/Fog"], "color", [0.98, 0.98, 0.99])
    _set_prop_safe(viewer["/Fog"], "near",  3.0)
    _set_prop_safe(viewer["/Fog"], "far",   40.0)

    # floor
    thickness = 0.02
    floor_mat = g.MeshPhongMaterial(color=0xF2F3F5, shininess=30, opacity=1.0, transparent=False)
    viewer["ground/plane"].set_object(g.Box([floor_size, floor_size, thickness]), floor_mat)
    viewer["ground/plane"].set_transform(tf.translation_matrix([0.0, 0.0, z0 - thickness / 2.0]))

    # subtle grid
    line_thick = 0.002
    grid_mat = g.MeshPhongMaterial(color=0xE1E4EA, shininess=5)
    n = int(floor_size / grid)
    for i in range(-n, n + 1):
        x = i * grid
        viewer[f"ground/grid/x_{i}"].set_object(g.Box([line_thick, floor_size, line_thick]), grid_mat)
        viewer[f"ground/grid/x_{i}"].set_transform(tf.translation_matrix([x, 0.0, z0 + 1e-4]))
        y = i * grid
        viewer[f"ground/grid/y_{i}"].set_object(g.Box([floor_size, line_thick, line_thick]), grid_mat)
        viewer[f"ground/grid/y_{i}"].set_transform(tf.translation_matrix([0.0, y, z0 + 1e-4]))


def configure_controls_bruteforce(viewer,
                                 min_dist=1e-4,
                                 max_dist=1e6,
                                 zoom_speed=2.5,
                                 near=1e-4,
                                 far=1e6):
    """
    No introspection (older meshcat-python). Just brute-force the common camera/control nodes.
    Also set BOTH OrbitControls-style and TrackballControls-style fields.
    """
    # Common camera nodes
    cam_paths = [
        "/Cameras/default",
        "/Cameras/default/rotated",
        "/Cameras/default/rotated/camera",
        "/Cameras/Default",
        "/Cameras/Default/rotated",
        "/Cameras/Default/rotated/camera",
        "/Cameras/camera",
    ]
    for cp in cam_paths:
        _set_prop_safe(viewer[cp], "near", float(near))
        _set_prop_safe(viewer[cp], "far",  float(far))

    # Common control nodes (OrbitControls, sometimes under rotated/)
    control_paths = [
        "/Cameras/default/controls",
        "/Cameras/default/Controls",
        "/Cameras/default/rotated/controls",
        "/Cameras/default/rotated/Controls",
        "/Cameras/default/rotated/camera/controls",
        "/Cameras/default/rotated/camera/Controls",
        "/Cameras/Default/controls",
        "/Cameras/Default/Controls",
        "/Cameras/Default/rotated/controls",
        "/Cameras/Default/rotated/Controls",
        "/Cameras/Default/rotated/camera/controls",
        "/Cameras/Default/rotated/camera/Controls",
    ]

    # Properties we try (covers orbit + some trackball variants)
    prop_sets = [
        ("enableZoom", True),
        ("enableRotate", True),
        ("enablePan", True),

        ("minDistance", float(min_dist)),
        ("maxDistance", float(max_dist)),
        ("zoomSpeed",  float(zoom_speed)),

        # Sometimes controls use "minZoom/maxZoom" instead
        ("minZoom", 1e-6),
        ("maxZoom", 1e6),

        # Some controls have dollySpeed instead of zoomSpeed
        ("dollySpeed", float(zoom_speed)),

        # Damping/panning niceties
        ("screenSpacePanning", True),
        ("enableDamping", True),
    ]

    any_attempt = False
    for p in control_paths:
        for k, v in prop_sets:
            ok = _set_prop_safe(viewer[p], k, v)
            any_attempt = any_attempt or ok

    print("[Meshcat] Control brute-force config sent (this build doesn't confirm existence).")


def set_group_offset(viewer, group_name, xyz):
    viewer[group_name].set_transform(tf.translation_matrix(list(xyz)))


def normalize_freeflyer_quat_xyzw(q):
    q = q.copy()
    quat = q[3:7]
    n = np.linalg.norm(quat)
    if n > 1e-12:
        q[3:7] = quat / n
    return q


def override_robot_materials(viewer, prefix, visual_model, color_hex=0xD9DDE3, shininess=90, opacity=1.0):
    mat = g.MeshPhongMaterial(color=color_hex, shininess=shininess, opacity=opacity, transparent=(opacity < 1.0))
    for go in visual_model.geometryObjects:
        node_path = f"{prefix}/{go.name}"
        try:
            viewer[node_path].set_property("material", mat)
        except Exception:
            pass


# ============================================================
# Overlay SIM joint-origin points onto DDP robot (comparison)
# ============================================================
class JointMarkersOverlay:
    def __init__(self, viewer, model: pin.Model, parent_prefix="g1_fddp",
                 radius=0.03, color=0xFFFF00, alpha=1.0,
                 name="sim_joint_points", max_joints=30):
        self.viewer = viewer
        self.model = model
        self.data = model.createData()

        self.root = f"{parent_prefix}/{name}"
        self.geom = g.Sphere(radius)
        self.mat = g.MeshBasicMaterial(color=color, opacity=alpha, transparent=(alpha < 1.0))

        joint_ids = list(range(1, model.njoints))
        self.joint_ids = joint_ids if max_joints is None else joint_ids[: int(max_joints)]

        for jid in self.joint_ids:
            jname = model.names[jid]
            path = f"{self.root}/{jid:02d}_{jname}"
            self.viewer[path].set_object(self.geom, self.mat)
            _set_prop_safe(self.viewer[path], "material.depthTest", False)
            _set_prop_safe(self.viewer[path], "material.depthWrite", False)
            _set_prop_safe(self.viewer[path], "renderOrder", 9999)

    def update_from_qsim(self, q_sim):
        qn = normalize_freeflyer_quat_xyzw(q_sim)
        pin.forwardKinematics(self.model, self.data, qn)
        for jid in self.joint_ids:
            p = self.data.oMi[jid].translation
            self.viewer[f"{self.root}/{jid:02d}_{self.model.names[jid]}"].set_transform(
                tf.translation_matrix([float(p[0]), float(p[1]), float(p[2])])
            )


# ============================================================
# 1) Load G1 model
# ============================================================
urdf_path = "/home/jkang/third_party/PRIME/data_g1_anitescu/unitree_description/urdf/g1/main.urdf"
package_dirs = ["/home/jkang/third_party/PRIME/data_g1_anitescu/"]

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_path,
    package_dirs=package_dirs,
    root_joint=pin.JointModelFreeFlyer()
)

print("Model has nq =", model.nq, " nv =", model.nv)
print_joint_indices(model)


# ============================================================
# 2) Viewer + scene (NO camera set; mouse orbit)
# ============================================================
viewer = connect_or_start_meshcat()
print("Meshcat URL (leave this open):", WEB_URL)

setup_mjlab_studio(viewer, z0=-0.04, floor_size=30.0, grid=1.0)

viz_fddp = MeshcatVisualizer(model, collision_model, visual_model)
viz_fddp.initViewer(viewer, open=False)
viz_fddp.loadViewerModel("g1_fddp")

viz_sim = MeshcatVisualizer(model, collision_model, visual_model)
viz_sim.initViewer(viewer, open=False)
viz_sim.loadViewerModel("g1_sim")

# Give browser a moment to attach controls
time.sleep(0.1)
configure_controls_bruteforce(viewer, min_dist=1e-4, max_dist=1e6, zoom_speed=2.5, near=1e-4, far=1e6)

ROBOT_SEP = 0.4
set_group_offset(viewer, "g1_fddp", [+ROBOT_SEP, 0.0, 0.0])
set_group_offset(viewer, "g1_sim",  [-ROBOT_SEP, 0.0, 0.0])

override_robot_materials(viewer, "g1_fddp", visual_model, color_hex=0xD9DDE3, shininess=90, opacity=1.0)
override_robot_materials(viewer, "g1_sim",  visual_model, color_hex=0xD9DDE3, shininess=90, opacity=0.20)

overlay = JointMarkersOverlay(
    viewer, model,
    parent_prefix="g1_fddp",
    radius=0.035,
    color=0xFFFF00,
    alpha=1.0,
    name="sim_joint_points",
    max_joints=30
)

q0 = pin.neutral(model)
viz_fddp.display(q0)
viz_sim.display(q0)
overlay.update_from_qsim(q0)

# Configure again after first render (some builds create controls lazily)
time.sleep(0.1)
configure_controls_bruteforce(viewer, min_dist=1e-4, max_dist=1e6, zoom_speed=2.5, near=1e-4, far=1e6)


# ============================================================
# 3) Load trajectories
# ============================================================
csv_fddp = "/home/jkang/third_party/PRIME/data_g1_anitescu/xs_results_fddp.csv"
csv_sim  = "/home/jkang/third_party/PRIME/data_g1_anitescu/xs_log.csv"

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


# ============================================================
# 4) Playback throttling (keeps browser responsive)
# ============================================================
start_idx = 0
n_knots   = min(3000, q_sim.shape[0] - start_idx)
n_scale   = 1
end_idx   = start_idx + n_knots
q_sim_sampled = q_sim[start_idx:end_idx:n_scale]

PLAYBACK_HZ   = 60.0
DT_TARGET     = 1.0 / PLAYBACK_HZ
MARKER_STRIDE = 2
ROBOT_STRIDE  = 1

while True:
    t_next = time.time()
    frame = 0
    for q_ddp, q_s in zip(q_fddp, q_sim_sampled):
        if (frame % ROBOT_STRIDE) == 0:
            viz_fddp.display(normalize_freeflyer_quat_xyzw(q_ddp))
            viz_sim.display(normalize_freeflyer_quat_xyzw(q_s))

        if (frame % MARKER_STRIDE) == 0:
            overlay.update_from_qsim(q_s)

        frame += 1

        t_next += DT_TARGET
        now = time.time()
        if t_next > now:
            time.sleep(t_next - now)
        else:
            t_next = now

    time.sleep(0.5)
