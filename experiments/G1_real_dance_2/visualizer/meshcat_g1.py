#!/usr/bin/env python3
import time
import os
import sys
import tempfile
import numpy as np

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

import meshcat
import meshcat.transformations as tf
import meshcat.geometry as g


# ---------------------------
# Background + terrain knobs (simple, no light tuning)
# ---------------------------
SKY_TOP    = [0.92, 0.96, 1.00]
SKY_BOTTOM = [1.00, 1.00, 1.00]

GRID_SIZE = 20.0
GRID_STEP = 0.50
GRID_THICKNESS = 0.002

GROUND_COLOR = 0xFFFFFF
GROUND_OPACITY = 1.0

GRID_COLOR = 0xE6E6E6
GRID_OPACITY = 0.55


# ---------------------------
# Force arrow knobs
# ---------------------------
FORCE_SCALE = 0.0006
SHAFT_RATIO = 0.75
ARROW_COLOR = 0xFFAA00
HEAD_COLOR  = 0xFFCC33
ARROW_OPACITY = 0.95
SHAFT_RADIUS = 0.006
HEAD_RADIUS  = 0.016
FORCES_IN_LOCAL_FRAME = False
LOG_ROBOT_COLOR = [0.55, 0.55, 0.55, 0.25]


# ---------------------------
# Background + unlit ground
# ---------------------------
def set_white_background(viewer):
    try:
        viewer["/Background"].set_property("top_color", SKY_TOP)
        viewer["/Background"].set_property("bottom_color", SKY_BOTTOM)
    except Exception:
        pass


def add_ground_with_grid_unlit(viewer, plane_z=-0.04):
    viewer["ground"].delete()

    mat_ground = g.MeshBasicMaterial(
        color=GROUND_COLOR,
        opacity=GROUND_OPACITY,
        transparent=(GROUND_OPACITY < 1.0),
    )
    mat_grid = g.MeshBasicMaterial(
        color=GRID_COLOR,
        opacity=GRID_OPACITY,
        transparent=True,
    )

    plane_thick = 0.01
    plane_size = (GRID_SIZE, GRID_SIZE, plane_thick)
    viewer["ground/plane"].set_object(g.Box(plane_size), mat_ground)
    viewer["ground/plane"].set_transform(
        tf.translation_matrix([0.0, 0.0, plane_z - plane_thick / 2.0])
    )

    z_grid = plane_z + 0.0008
    n = int(GRID_SIZE / GRID_STEP)

    for i in range(-n // 2, n // 2 + 1):
        y = i * GRID_STEP
        box = g.Box([GRID_SIZE, GRID_THICKNESS, GRID_THICKNESS])
        viewer[f"ground/grid_x/{i}"].set_object(box, mat_grid)
        viewer[f"ground/grid_x/{i}"].set_transform(tf.translation_matrix([0.0, y, z_grid]))

    for i in range(-n // 2, n // 2 + 1):
        x = i * GRID_STEP
        box = g.Box([GRID_THICKNESS, GRID_SIZE, GRID_THICKNESS])
        viewer[f"ground/grid_y/{i}"].set_object(box, mat_grid)
        viewer[f"ground/grid_y/{i}"].set_transform(tf.translation_matrix([x, 0.0, z_grid]))


# ---------------------------
# Arrow helpers
# ---------------------------
def _set_prop_safe(node, key, value):
    try:
        node.set_property(key, value)
        return True
    except Exception:
        return False


def _skew(v):
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def rot_y_to_vec(d):
    d = np.asarray(d).reshape(3)
    nd = np.linalg.norm(d)
    if nd < 1e-12:
        return np.eye(3)
    d = d / nd

    y = np.array([0.0, 1.0, 0.0])
    c = float(np.dot(y, d))
    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        return np.array([[1.0, 0.0, 0.0],
                         [0.0,-1.0, 0.0],
                         [0.0, 0.0,-1.0]])

    v = np.cross(y, d)
    s = np.linalg.norm(v)
    vx = _skew(v)
    R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
    return R


# --------- robust arrowhead mesh (OBJ) ----------
_CONE_OBJ_PATH = None

def _write_unit_cone_obj(path, n_sides=24):
    """
    Write a unit cone OBJ oriented along +Y, centered at origin along Y:
      - base circle at y=-0.5
      - tip at y=+0.5
      - radius = 0.5 (so width is reasonable; we will scale later)
    """
    # vertices: tip + base ring
    tip = np.array([0.0,  0.5, 0.0])
    base_y = -0.5
    r = 0.5
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    ring = np.stack([r*np.cos(angles), np.full_like(angles, base_y), r*np.sin(angles)], axis=1)

    # Write OBJ
    with open(path, "w") as f:
        f.write("# unit cone along +Y\n")
        f.write(f"v {tip[0]} {tip[1]} {tip[2]}\n")
        for v in ring:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # faces (tip to ring)
        # tip index = 1, ring indices = 2..n_sides+1
        for i in range(n_sides):
            a = 2 + i
            b = 2 + ((i + 1) % n_sides)
            f.write(f"f 1 {a} {b}\n")
        # optional base cap (triangulate fan around first ring vertex)
        # this helps it look solid from below
        for i in range(1, n_sides - 1):
            a = 2
            b = 2 + i
            c = 2 + i + 1
            f.write(f"f {a} {c} {b}\n")


def get_cone_obj_geometry():
    """
    Returns an ObjMeshGeometry cone oriented along +Y.
    Generated once and cached.
    """
    global _CONE_OBJ_PATH
    if _CONE_OBJ_PATH is None:
        tmpdir = tempfile.gettempdir()
        path = os.path.join(tmpdir, "meshcat_unit_cone_y.obj")
        if not os.path.exists(path):
            _write_unit_cone_obj(path, n_sides=24)
        _CONE_OBJ_PATH = path
    return g.ObjMeshGeometry.from_file(_CONE_OBJ_PATH)


def make_arrow_objects_unlit(
    viewer,
    base_path,
    shaft_radius=SHAFT_RADIUS,
    head_radius=HEAD_RADIUS,
    opacity=ARROW_OPACITY,
    color=ARROW_COLOR,
    head_color=HEAD_COLOR,
):
    """
    True arrowhead shape:
      - shaft: Cylinder(1.0, radius) (along +Y)
      - head : OBJ cone mesh (along +Y)
    Both unit-length and scaled each frame.
    """
    mat_shaft = g.MeshBasicMaterial(color=color, opacity=opacity, transparent=(opacity < 1.0))
    mat_head  = g.MeshBasicMaterial(color=head_color, opacity=opacity, transparent=(opacity < 1.0))

    viewer[f"{base_path}/shaft"].set_object(g.Cylinder(1.0, shaft_radius), mat_shaft)

    # True cone mesh head (unit size, will be scaled)
    cone_geom = get_cone_obj_geometry()
    viewer[f"{base_path}/head"].set_object(cone_geom, mat_head)

    # Render on top
    for part in ("shaft", "head"):
        _set_prop_safe(viewer[f"{base_path}/{part}"], "material.depthTest", False)
        _set_prop_safe(viewer[f"{base_path}/{part}"], "material.depthWrite", False)
        _set_prop_safe(viewer[f"{base_path}/{part}"], "renderOrder", 9999)


def set_arrow(viewer, base_path, origin_w, vec_w,
              force_scale=FORCE_SCALE, min_len=1e-4, shaft_ratio=SHAFT_RATIO):
    origin_w = np.asarray(origin_w).reshape(3)
    vec_w    = np.asarray(vec_w).reshape(3)

    fmag = float(np.linalg.norm(vec_w))
    L = fmag * float(force_scale)

    if (fmag < 1e-12) or (L < min_len):
        viewer[f"{base_path}/shaft"].set_property("visible", False)
        viewer[f"{base_path}/head"].set_property("visible", False)
        return

    viewer[f"{base_path}/shaft"].set_property("visible", True)
    viewer[f"{base_path}/head"].set_property("visible", True)

    d = vec_w / fmag
    R3 = rot_y_to_vec(d)
    R4 = np.eye(4)
    R4[:3, :3] = R3

    shaft_len = shaft_ratio * L
    head_len  = (1.0 - shaft_ratio) * L
    head_len = max(head_len, 0.18 * L)

    # IMPORTANT:
    # - Cylinder radius is set at creation (shaft_radius), so we scale only along Y for it
    # - OBJ cone has unit radius ~0.5; we want it to match HEAD_RADIUS, so scale X/Z too
    # The unit cone has radius 0.5, so to get desired radius:
    #   scale_xz = HEAD_RADIUS / 0.5 = 2*HEAD_RADIUS
    scale_xz = 2.0 * HEAD_RADIUS

    S_shaft = np.diag([1.0, shaft_len, 1.0, 1.0])
    S_head  = np.diag([scale_xz, head_len, scale_xz, 1.0])

    shaft_center = origin_w + d * (shaft_len / 2.0)
    head_center  = origin_w + d * (shaft_len + head_len / 2.0)

    T_shaft = tf.translation_matrix(shaft_center)
    T_head  = tf.translation_matrix(head_center)

    viewer[f"{base_path}/shaft"].set_transform(T_shaft @ R4 @ S_shaft)
    viewer[f"{base_path}/head"].set_transform(T_head  @ R4 @ S_head)


# ---------------------------
# Main
# ---------------------------
def resolve_results_dir(experiment_dir):
    if len(sys.argv) > 2:
        raise SystemExit("Usage: meshcat_arrow_transparent_log.py [results_dir]")
    if len(sys.argv) == 2:
        return os.path.abspath(sys.argv[1])
    return os.path.join(experiment_dir, "results")


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required visualizer input: {path}\n"
            "Run the G1 experiment first or pass a results directory, e.g.\n"
            "  python3 experiments/G1/visualizer/meshcat_arrow_transparent_log.py "
            "experiments/G1/results"
        )
    return path


def main():
    experiment_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    descriptions_dir = os.path.join(experiment_dir, "descriptions")
    results_dir = resolve_results_dir(experiment_dir)
    urdf_path = os.path.join(descriptions_dir, "urdf", "main.urdf")
    package_dirs = [descriptions_dir]

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_path,
        package_dirs=package_dirs,
        root_joint=pin.JointModelFreeFlyer(),
    )
    data = model.createData()
    print("Model nq =", model.nq, "nv =", model.nv)

    viewer = meshcat.Visualizer().open()
    print("Meshcat URL:", viewer.url())
    viewer.delete()

    set_white_background(viewer)
    add_ground_with_grid_unlit(viewer, plane_z=-0.04)

    viz_fddp = MeshcatVisualizer(model, collision_model, visual_model)
    viz_fddp.initViewer(viewer, open=False)
    viz_fddp.loadViewerModel("g1_fddp")

    viz_sim = MeshcatVisualizer(model, collision_model, visual_model)
    viz_sim.initViewer(viewer, open=False)
    viz_sim.loadViewerModel("g1_sim", visual_color=LOG_ROBOT_COLOR)

    q0 = pin.neutral(model)
    viz_fddp.display(q0)
    viz_sim.display(q0)

    foot_frames = [
        "LL_FOOT_FL", "LL_FOOT_FR", "LL_FOOT_RL", "LL_FOOT_RR",
        "LR_FOOT_FL", "LR_FOOT_FR", "LR_FOOT_RL", "LR_FOOT_RR",
    ]

    foot_frame_ids = []
    kept_names = []
    for name in foot_frames:
        fid = model.getFrameId(name)
        if fid == len(model.frames):
            print(f"[warn] frame not found: {name} (skip)")
            continue
        foot_frame_ids.append(fid)
        kept_names.append(name)

    for name in kept_names:
        make_arrow_objects_unlit(viewer, f"forces/{name}")

    csv_fddp = require_file(os.path.join(results_dir, "xs_results_fddp.csv"))
    csv_sim = require_file(os.path.join(results_dir, "xs_log.csv"))
    f_fddp = require_file(os.path.join(results_dir, "f_rollout.csv"))
    print("Using results:", results_dir)

    q_fddp = np.atleast_2d(np.loadtxt(csv_fddp, delimiter=","))
    q_sim  = np.atleast_2d(np.loadtxt(csv_sim,  delimiter=","))
    f_arr  = np.atleast_2d(np.loadtxt(f_fddp,  delimiter=","))

    q_fddp = q_fddp[:, :model.nq]
    q_sim  = q_sim[:,  :model.nq]
    f_arr  = f_arr[:, :24]  # 8*3

    # For your debug use:
    # q_sim = q_fddp
    # q_fddp = q_sim

    sleep_dt = 0.01

    while True:
        for q1, q2, ff in zip(q_fddp, q_sim, f_arr):
            viz_fddp.display(q1)
            viz_sim.display(q2)

            pin.forwardKinematics(model, data, q1)
            pin.updateFramePlacements(model, data)

            forces_all = ff.reshape(8, 3)

            for name, fid in zip(kept_names, foot_frame_ids):
                k = foot_frames.index(name)
                oMf = data.oMf[fid]
                p_w = oMf.translation

                f = forces_all[k]
                f_w = (oMf.rotation @ f) if FORCES_IN_LOCAL_FRAME else f

                set_arrow(viewer, f"forces/{name}", p_w, f_w)

            time.sleep(sleep_dt)

        time.sleep(1.0)


if __name__ == "__main__":
    main()
