#!/usr/bin/env python3
import argparse
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path


def _preload_system_numpy():
    """Load the OS NumPy before OpenRobots bindings see user-site NumPy 2.x."""
    system_dist = "/usr/lib/python3/dist-packages"
    if system_dist not in sys.path:
        return

    original_path = list(sys.path)
    sys.path.remove(system_dist)
    insert_at = 0
    for i, path in enumerate(sys.path):
        if path.endswith("/lib-dynload"):
            insert_at = i + 1
            break
    sys.path.insert(insert_at, system_dist)
    try:
        import numpy  # noqa: F401
    finally:
        sys.path[:] = original_path


_preload_system_numpy()

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


SKY_TOP = [0.92, 0.96, 1.00]
SKY_BOTTOM = [1.00, 1.00, 1.00]
GRID_SIZE = 20.0
GRID_STEP = 0.50
GRID_THICKNESS = 0.002
GROUND_COLOR = 0xFFFFFF
GRID_COLOR = 0xE6E6E6
GRID_OPACITY = 0.55
RAW_ORDER_COLOR = [0.95, 0.20, 0.10, 0.32]
PROCESSED_COLOR = [0.20, 0.45, 0.95, 0.92]


def attr_bool(node, name, default=False):
    value = node.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_int_list(text):
    if not text:
        return []
    return [int(piece.strip()) for piece in text.split(",") if piece.strip()]


def resolve_path(path_text, variables):
    path = path_text
    for key, value in variables.items():
        path = path.replace("${" + key + "}", value)
    return str(Path(path).expanduser())


def load_config(path):
    root = ET.parse(path).getroot()
    paths = root.find("paths")
    robot = root.find("robot")
    data = root.find("data")
    solver = root.find("solver")
    contacts = root.find("contacts")
    if paths is None or robot is None or data is None or solver is None:
        raise ValueError("Config is missing paths, robot, data, or solver block.")

    variables = dict(paths.attrib)
    for key, value in list(variables.items()):
        variables[key] = resolve_path(value, variables)

    contact_frames = []
    if contacts is not None:
        contact_frames = [frame.get("name") for frame in contacts.findall("frame")]
        contact_frames = [name for name in contact_frames if name]

    return {
        "workspace": Path(variables["workspace"]),
        "urdf": Path(resolve_path(robot.get("urdf"), variables)),
        "q_csv": Path(resolve_path(data.get("q"), variables)),
        "q_has_time_column": attr_bool(data, "q_has_time_column"),
        "joint_order": parse_int_list(data.get("joint_order", "")),
        "normalize_base_quaternion": attr_bool(data, "normalize_base_quaternion", True),
        "shift_base_to_ground": attr_bool(data, "shift_base_to_ground", False),
        "reuse_initial_ground_height": attr_bool(
            data, "reuse_initial_ground_height", True
        ),
        "start_idx": int(solver.get("start_idx", "0")),
        "down_sample": int(solver.get("down_sample", "1")),
        "interval": float(solver.get("interval", "0.005")),
        "contact_frames": contact_frames,
    }


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_csv_2d(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        data = np.atleast_2d(np.loadtxt(require_file(path), delimiter=","))
    if data.size == 0 or data.shape[1] == 0:
        raise ValueError(f"CSV file is empty: {path}")
    return data


def normalize_quaternions(qs):
    norms = np.linalg.norm(qs[:, 3:7], axis=1)
    valid = norms > 1e-12
    qs[valid, 3:7] /= norms[valid, None]


def apply_joint_order(qs, order):
    if not order:
        return qs.copy()
    if len(order) != qs.shape[1] - 7:
        raise ValueError(
            f"joint_order has {len(order)} entries, but q has {qs.shape[1] - 7} joints"
        )
    out = qs.copy()
    raw_joints = qs[:, 7:].copy()
    for i, src in enumerate(order):
        if src < 0 or src >= raw_joints.shape[1]:
            raise ValueError(f"joint_order entry out of range: {src}")
        out[:, 7 + i] = raw_joints[:, src]
    return out


def set_white_background(viewer):
    try:
        viewer["/Background"].set_property("top_color", SKY_TOP)
        viewer["/Background"].set_property("bottom_color", SKY_BOTTOM)
    except Exception:
        pass


def add_ground_with_grid(viewer, plane_z=-0.04):
    viewer["ground"].delete()
    mat_ground = g.MeshBasicMaterial(color=GROUND_COLOR, opacity=1.0)
    mat_grid = g.MeshBasicMaterial(
        color=GRID_COLOR, opacity=GRID_OPACITY, transparent=True
    )
    plane_thick = 0.01
    viewer["ground/plane"].set_object(
        g.Box([GRID_SIZE, GRID_SIZE, plane_thick]), mat_ground
    )
    viewer["ground/plane"].set_transform(
        tf.translation_matrix([0.0, 0.0, plane_z - plane_thick / 2.0])
    )

    z_grid = plane_z + 0.0008
    n = int(GRID_SIZE / GRID_STEP)
    for i in range(-n // 2, n // 2 + 1):
        y = i * GRID_STEP
        viewer[f"ground/grid_x/{i}"].set_object(
            g.Box([GRID_SIZE, GRID_THICKNESS, GRID_THICKNESS]), mat_grid
        )
        viewer[f"ground/grid_x/{i}"].set_transform(
            tf.translation_matrix([0.0, y, z_grid])
        )
        x = i * GRID_STEP
        viewer[f"ground/grid_y/{i}"].set_object(
            g.Box([GRID_THICKNESS, GRID_SIZE, GRID_THICKNESS]), mat_grid
        )
        viewer[f"ground/grid_y/{i}"].set_transform(
            tf.translation_matrix([x, 0.0, z_grid])
        )


def lowest_contact_height(model, data, frame_names, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    zs = []
    for name in frame_names:
        fid = model.getFrameId(name)
        if fid == len(model.frames):
            print(f"[warn] contact frame not found for ground shift: {name}")
            continue
        zs.append(data.oMf[fid].translation[2])
    if not zs:
        return 0.0
    return float(min(zs))


def shift_base_to_ground(model, frame_names, qs, reuse_initial=True):
    data = model.createData()
    if reuse_initial:
        ground = lowest_contact_height(model, data, frame_names, qs[0])
        qs[:, 2] -= ground
        return
    for q in qs:
        q[2] -= lowest_contact_height(model, data, frame_names, q)


def load_q_data(cfg, model, start_idx, stride, max_frames, use_joint_order):
    raw = load_csv_2d(cfg["q_csv"])
    offset = 1 if cfg["q_has_time_column"] else 0
    if raw.shape[1] < offset + model.nq:
        raise ValueError(
            f"{cfg['q_csv']} has {raw.shape[1]} columns, need at least "
            f"{offset + model.nq} for nq={model.nq}"
        )
    stop = raw.shape[0] if max_frames is None else min(raw.shape[0], start_idx + stride * max_frames)
    rows = raw[start_idx:stop:stride]
    qs_raw_order = rows[:, offset : offset + model.nq].copy()
    if cfg["normalize_base_quaternion"]:
        normalize_quaternions(qs_raw_order)
    qs = apply_joint_order(qs_raw_order, cfg["joint_order"]) if use_joint_order else qs_raw_order
    if cfg["shift_base_to_ground"]:
        shift_base_to_ground(
            model, cfg["contact_frames"], qs, cfg["reuse_initial_ground_height"]
        )
        if use_joint_order:
            raw_overlay = qs_raw_order.copy()
            shift_base_to_ground(
                model,
                cfg["contact_frames"],
                raw_overlay,
                cfg["reuse_initial_ground_height"],
            )
            qs_raw_order = raw_overlay
    return qs, qs_raw_order, rows[:, 0] if cfg["q_has_time_column"] else None


def model_joint_names(model):
    names = []
    for jid, name in enumerate(model.names):
        if model.nqs[jid] == 1 and model.idx_qs[jid] >= 7:
            names.append(name)
    return names


def print_joint_mapping(model, cfg):
    names = model_joint_names(model)
    print("Pinocchio/model joint order:")
    for i, name in enumerate(names):
        src = cfg["joint_order"][i] if cfg["joint_order"] else i
        csv_col = (1 if cfg["q_has_time_column"] else 0) + 7 + src
        print(f"  model[{i:02d}] {name:16s} <- raw_joint[{src:02d}] csv_col={csv_col}")


def main():
    experiment_dir = Path(__file__).resolve().parents[1]
    default_config = experiment_dir / "config" / "Go2_sim_kunzhao_mhe.xml"
    parser = argparse.ArgumentParser(
        description="Visualize Kunzhao Go2 p_sense.csv without running optimization."
    )
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--sleep", type=float, default=None)
    parser.add_argument("--no-joint-order", action="store_true")
    parser.add_argument("--compare-raw-order", action="store_true")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).expanduser())
    os.chdir(cfg["workspace"])
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(require_file(cfg["urdf"])),
        package_dirs=[str(cfg["workspace"])],
        root_joint=pin.JointModelFreeFlyer(),
    )

    start_idx = cfg["start_idx"] if args.start_idx is None else args.start_idx
    stride = cfg["down_sample"] if args.stride is None else args.stride
    sleep_dt = cfg["interval"] * stride if args.sleep is None else args.sleep
    use_joint_order = not args.no_joint_order
    qs, qs_raw_order, times = load_q_data(
        cfg, model, start_idx, stride, args.max_frames, use_joint_order
    )

    print("Using q data:", cfg["q_csv"])
    print("Using URDF:", cfg["urdf"])
    print(f"Loaded q frames={len(qs)} start_idx={start_idx} stride={stride}")
    print(f"Displayed data: {'configured joint_order' if use_joint_order else 'raw CSV joint order'}")
    print_joint_mapping(model, cfg)
    if args.check_only:
        return

    viewer = meshcat.Visualizer().open()
    print("Meshcat URL:", viewer.url())
    viewer.delete()
    set_white_background(viewer)
    add_ground_with_grid(viewer)

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(viewer, open=False)
    viz.loadViewerModel(
        "go2_data", visual_color=PROCESSED_COLOR if args.compare_raw_order else None
    )

    viz_raw = None
    if args.compare_raw_order and use_joint_order:
        viz_raw = MeshcatVisualizer(model, collision_model, visual_model)
        viz_raw.initViewer(viewer, open=False)
        viz_raw.loadViewerModel("go2_raw_order", visual_color=RAW_ORDER_COLOR)
        print("Overlay: blue = configured joint_order, red = raw CSV order")

    while True:
        for i, q in enumerate(qs):
            viz.display(q)
            if viz_raw is not None:
                viz_raw.display(qs_raw_order[i])
            time.sleep(sleep_dt)
        if args.no_loop:
            break
        time.sleep(0.5)


if __name__ == "__main__":
    main()
