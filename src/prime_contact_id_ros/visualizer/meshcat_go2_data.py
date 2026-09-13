#!/usr/bin/env python3
"""Visualize PRIME-compatible Go2 configuration logs in MeshCat."""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path


def _preload_system_numpy():
    """Load NumPy 1.x before the OpenRobots bindings see user-site NumPy 2.x."""
    system_dist = "/usr/lib/python3/dist-packages"
    if system_dist not in sys.path:
        return

    original_path = list(sys.path)
    sys.path.remove(system_dist)
    insert_at = 0
    for index, path in enumerate(sys.path):
        if path.endswith("/lib-dynload"):
            insert_at = index + 1
            break
    sys.path.insert(insert_at, system_dist)
    try:
        import numpy  # noqa: F401
    finally:
        sys.path[:] = original_path


_preload_system_numpy()

import meshcat
import meshcat.geometry as geometry
import meshcat.transformations as transformations
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


JOINT_ORDER = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
CONTACT_FRAMES = ["RR_foot", "RL_foot", "FR_foot", "FL_foot"]
SKY_TOP = [0.92, 0.96, 1.00]
SKY_BOTTOM = [1.00, 1.00, 1.00]
GRID_SIZE = 20.0
GRID_STEP = 0.50
GRID_THICKNESS = 0.002


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_configurations(path, nq, start_idx, stride, max_frames):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = np.atleast_2d(np.loadtxt(require_file(path), delimiter=","))
    expected_columns = 1 + nq
    if raw.shape[1] < expected_columns:
        raise ValueError(
            f"{path} has {raw.shape[1]} columns; expected at least {expected_columns}"
        )
    if start_idx < 0 or start_idx >= raw.shape[0]:
        raise ValueError(f"start index {start_idx} is outside {raw.shape[0]} rows")
    stop = raw.shape[0]
    if max_frames is not None:
        stop = min(stop, start_idx + stride * max_frames)
    configurations = raw[start_idx:stop:stride, 1 : 1 + nq].copy()
    norms = np.linalg.norm(configurations[:, 3:7], axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("configuration log contains an invalid base quaternion")
    configurations[:, 3:7] /= norms[:, None]
    return configurations


def apply_joint_order(configurations):
    if configurations.shape[1] - 7 != len(JOINT_ORDER):
        raise ValueError("logged joint count does not match the Go2 permutation")
    ordered = configurations.copy()
    ordered[:, 7:] = configurations[:, 7:][:, JOINT_ORDER]
    return ordered


def model_joint_names(model):
    return [
        name
        for joint_id, name in enumerate(model.names)
        if model.nqs[joint_id] == 1 and model.idx_qs[joint_id] >= 7
    ]


def print_joint_mapping(model):
    print("Unitree CSV to Pinocchio joint mapping:")
    for model_index, name in enumerate(model_joint_names(model)):
        raw_index = JOINT_ORDER[model_index]
        print(
            f"  model[{model_index:02d}] {name:16s} "
            f"<- raw_joint[{raw_index:02d}] csv_col={8 + raw_index}"
        )


def lowest_contact_height(model, data, configuration):
    pin.forwardKinematics(model, data, configuration)
    pin.updateFramePlacements(model, data)
    heights = []
    for frame_name in CONTACT_FRAMES:
        frame_id = model.getFrameId(frame_name)
        if frame_id < len(model.frames):
            heights.append(data.oMf[frame_id].translation[2])
    if not heights:
        raise ValueError("none of the Go2 contact frames exists in the model")
    return float(min(heights))


def shift_to_ground(model, configurations):
    ground_height = lowest_contact_height(
        model, model.createData(), configurations[0]
    )
    configurations[:, 2] -= ground_height


def set_scene(viewer):
    try:
        viewer["/Background"].set_property("top_color", SKY_TOP)
        viewer["/Background"].set_property("bottom_color", SKY_BOTTOM)
    except Exception:
        pass

    viewer["ground"].delete()
    ground_material = geometry.MeshBasicMaterial(color=0xFFFFFF, opacity=1.0)
    grid_material = geometry.MeshBasicMaterial(
        color=0xE6E6E6, opacity=0.55, transparent=True
    )
    plane_z = -0.04
    plane_thickness = 0.01
    viewer["ground/plane"].set_object(
        geometry.Box([GRID_SIZE, GRID_SIZE, plane_thickness]), ground_material
    )
    viewer["ground/plane"].set_transform(
        transformations.translation_matrix(
            [0.0, 0.0, plane_z - plane_thickness / 2.0]
        )
    )
    grid_z = plane_z + 0.0008
    count = int(GRID_SIZE / GRID_STEP)
    for index in range(-count // 2, count // 2 + 1):
        coordinate = index * GRID_STEP
        viewer[f"ground/grid_x/{index}"].set_object(
            geometry.Box([GRID_SIZE, GRID_THICKNESS, GRID_THICKNESS]),
            grid_material,
        )
        viewer[f"ground/grid_x/{index}"].set_transform(
            transformations.translation_matrix([0.0, coordinate, grid_z])
        )
        viewer[f"ground/grid_y/{index}"].set_object(
            geometry.Box([GRID_THICKNESS, GRID_SIZE, GRID_THICKNESS]),
            grid_material,
        )
        viewer[f"ground/grid_y/{index}"].set_transform(
            transformations.translation_matrix([coordinate, 0.0, grid_z])
        )


def main():
    inferred_workspace = Path(__file__).resolve().parents[3]
    workspace = Path(os.environ.get("WORKSPACE", inferred_workspace)).expanduser()
    default_csv = workspace / "results" / "iekf_bag_replay" / "p_sense.csv"
    default_urdf = (
        workspace
        / "core"
        / "experiments"
        / "Go2_sim_kunzhao_long_mhe"
        / "descriptions"
        / "urdf"
        / "go2.urdf"
    )

    parser = argparse.ArgumentParser(
        description="Visualize the Go2 motion logged by IEKF bag replay."
    )
    parser.add_argument("--csv", default=str(default_csv))
    parser.add_argument("--urdf", default=str(default_urdf))
    parser.add_argument("--package-dir", default=str(workspace / "core"))
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum displayed frames; by default all remaining rows are loaded.",
    )
    parser.add_argument("--sleep", type=float, default=0.005)
    parser.add_argument("--no-ground-shift", action="store_true")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.sleep < 0.0:
        parser.error("--sleep must be nonnegative")

    urdf = require_file(Path(args.urdf).expanduser())
    csv_path = require_file(Path(args.csv).expanduser())
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(urdf),
        package_dirs=[str(Path(args.package_dir).expanduser())],
        root_joint=pin.JointModelFreeFlyer(),
    )
    configurations = load_configurations(
        csv_path, model.nq, args.start_idx, args.stride, args.max_frames
    )
    configurations = apply_joint_order(configurations)
    if not args.no_ground_shift:
        shift_to_ground(model, configurations)

    print("Using q data:", csv_path)
    print("Using URDF:", urdf)
    print(
        f"Loaded frames={len(configurations)} start_idx={args.start_idx} "
        f"stride={args.stride}"
    )
    print_joint_mapping(model)
    if args.check_only:
        return

    viewer = meshcat.Visualizer().open()
    print("MeshCat URL:", viewer.url())
    viewer.delete()
    set_scene(viewer)

    visualizer = MeshcatVisualizer(model, collision_model, visual_model)
    visualizer.initViewer(viewer, open=False)
    visualizer.loadViewerModel("go2_iekf_data")
    while True:
        for configuration in configurations:
            visualizer.display(configuration)
            time.sleep(args.sleep)
        if args.no_loop:
            break
        time.sleep(0.5)


if __name__ == "__main__":
    main()
