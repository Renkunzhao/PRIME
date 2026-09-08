#!/usr/bin/env python3
import os
import sys
import tempfile
import time
import warnings
import argparse
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
GROUND_OPACITY = 1.0
GRID_COLOR = 0xE6E6E6
GRID_OPACITY = 0.55

FORCE_SCALE = 0.0015
SHAFT_RATIO = 0.75
SHAFT_RADIUS = 0.006
HEAD_RADIUS = 0.016
ARROW_COLOR = 0xFFAA00
HEAD_COLOR = 0xFFCC33
ARROW_OPACITY = 0.95
FORCES_IN_LOCAL_FRAME = False

LOG_ROBOT_COLOR = [0.55, 0.55, 0.55, 0.25]
FOOT_FRAMES = ["RR_foot", "RL_foot", "FR_foot", "FL_foot"]
FORCE_ROOT = "go2_forces"


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

    for i in range(-n // 2, n // 2 + 1):
        x = i * GRID_STEP
        viewer[f"ground/grid_y/{i}"].set_object(
            g.Box([GRID_THICKNESS, GRID_SIZE, GRID_THICKNESS]), mat_grid
        )
        viewer[f"ground/grid_y/{i}"].set_transform(
            tf.translation_matrix([x, 0.0, z_grid])
        )


def _skew(v):
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def rot_y_to_vec(d):
    d = np.asarray(d).reshape(3)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.eye(3)
    d = d / n

    y = np.array([0.0, 1.0, 0.0])
    c = float(np.dot(y, d))

    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])

    v = np.cross(y, d)
    s = np.linalg.norm(v)
    vx = _skew(v)
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


_CONE_OBJ_PATH = None


def _write_unit_cone_obj(path, n_sides=24):
    tip = np.array([0.0, 0.5, 0.0])
    base_y = -0.5
    radius = 0.5
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    ring = np.stack(
        [radius * np.cos(angles), np.full_like(angles, base_y), radius * np.sin(angles)],
        axis=1,
    )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# unit cone +Y\n")
        handle.write(f"v {tip[0]} {tip[1]} {tip[2]}\n")
        for vertex in ring:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for i in range(n_sides):
            a = 2 + i
            b = 2 + ((i + 1) % n_sides)
            handle.write(f"f 1 {a} {b}\n")


def get_cone_geometry():
    global _CONE_OBJ_PATH
    if _CONE_OBJ_PATH is None:
        path = os.path.join(tempfile.gettempdir(), "meshcat_unit_cone_y.obj")
        if not os.path.exists(path):
            _write_unit_cone_obj(path)
        _CONE_OBJ_PATH = path
    return g.ObjMeshGeometry.from_file(_CONE_OBJ_PATH)


def make_arrow(viewer, base):
    mat_s = g.MeshBasicMaterial(
        color=ARROW_COLOR, opacity=ARROW_OPACITY, transparent=True
    )
    mat_h = g.MeshBasicMaterial(
        color=HEAD_COLOR, opacity=ARROW_OPACITY, transparent=True
    )

    viewer[f"{base}/shaft"].set_object(g.Cylinder(1.0, SHAFT_RADIUS), mat_s)
    viewer[f"{base}/head"].set_object(get_cone_geometry(), mat_h)


def set_arrow(viewer, base, p, f):
    mag = np.linalg.norm(f)
    if mag < 1e-12:
        viewer[f"{base}/shaft"].set_property("visible", False)
        viewer[f"{base}/head"].set_property("visible", False)
        return

    viewer[f"{base}/shaft"].set_property("visible", True)
    viewer[f"{base}/head"].set_property("visible", True)

    d = f / mag
    length = mag * FORCE_SCALE
    shaft_len = SHAFT_RATIO * length
    head_len = max((1.0 - SHAFT_RATIO) * length, 0.18 * length)

    rotation = np.eye(4)
    rotation[:3, :3] = rot_y_to_vec(d)

    scale_shaft = np.diag([1.0, shaft_len, 1.0, 1.0])
    scale_head = np.diag([2 * HEAD_RADIUS, head_len, 2 * HEAD_RADIUS, 1.0])

    p_shaft = p + d * (shaft_len / 2.0)
    p_head = p + d * (shaft_len + head_len / 2.0)

    viewer[f"{base}/shaft"].set_transform(
        tf.translation_matrix(p_shaft) @ rotation @ scale_shaft
    )
    viewer[f"{base}/head"].set_transform(
        tf.translation_matrix(p_head) @ rotation @ scale_head
    )


def list_window_dirs(results_root):
    if not results_root.exists():
        return []
    def window_sort_key(path):
        parts = path.name.split("_")
        try:
            return (0, int(parts[1]), int(parts[3]), path.name)
        except (IndexError, ValueError):
            return (1, 0, 0, path.name)

    return sorted(
        (path for path in results_root.glob("window_*_start_*") if path.is_dir()),
        key=window_sort_key,
    )


def resolve_window_dir(experiment_dir, args):
    def resolve_cli_path(path):
        path = path.expanduser()
        if path.is_absolute():
            return path.resolve()
        cwd_path = path.resolve()
        if cwd_path.exists():
            return cwd_path
        return (experiment_dir / path).resolve()

    results_root = resolve_cli_path(args.results_root)

    if args.window_dir is not None:
        return resolve_cli_path(args.window_dir)

    windows = list_window_dirs(results_root)
    if not windows:
        raise FileNotFoundError(
            f"No MHE window result directories found under {results_root}."
        )
    if args.window < 0:
        raise IndexError("Window index must be non-negative.")
    if args.window >= len(windows):
        raise IndexError(
            f"Requested window {args.window}, but only {len(windows)} windows exist."
        )
    return windows[args.window]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize one Go2 mass+3kg MHE window: optimized trajectory vs raw/reference state log."
    )
    parser.add_argument(
        "window_dir",
        nargs="?",
        type=Path,
        help="specific window result directory, e.g. results/window_000_start_0200",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="window index to load from --results-root when no window_dir is provided",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="MHE results root containing window_*_start_* directories",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="playback sleep time per frame",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="play the selected window once instead of looping forever",
    )
    return parser.parse_args()


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required visualizer input: {path}\n"
            "Run the MHE experiment first or pass a window result directory, e.g.\n"
            "  python3 experiments/Go2_sim_kunzhao_long_mhe/visualizer/meshcat_go2_window.py "
            "experiments/Go2_sim_kunzhao_long_mhe/results/window_000_start_1600"
        )
    return path


def load_csv_2d(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        data = np.atleast_2d(np.loadtxt(require_file(path), delimiter=","))
    if data.size == 0 or data.shape[1] == 0:
        raise ValueError(f"CSV file is empty: {path}")
    return data


def load_state_csv(path, nq, fallback=None):
    try:
        data = load_csv_2d(path)
        if data.shape[1] < nq:
            raise ValueError(
                f"State CSV has {data.shape[1]} columns but model.nq is {nq}: {path}"
            )
        return data[:, :nq]
    except Exception as exc:
        if fallback is None:
            raise
        print(f"[warn] {path} is not usable ({exc}); using the log states instead.")
        return fallback.copy()


def main():
    args = parse_args()
    experiment_dir = Path(__file__).resolve().parents[1]
    workspace_dir = experiment_dir.parents[1]
    descriptions_dir = experiment_dir / "descriptions"
    results_dir = resolve_window_dir(experiment_dir, args)
    urdf_path = require_file(descriptions_dir / "urdf" / "go2.urdf")

    os.chdir(workspace_dir)
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(urdf_path),
        package_dirs=[str(workspace_dir)],
        root_joint=pin.JointModelFreeFlyer(),
    )
    data = model.createData()

    viewer = meshcat.Visualizer().open()
    print("Meshcat URL:", viewer.url())
    print("Using descriptions:", descriptions_dir)
    print("Using MHE window results:", results_dir)
    viewer.delete()

    set_white_background(viewer)
    add_ground_with_grid_unlit(viewer)

    viz_fddp = MeshcatVisualizer(model, collision_model, visual_model)
    viz_fddp.initViewer(viewer, open=False)
    viz_fddp.loadViewerModel("go2_fddp")

    viz_log = MeshcatVisualizer(model, collision_model, visual_model)
    viz_log.initViewer(viewer, open=False)
    viz_log.loadViewerModel("go2_log", visual_color=LOG_ROBOT_COLOR)

    q0 = pin.neutral(model)
    viz_fddp.display(q0)
    viz_log.display(q0)

    foot_ids = []
    kept_names = []
    for name in FOOT_FRAMES:
        fid = model.getFrameId(name)
        if fid == len(model.frames):
            print(f"[warn] frame not found: {name} (skip)")
            continue
        foot_ids.append(fid)
        kept_names.append(name)
        make_arrow(viewer, f"{FORCE_ROOT}/{name}")

    q_log = load_state_csv(results_dir / "xs_log.csv", model.nq)
    q_fddp = load_state_csv(
        results_dir / "xs_results_fddp.csv", model.nq, fallback=q_log
    )
    f_arr = load_csv_2d(results_dir / "f_rollout.csv")[:, : 3 * len(FOOT_FRAMES)]

    n_frames = min(q_fddp.shape[0], q_log.shape[0], f_arr.shape[0])
    print(
        "Loaded frames:",
        f"fddp={q_fddp.shape[0]}",
        f"log={q_log.shape[0]}",
        f"forces={f_arr.shape[0]}",
        f"playback={n_frames}",
    )

    sleep_dt = args.dt
    while True:
        for q_est, q_meas, force_row in zip(
            q_fddp[:n_frames], q_log[:n_frames], f_arr[:n_frames]
        ):
            viz_fddp.display(q_est)
            viz_log.display(q_meas)

            pin.forwardKinematics(model, data, q_est)
            pin.updateFramePlacements(model, data)

            forces = force_row.reshape(len(FOOT_FRAMES), 3)
            for name, fid, force in zip(kept_names, foot_ids, forces):
                oMf = data.oMf[fid]
                force_w = oMf.rotation @ force if FORCES_IN_LOCAL_FRAME else force
                set_arrow(viewer, f"{FORCE_ROOT}/{name}", oMf.translation, force_w)

            time.sleep(sleep_dt)

        if args.once:
            break
        time.sleep(1.0)


if __name__ == "__main__":
    main()
