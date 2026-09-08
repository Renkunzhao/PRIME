#!/usr/bin/env python3
import argparse
import math
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


Q_BASE_LABELS = ["base_x", "base_y", "base_z", "quat_x", "quat_y", "quat_z", "quat_w"]
V_BASE_LABELS = ["base_vx", "base_vy", "base_vz", "base_wx", "base_wy", "base_wz"]
U_BASE_LABELS = ["root_fx", "root_fy", "root_fz", "root_tx", "root_ty", "root_tz"]


def attr_bool(node, name, default=False):
    value = node.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_path(path_text, variables):
    path = path_text
    for key, value in variables.items():
        path = path.replace("${" + key + "}", value)
    return str(Path(path).expanduser())


def load_config(path):
    root = ET.parse(path).getroot()
    paths = root.find("paths")
    robot = root.find("robot")
    solver = root.find("solver")
    outputs = root.find("outputs")
    if paths is None or robot is None or solver is None or outputs is None:
        raise ValueError("Config is missing paths, robot, solver, or outputs block.")

    variables = dict(paths.attrib)
    for key, value in list(variables.items()):
        variables[key] = resolve_path(value, variables)

    results_dir = Path(variables["results"])
    return {
        "workspace": Path(variables["workspace"]),
        "urdf": Path(resolve_path(robot.get("urdf"), variables)),
        "results": results_dir,
        "start_idx": int(solver.get("start_idx", "0")),
        "down_sample": int(solver.get("down_sample", "1")),
        "interval": float(solver.get("interval", "0.005")),
        "xs_log": outputs.get("xs_log", "xs_log.csv"),
        "us_log": outputs.get("us_log", "us_log.csv"),
        "xs_results": outputs.get("xs_results", "xs_results_fddp.csv"),
        "us_results": outputs.get("us_results", "us_results_fddp.csv"),
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


def urdf_model_info(urdf_path):
    root = ET.parse(urdf_path).getroot()
    names = []
    for joint in root.findall("joint"):
        joint_type = joint.get("type", "")
        if joint_type in {"revolute", "continuous", "prismatic"}:
            names.append(joint.get("name", f"joint_{len(names)}"))
    return {
        "nq": 7 + len(names),
        "nv": 6 + len(names),
        "joint_names": names,
    }


def labels_for_model(model_info):
    joint_names = model_info["joint_names"]
    q_labels = Q_BASE_LABELS + joint_names
    v_labels = V_BASE_LABELS + [name.replace("_joint", "_vel") for name in joint_names]
    u_labels = U_BASE_LABELS + [name.replace("_joint", "_tau") for name in joint_names]
    return q_labels, v_labels, u_labels


def state_slices(xs, model_info):
    nq = model_info["nq"]
    nv = model_info["nv"]
    remaining = xs.shape[1] - nq - nv
    if remaining < 0 or remaining % 20 != 0:
        raise ValueError(
            f"Cannot infer contact-ID state layout from xs columns={xs.shape[1]}, "
            f"nq={nq}, nv={nv}."
        )
    n_identified = remaining // 20
    v_start = nq + 10 * n_identified
    return xs[:, :nq], xs[:, v_start : v_start + nv], n_identified


def plot_group(pdf, output_dir, stem, title, x, raw, opt, labels, per_page):
    n = min(raw.shape[0], opt.shape[0], len(x))
    raw = raw[:n]
    opt = opt[:n]
    x = x[:n]
    pages = int(math.ceil(raw.shape[1] / float(per_page)))

    for page in range(pages):
        start = page * per_page
        stop = min(raw.shape[1], start + per_page)
        count = stop - start
        ncols = 3
        nrows = int(math.ceil(count / float(ncols)))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(15, max(3.0, 2.55 * nrows)), squeeze=False
        )
        axes_flat = axes.ravel()

        for local, idx in enumerate(range(start, stop)):
            ax = axes_flat[local]
            label = labels[idx] if idx < len(labels) else f"{stem}[{idx}]"
            ax.plot(x, raw[:, idx], color="0.25", linewidth=1.2, label="raw log")
            ax.plot(x, opt[:, idx], color="#1f77b4", linewidth=1.1, label="optimized")
            ax.plot(
                x,
                opt[:, idx] - raw[:, idx],
                color="#d62728",
                linewidth=0.8,
                alpha=0.65,
                label="opt - raw",
            )
            ax.set_title(f"{idx}: {label}", fontsize=9)
            ax.grid(True, alpha=0.25)
            if local % ncols == 0:
                ax.set_ylabel("value")
            if local >= (nrows - 1) * ncols:
                ax.set_xlabel("knot")

        for ax in axes_flat[count:]:
            ax.axis("off")

        handles, handle_labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, handle_labels, loc="upper right", ncol=3)
        fig.suptitle(f"{title} ({page + 1}/{pages})", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        png_path = output_dir / f"{stem}_comparison_page_{page + 1:02d}.png"
        fig.savefig(png_path, dpi=180)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"Wrote {png_path}")


def main():
    experiment_dir = Path(__file__).resolve().parents[1]
    default_config = experiment_dir / "config" / "Go2_sim_kunzhao.xml"
    parser = argparse.ArgumentParser(
        description="Compare optimized generalized states/controls against logs."
    )
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--results", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--per-page", type=int, default=12)
    parser.add_argument("--start-knot", type=int, default=0)
    parser.add_argument("--end-knot", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config).expanduser())
    results_dir = Path(args.results).expanduser() if args.results else cfg["results"]
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_info = urdf_model_info(require_file(cfg["urdf"]))
    q_labels, v_labels, u_labels = labels_for_model(model_info)

    xs_log = load_csv_2d(results_dir / cfg["xs_log"])
    xs_opt = load_csv_2d(results_dir / cfg["xs_results"])
    us_log = load_csv_2d(results_dir / cfg["us_log"])
    us_opt = load_csv_2d(results_dir / cfg["us_results"])

    q_log, v_log, n_identified = state_slices(xs_log, model_info)
    q_opt, v_opt, _ = state_slices(xs_opt, model_info)

    state_n = min(q_log.shape[0], q_opt.shape[0], v_log.shape[0], v_opt.shape[0])
    ctrl_n = min(us_log.shape[0], us_opt.shape[0])
    end = args.end_knot if args.end_knot is not None else max(state_n, ctrl_n)
    start = max(args.start_knot, 0)
    if start >= end:
        raise ValueError("start-knot must be smaller than end-knot.")

    q_log = q_log[start:min(end, state_n)]
    q_opt = q_opt[start:min(end, state_n)]
    v_log = v_log[start:min(end, state_n)]
    v_opt = v_opt[start:min(end, state_n)]
    us_log = us_log[start:min(end, ctrl_n)]
    us_opt = us_opt[start:min(end, ctrl_n)]

    state_x = np.arange(start, start + q_log.shape[0])
    ctrl_x = np.arange(start, start + us_log.shape[0])

    pdf_path = output_dir / "state_control_comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        plot_group(pdf, output_dir, "q", "Generalized Position q", state_x,
                   q_log, q_opt, q_labels, args.per_page)
        plot_group(pdf, output_dir, "v", "Generalized Velocity v", state_x,
                   v_log, v_opt, v_labels, args.per_page)
        plot_group(pdf, output_dir, "u", "Control u", ctrl_x,
                   us_log, us_opt, u_labels, args.per_page)

    print(f"Wrote {pdf_path}")
    print(
        "Compared "
        f"q/v rows={q_log.shape[0]}, u rows={us_log.shape[0]}, "
        f"nq={model_info['nq']}, nv={model_info['nv']}, "
        f"identified_links={n_identified}"
    )


if __name__ == "__main__":
    main()
