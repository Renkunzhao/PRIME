#!/usr/bin/env python3
"""Plot MHE inertia-parameter convergence from mhe_summary.csv."""

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/prime_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
DEFAULT_SUMMARY = EXPERIMENT_DIR / "results" / "mhe_summary.csv"
DEFAULT_GT = EXPERIMENT_DIR / "data" / "inertial_delta.csv"
LOG_CHOLESKY_NAMES = ["alpha", "d1", "d2", "d3", "s12", "s23", "s13", "t1", "t2", "t3"]


def log_cholesky_to_dynamic(theta):
    if len(theta) != 10:
        raise ValueError("Each log-Cholesky parameter block must have size 10.")

    alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3 = theta
    ed1 = math.exp(d1)
    ed2 = math.exp(d2)
    ed3 = math.exp(d3)
    scale = math.exp(2.0 * alpha)

    return [
        scale,
        scale * t1,
        scale * t2,
        scale * t3,
        scale * (s23 * s23 + t2 * t2 + t3 * t3 + ed2 * ed2 + ed3 * ed3),
        scale * (-s12 * ed2 - s13 * s23 - t1 * t2),
        scale
        * (s12 * s12 + s13 * s13 + t1 * t1 + t3 * t3 + ed1 * ed1 + ed3 * ed3),
        scale * (-s13 * ed3 - t1 * t3),
        scale * (-s23 * ed3 - t2 * t3),
        scale
        * (s12 * s12 + s13 * s13 + s23 * s23 + t1 * t1 + t2 * t2 + ed1 * ed1 + ed2 * ed2),
    ]


def dynamic_to_plot_fields(dyn):
    mass = dyn[0]
    com = [math.nan, math.nan, math.nan]
    if abs(mass) > 0.0:
        com = [dyn[1] / mass, dyn[2] / mass, dyn[3] / mass]
    return {
        "mass": mass,
        "Ixx": dyn[4],
        "Iyy": dyn[6],
        "Izz": dyn[9],
        "com_x": com[0],
        "com_y": com[1],
        "com_z": com[2],
    }


def read_log_cholesky_csv(path):
    if path is None or not path.exists():
        return None
    with path.open("r", newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            values = [float(value) for value in row if value.strip()]
            if len(values) == 10:
                theta = values
            elif len(values) >= 11:
                theta = values[1:11]
            else:
                raise ValueError(
                    f"GT file {path} must contain either 10 log-Cholesky values "
                    "or one leading timestamp/index followed by 10 values."
                )
            return theta
    raise ValueError(f"GT file {path} is empty.")


def parse_vector(text):
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def infer_initial_log_cholesky(summary_path, data):
    summary_dir = summary_path.parent
    first_window = data["window"][0]
    first_start = data["start_idx"][0]
    candidates = [
        summary_dir
        / f"window_{first_window:03d}_start_{first_start:04d}"
        / "inertia_identification.txt"
    ]
    candidates.extend(sorted(summary_dir.glob("window_*_start_*/inertia_identification.txt")))

    for report_path in candidates:
        if not report_path.exists():
            continue
        with report_path.open("r") as stream:
            for line in stream:
                line = line.strip()
                if line.startswith("initial_log_cholesky = [") and line.endswith("]"):
                    vector_text = line.split("[", 1)[1].rsplit("]", 1)[0]
                    theta = parse_vector(vector_text)
                    if len(theta) != 10:
                        raise ValueError(
                            f"Expected 10 initial log-Cholesky values in {report_path}."
                        )
                    return theta, report_path
    raise ValueError(
        "Could not infer nominal initial_log_cholesky from window reports. "
        "Rerun the MHE executable with inertia reports enabled, or use --gt-mode absolute."
    )


def prepare_gt(path, mode, summary_path, data):
    theta = read_log_cholesky_csv(path)
    if theta is None:
        return None, None
    if mode == "absolute":
        return dynamic_to_plot_fields(log_cholesky_to_dynamic(theta)), None
    base_theta, base_path = infer_initial_log_cholesky(summary_path, data)
    theta_absolute = [base + delta for base, delta in zip(base_theta, theta)]
    return dynamic_to_plot_fields(log_cholesky_to_dynamic(theta_absolute)), base_path


def read_summary(path, start_window=None, end_window=None, start_idx=None, end_idx=None):
    with path.open("r", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {path}.")

    ranges = [
        ("window", start_window, end_window),
        ("start_idx", start_idx, end_idx),
    ]
    for label, start, end in ranges:
        if start is not None and end is not None and start > end:
            raise ValueError(f"{label} filter start ({start}) must be <= end ({end}).")

    selected_rows = []
    for row in rows:
        window = int(row["window"])
        raw_start_idx = int(row["start_idx"])
        if start_window is not None and window < start_window:
            continue
        if end_window is not None and window > end_window:
            continue
        if start_idx is not None and raw_start_idx < start_idx:
            continue
        if end_idx is not None and raw_start_idx > end_idx:
            continue
        selected_rows.append(row)

    rows = selected_rows
    if not rows:
        raise ValueError("The requested plot range did not match any MHE summary rows.")

    theta_names = sorted(
        (name for name in rows[0] if name.startswith("theta_")),
        key=lambda name: int(name.split("_", 1)[1]),
    )
    if not theta_names or len(theta_names) % 10 != 0:
        raise ValueError("mhe_summary.csv must contain theta_0...theta_N columns in 10D blocks.")

    n_links = len(theta_names) // 10
    data = {
        "window": [int(row["window"]) for row in rows],
        "start_idx": [int(row["start_idx"]) for row in rows],
        "mass_estimate": [float(row["mass_estimate"]) for row in rows],
        "links": [],
        "arrival_prior_diag": [],
        "marginal_hessian_diag": [],
    }

    for prefix in ("arrival_prior_diag", "marginal_hessian_diag"):
        diag_names = sorted(
            (name for name in rows[0] if name.startswith(f"{prefix}_")),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        if diag_names:
            data[prefix] = [
                [float(row[name]) for name in diag_names]
                for row in rows
            ]

    for link_id in range(n_links):
        link = {"mass": [], "Ixx": [], "Iyy": [], "Izz": [], "com_x": [], "com_y": [], "com_z": []}
        offset = 10 * link_id
        for row in rows:
            theta = [float(row[f"theta_{offset + i}"]) for i in range(10)]
            fields = dynamic_to_plot_fields(log_cholesky_to_dynamic(theta))
            for name, value in fields.items():
                link[name].append(value)
        data["links"].append(link)

    return data


def plot_convergence(data, output_path, x_key, gt=None):
    x = data[x_key]
    x_label = "window" if x_key == "window" else "start index"
    n_links = len(data["links"])

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
    fig.suptitle("Go2 Kunzhao MHE Inertia Convergence", fontsize=14)

    if n_links == 1:
        axes[0].plot(x, data["links"][0]["mass"], marker="o", linewidth=1.8, label="mass")
        if gt is not None:
            axes[0].axhline(
                gt["mass"],
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.75,
                label="GT mass",
            )
    else:
        axes[0].plot(x, data["mass_estimate"], color="black", linewidth=2.0, label="total mass")
        for link_id, link in enumerate(data["links"]):
            axes[0].plot(x, link["mass"], marker="o", linewidth=1.2, label=f"link {link_id} mass")
    axes[0].set_ylabel("mass [kg]")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    inertia_labels = [("Ixx", "tab:blue"), ("Iyy", "tab:orange"), ("Izz", "tab:green")]
    for link_id, link in enumerate(data["links"]):
        prefix = "" if n_links == 1 else f"link {link_id} "
        for name, color in inertia_labels:
            axes[1].plot(
                x,
                link[name],
                marker="o",
                linewidth=1.5,
                color=color if n_links == 1 else None,
                label=f"{prefix}{name}",
            )
    if gt is not None and n_links == 1:
        for name, color in inertia_labels:
            axes[1].axhline(
                gt[name],
                color=color,
                linestyle="--",
                linewidth=1.3,
                alpha=0.75,
                label=f"GT {name}",
            )
    axes[1].set_ylabel("rotational inertia diag [kg m^2]")
    axes[1].legend(loc="best", ncol=3)
    axes[1].grid(True, alpha=0.25)

    com_labels = [("com_x", "x", "tab:red"), ("com_y", "y", "tab:purple"), ("com_z", "z", "tab:brown")]
    for link_id, link in enumerate(data["links"]):
        prefix = "" if n_links == 1 else f"link {link_id} "
        for name, axis_name, color in com_labels:
            axes[2].plot(
                x,
                link[name],
                marker="o",
                linewidth=1.5,
                color=color if n_links == 1 else None,
                label=f"{prefix}com_{axis_name}",
            )
    if gt is not None and n_links == 1:
        for name, axis_name, color in com_labels:
            axes[2].axhline(
                gt[name],
                color=color,
                linestyle="--",
                linewidth=1.3,
                alpha=0.75,
                label=f"GT com_{axis_name}",
            )
    axes[2].set_ylabel("COM position [m]")
    axes[2].set_xlabel(x_label)
    axes[2].legend(loc="best", ncol=3)
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    return pdf_path


def plot_hessian_diagonal(data, output_path, x_key, source):
    diag = data[source]
    if not diag:
        raise ValueError(
            f"mhe_summary.csv does not contain {source}_* columns. "
            "Rerun go2_sim_kunzhao_mhe with the updated logger."
        )

    x = data[x_key]
    x_label = "window" if x_key == "window" else "start index"
    n_cols = len(diag[0])
    n_links = n_cols // 10 if n_cols % 10 == 0 else 1
    title_source = (
        "Arrival Prior Hessian Diagonal"
        if source == "arrival_prior_diag"
        else "Marginal Hessian Diagonal Carried To Next Prior"
    )

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    fig.suptitle(f"Go2 Kunzhao MHE {title_source}", fontsize=14)

    for col in range(n_cols):
        link_id = col // 10
        local_id = col % 10
        label = LOG_CHOLESKY_NAMES[local_id] if n_links == 1 else f"link {link_id} {LOG_CHOLESKY_NAMES[local_id]}"
        y = [row[col] for row in diag]
        ax.plot(x, y, marker="o", linewidth=1.3, label=label)

    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("diagonal information")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", ncol=2 if n_cols <= 10 else 3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot mass, inertia diagonal, and COM convergence from Go2 Kunzhao MHE summary output."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"path to mhe_summary.csv (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output image path (default: <summary_dir>/inertia_convergence.png)",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=DEFAULT_GT,
        help=(
            "ground-truth log-Cholesky CSV to overlay; accepts 10 values or "
            f"timestamp/index plus 10 values (default: {DEFAULT_GT})"
        ),
    )
    parser.add_argument(
        "--gt-mode",
        choices=("delta", "absolute"),
        default="delta",
        help=(
            "interpret --gt as a log-Cholesky delta from the nominal initial "
            "torso inertia, or as absolute log-Cholesky parameters"
        ),
    )
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="disable the ground-truth overlay",
    )
    parser.add_argument(
        "--hessian-output",
        type=Path,
        default=None,
        help="output path for prior Hessian diagonal plot (default: <summary_dir>/prior_hessian_diagonal.png)",
    )
    parser.add_argument(
        "--hessian-source",
        choices=("marginal", "arrival"),
        default="marginal",
        help="plot marginal Hessian carried to the next prior, or the arrival prior used by each current window",
    )
    parser.add_argument(
        "--skip-hessian",
        action="store_true",
        help="only write the inertia convergence plot",
    )
    parser.add_argument(
        "--x",
        choices=("window", "start_idx"),
        default="window",
        help="horizontal axis for the convergence plot",
    )
    parser.add_argument(
        "--start-window",
        type=int,
        default=None,
        help="first MHE window to include",
    )
    parser.add_argument(
        "--end-window",
        type=int,
        default=None,
        help="last MHE window to include",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="first raw-data start_idx to include",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="last raw-data start_idx to include",
    )
    args = parser.parse_args()

    summary = args.summary.resolve()
    output = args.output
    if output is None:
        output = summary.parent / "inertia_convergence.png"
    output = output.resolve()
    hessian_output = args.hessian_output
    if hessian_output is None:
        hessian_output = summary.parent / "prior_hessian_diagonal.png"
    hessian_output = hessian_output.resolve()

    data = read_summary(
        summary,
        start_window=args.start_window,
        end_window=args.end_window,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
    gt, gt_base_path = (
        (None, None)
        if args.no_gt
        else prepare_gt(args.gt.resolve(), args.gt_mode, summary, data)
    )
    print(
        "Plotting rows:",
        f"{len(data['window'])}",
        f"window={data['window'][0]}..{data['window'][-1]}",
        f"start_idx={data['start_idx'][0]}..{data['start_idx'][-1]}",
    )
    if gt is not None:
        print(f"Overlaying GT from {args.gt.resolve()} as {args.gt_mode}")
        if gt_base_path is not None:
            print(f"GT delta base: {gt_base_path}")
    pdf_path = plot_convergence(data, output, args.x, gt=gt)
    print(f"Wrote {output}")
    print(f"Wrote {pdf_path}")
    if not args.skip_hessian:
        hessian_source = (
            "arrival_prior_diag"
            if args.hessian_source == "arrival"
            else "marginal_hessian_diag"
        )
        if data[hessian_source]:
            hessian_pdf_path = plot_hessian_diagonal(
                data, hessian_output, args.x, hessian_source
            )
            print(f"Wrote {hessian_output}")
            print(f"Wrote {hessian_pdf_path}")
        else:
            print(
                f"Skipped Hessian plot: {summary} has no {hessian_source}_* columns. "
                "Rerun go2_sim_kunzhao_mhe with the updated logger."
            )


if __name__ == "__main__":
    main()
