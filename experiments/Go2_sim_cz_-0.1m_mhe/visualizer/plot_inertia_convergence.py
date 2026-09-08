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
            dyn = log_cholesky_to_dynamic(theta)
            mass = dyn[0]
            com = [math.nan, math.nan, math.nan]
            if abs(mass) > 0.0:
                com = [dyn[1] / mass, dyn[2] / mass, dyn[3] / mass]
            link["mass"].append(mass)
            link["Ixx"].append(dyn[4])
            link["Iyy"].append(dyn[6])
            link["Izz"].append(dyn[9])
            link["com_x"].append(com[0])
            link["com_y"].append(com[1])
            link["com_z"].append(com[2])
        data["links"].append(link)

    return data


def plot_convergence(data, output_path, x_key):
    x = data[x_key]
    x_label = "window" if x_key == "window" else "start index"
    n_links = len(data["links"])

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
    fig.suptitle("Go2 Shifted-COM MHE Inertia Convergence", fontsize=14)

    if n_links == 1:
        axes[0].plot(x, data["links"][0]["mass"], marker="o", linewidth=1.8, label="mass")
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
            "Rerun the shifted-COM MHE executable with the updated logger."
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
    fig.suptitle(f"Go2 Shifted-COM MHE {title_source}", fontsize=14)

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
        description="Plot mass, inertia diagonal, and COM convergence from Go2 shifted-COM MHE summary output."
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
    print(
        "Plotting rows:",
        f"{len(data['window'])}",
        f"window={data['window'][0]}..{data['window'][-1]}",
        f"start_idx={data['start_idx'][0]}..{data['start_idx'][-1]}",
    )
    pdf_path = plot_convergence(data, output, args.x)
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
                "Rerun the shifted-COM MHE executable with the updated logger."
            )


if __name__ == "__main__":
    main()
