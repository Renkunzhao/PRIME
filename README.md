# PRIME Moving Window Development Branch

This branch is under development for moving window / online-style Go2
inertial-parameter identification.

## Build

Default build:

```bash
cmake -S . -B build_codex
cmake --build build_codex --target go2_sim_kunzhao_long_mhe -j8
```

OpenMP build for node-level `calc/calcDiff` parallelism:

```bash
cmake -S . -B build_codex_omp \
  -DBUILD_WITH_MULTITHREADS=ON \
  -DBUILD_WITH_NTHREADS=8
cmake --build build_codex_omp --target go2_sim_kunzhao_long_mhe -j8
```

The XML `n_thread` option only matters when the binary is built with
`BUILD_WITH_MULTITHREADS=ON`.

## Run Moving Window

```bash
build_codex_omp/experiments/Go2_sim_kunzhao_long_mhe/go2_sim_kunzhao_long_mhe \
  experiments/Go2_sim_kunzhao_long_mhe/config/Go2_sim_kunzhao_long_mhe.xml
```

To parse the config without solving, temporarily set `dry_run="true"` in the
`<solver>` block and run the same command.

Useful XML knobs:

```xml
<solver
  horizon="300"
  down_sample="2"
  start_idx="10000"
  n_thread="8" />

<moving_horizon
  max_windows="200"
  stride_knots="50"
  information_forgetting_enabled="false"
  information_forgetting_factor="0.99" />

<outputs
  log_initial_guess="false"
  rollout="false"
  save_window_outputs="true"
  save_debug_marginals="false"
  save_force_rollout="false"
  save_every_n_windows="1" />
```

For fast sweeps, keep `callbacks`, `save_debug_marginals`, and
`save_force_rollout` disabled. For window visualizers, rerun selected windows
with `log_initial_guess="true"`; MeshCat force arrows also need
`save_force_rollout="true"`.

Main outputs:

```text
experiments/Go2_sim_kunzhao_long_mhe/results/mhe_summary.csv
experiments/Go2_sim_kunzhao_long_mhe/results/timing_summary.csv
experiments/Go2_sim_kunzhao_long_mhe/results/window_###_start_####/
```

## Visualize Convergence

```bash
python3 experiments/Go2_sim_kunzhao_long_mhe/visualizer/plot_inertia_convergence.py
```

Common options:

```bash
python3 experiments/Go2_sim_kunzhao_long_mhe/visualizer/plot_inertia_convergence.py \
  --x start_idx \
  --start-window 0 \
  --end-window 80
```

```bash
python3 experiments/Go2_sim_kunzhao_long_mhe/visualizer/plot_inertia_convergence.py \
  --hessian-source arrival
```

The plotter overlays `data/inertial_delta.csv` as ground truth when it exists.
Generated files are written beside the summary:

```text
results/inertia_convergence.png
results/inertia_convergence.pdf
results/prior_hessian_diagonal.png
results/prior_hessian_diagonal.pdf
```

## Visualize Raw Data

```bash
python3 experiments/Go2_sim_kunzhao_long_mhe/visualizer/meshcat_go2_data.py
```

Useful options:

```bash
python3 experiments/Go2_sim_kunzhao_long_mhe/visualizer/meshcat_go2_data.py \
  --check-only
```

```bash
python3 experiments/Go2_sim_kunzhao_long_mhe/visualizer/meshcat_go2_data.py \
  --all-data \
  --stride 10
```

```bash
python3 experiments/Go2_sim_kunzhao_long_mhe/visualizer/meshcat_go2_data.py \
  --compare-raw-order
```

## Visualize One Window

MeshCat replay:

```bash
python3 experiments/Go2_sim_kunzhao_long_mhe/visualizer/meshcat_go2_window.py \
  --results-root experiments/Go2_sim_kunzhao_long_mhe/results \
  --window 12
```

State/control plots:

```bash
python3 experiments/Go2_sim_kunzhao_long_mhe/visualizer/plot_state_control_window.py \
  --results-root experiments/Go2_sim_kunzhao_long_mhe/results \
  --window 12
```

Optional range:

```bash
python3 experiments/Go2_sim_kunzhao_long_mhe/visualizer/plot_state_control_window.py \
  --results-root experiments/Go2_sim_kunzhao_long_mhe/results \
  --window 12 \
  --start-knot 0 \
  --end-knot 80 \
  --per-page 12
```

These tools need per-window logs such as `xs_log.csv`, `us_log.csv`,
`xs_results_fddp.csv`, and `us_results_fddp.csv`.

## Other Moving Window Experiments

Use the same workflow with the corresponding folder and executable:

```text
experiments/Go2_sim_kunzhao_mhe          -> go2_sim_kunzhao_mhe
experiments/Go2_sim_m_+3kg_mhe          -> go2_sim_m_p3kg_mhe
experiments/Go2_sim_cz_-0.1m_mhe        -> go2_sim_cz_m0p1_mhe
experiments/Go2_sim_kunzhao_long_mhe    -> go2_sim_kunzhao_long_mhe
```

Moving window result folders are ignored by git.
