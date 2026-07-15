# Route-Profile Selection for Risk-Budgeted Multi-UAV Assignment

This repository contains the experiments for studying when a multi-UAV system
should choose among planner-generated route profiles. A frozen route planner
supplies route length and additive exposure under several profiles. The
experiments compare a profile fixed during development, one profile selected
for each dispatch, and profiles selected per edge during allocation. A shared
predictor avoids planning every candidate under every profile, while the
planner still generates all routes selected for execution.

The repository accompanies the manuscript:

> *Delaying Route-Profile Selection in Risk-Budgeted Multi-UAV Assignment with
> a Learned Planner Interface*

## What Is Included

- a deterministic SE(2) state-lattice planner used by the primary learned system;
- a structurally distinct grid risk planner used to repeat the profile-selection comparison;
- generation of map-disjoint edge datasets and fleet assignment benchmarks;
- shared and profile-specific route-outcome predictors;
- fixed, dispatch-wide, and per-edge profile selection;
- exact finite evaluation for 5x5 assignment and MILP evaluation for 10x10 and 20x20;
- static runtime accounting, rolling dispatch, statistical summaries, and figures.

The authoritative experiment protocol is
[`paper/protocol_lock.yaml`](paper/protocol_lock.yaml). Generated datasets,
models, caches, and result files are intentionally not stored in Git.

## Installation

Python 3.10 or newer is recommended. Create an isolated environment and install
the dependencies from the repository root:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Verification

The public smoke test requires no paper data or trained model. It checks the
environment representation, the grid planner under all three profiles, and the
assignment solver:

```bash
python scripts/smoke_public.py
```

Individual checks are also available:

```bash
python scripts/demo_env.py --no-show --save outputs/smoke/scenario.png
python scripts/smoke_grid_risk_planner.py
```

## Reproducing the Paper

The complete pipeline is expensive because it regenerates planner supervision
and fleet benchmarks. It is resume-safe and writes only under
`outputs/final_ras`:

```bash
python scripts/run_final_data_generation.py --workers 10
python scripts/run_final_ras_experiments.py --workers 10
```

Both runners skip completed stages and keep per-stage logs. The protocol uses
independent randomization namespaces for edge maps, edge pairs, assignment
maps, assignment points, and rolling missions; results are invariant to worker
count where specified in the protocol lock.

Main stages can also be run separately:

```bash
# Generate or replan edge supervision
python scripts/generate_final_edge_dataset.py --help
python scripts/replan_edge_dataset_profile.py --help

# Train the shared profile-conditioned predictor
python scripts/train_profile_onehot_cnn.py --help

# Generate fleet benchmarks
python scripts/run_sharded_assignment_benchmark.py --help
python scripts/replan_assignment_benchmark_profile.py --help

# Evaluate profile selection with predicted outcomes
python scripts/evaluate_route_portfolio_budget.py --help
python scripts/evaluate_learned_route_portfolio_budget.py --help
python scripts/evaluate_scale_assignment_milp.py --help

# Runtime, rolling missions, summaries, and figures
python scripts/profile_static_runtime_serial.py --help
python scripts/evaluate_lazy_rolling_stress.py --help
python scripts/summarize_profile_commitment.py --help
python scripts/make_ras_paper_figures_polished.py --help
```

## Data Semantics

The final edge data use fixed map-disjoint train, validation, and test splits.
Training scripts consume these splits and do not create a new random split.
The three profile datasets share map and start--goal geometry and differ only
in the frozen planner profile.

For assignment benchmarks, the near-edge policy is:

- Euclidean distance `<= 24`: zero incremental outcome;
- `24 < distance <= 120`: exact reference-planner outcome;
- distance `> 120`: learned or analytic interface prediction.

Exposure is additive along a route. The associated survival proxy is
`survival = exp(-exposure)`, so selected route exposures can be composed by the
profile-selection rules evaluated in the paper.

## Repository Layout

```text
configs/       experiment configuration files
paper/         authoritative experiment protocol
scripts/       data generation, training, evaluation, audit, and figure scripts
src/           environments, planners, learning modules, and assignment utilities
```

## Data and Models

Paper datasets, trained checkpoints, and result files are archived in Zenodo:

> Zhouning Xu, Qiupeng Wu, and Bolin Chen. *Route-Profile Selection Data,
> Models, and Results for Risk-Budgeted Multi-UAV Assignment*, version 1.0.0.
> Zenodo, 2026. https://doi.org/10.5281/zenodo.21357325

The archive includes a checksum manifest. The code can also regenerate all
assets from the locked protocol.

## License

Source code is released under the MIT License. Manuscript text and paper figures
are not covered by that software license unless stated otherwise.
