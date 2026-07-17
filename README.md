# Learned Multi-Profile Planner Outcomes for Risk-Budgeted Multi-UAV Assignment

This repository implements a learned multi-profile planner-outcome interface
for risk-budgeted multi-UAV assignment. A shared predictor supplies route
length and additive exposure for three planner profiles, avoiding candidate
planning under every profile. Controlled allocation experiments test whether
the predicted portfolio retains the value of planner-generated alternatives,
while the reference planner still generates every route selected for
execution.

The repository accompanies the manuscript:

> *A Learned Interface for Multi-Profile Planner Outcomes in Risk-Budgeted
> Multi-UAV Assignment*

## What Is Included

- a deterministic SE(2) state-lattice planner used by the primary learned system;
- a structurally distinct grid risk planner used to repeat the profile-selection comparison;
- generation of map-disjoint edge datasets and fleet assignment benchmarks;
- shared and profile-specific route-outcome predictors;
- one-, two-, and three-profile portfolio comparisons;
- uniform and per-edge consumers for controlled downstream evaluation;
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
portfolio consumers evaluated in the paper.

## Repository Layout

```text
configs/       experiment configuration files
paper/         authoritative experiment protocol
scripts/       data generation, training, evaluation, audit, and figure scripts
src/           environments, planners, learning modules, and assignment utilities
```

## Data and Models

Paper datasets, trained checkpoints, and result files are archived in Zenodo:

> Zhouning Xu, Qiupeng Wu, and Bolin Chen. *Learned Multi-Profile
> Planner-Outcome Interface: Data, Models, and Results for Risk-Budgeted
> Multi-UAV Assignment*, version 1.0.1.
> Zenodo, 2026. https://doi.org/10.5281/zenodo.21357325

The archive includes a checksum manifest. The code can also regenerate all
assets from the locked protocol.

## License

Source code is released under the MIT License. Manuscript text and paper figures
are not covered by that software license unless stated otherwise.
