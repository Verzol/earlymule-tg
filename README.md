# EarlyMule-TG

**Early Mule-Account Detection through Multi-Scale Temporal Role Transitions in Digital Banking Transaction Graphs**

Undergraduate thesis · Giang Tuấn Minh · VNU University of Engineering and Technology (UET)

## Status

**Phase 1 · Requirements and data feasibility (2026-10-19 to 2026-11-01).** AMLSim runs end to end; its schedule and member-sampling issues were fixed in a fork, and the feasibility checks are in the EDA notebook.

This repository grows with the actual work: folders and files are added only when the corresponding task starts. The detailed plan and current tasks are tracked in Jira (project `EMTG`) and the Master Timeline sheet.

## Contents

| Path | Description |
|---|---|
| `docs/research_plan.md` | Research plan proposed by the thesis advisor; the basis for the implementation plan. Kept verbatim in its original language. |
| `docs/setup.md` | How every environment was built, with exact versions. |
| `notebooks/01_amlsim_eda.ipynb` | Exploration of AMLSim output and the data feasibility checks. |
| `AMLSim/` | Submodule: the AMLSim fork used to generate the synthetic dataset (see below). |
| `environment.yml` | Main conda environment `earlymule` (CPU): pandas, LightGBM, pytest. |
| `pyproject.toml` | Makes `src/earlymule` installable; pytest settings. |
| `src/earlymule/` | Project code: `config` (data locations), `data`, `features`, `models`, `evaluation`. |
| `configs/` | YAML configs. `paths.yaml` sets data locations; a git-ignored `local.yaml` overrides them per machine. |
| `tests/` | Unit tests, run with `pytest`. |
| `data/processed/` | Derived tables (ground truth, features); contents never committed. |

## Quick start

Inside WSL, after the AMLSim data has been generated (see `docs/setup.md`):

```bash
conda env create -f environment.yml
conda activate earlymule
pytest
```

## Third-party code

`AMLSim/` is a Git submodule pointing to [Verzol/AMLSim](https://github.com/Verzol/AMLSim), a fork of [IBM/AMLSim](https://github.com/IBM/AMLSim) (Apache License 2.0). The submodule is pinned to the exact commit used to generate the data.

- **What was changed and why:** [`EARLYMULE-TG.md`](https://github.com/Verzol/AMLSim/blob/earlymule-fixes/EARLYMULE-TG.md)
- **Full diff against upstream:** [`7338a4b...earlymule-fixes`](https://github.com/Verzol/AMLSim/compare/7338a4b...earlymule-fixes)
- **In the code:** every changed block carries an `EarlyMule-TG:` comment.

Clone with the submodule:

```bash
git clone --recurse-submodules https://github.com/Verzol/earlymule-tg.git
```

## Conventions

- Everything in this repository is written in English: identifiers, comments, docstrings, configs, logs, commit messages and branch names. The thesis itself is written in Vietnamese.
- Commit messages start with the Jira key, e.g. `EMTG-12: add point-in-time feature builder`.
- Raw and generated data are never committed (blocked in `.gitignore`).
