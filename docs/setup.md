# Environment Setup

This document records how the EarlyMule-TG environments were built, with exact versions, so that every result in the thesis can be reproduced. It is updated whenever an environment changes.

| Environment | Where | Used for | Status |
|---|---|---|---|
| AMLSim | WSL2 (Ubuntu) on the laptop | Generating the synthetic dataset D1 | Ready (2026-10-19) |
| Main Python | WSL2 on the laptop, CPU only | Data processing, features, tabular baselines, tests | Ready (2026-10-31) |
| GPU | Kaggle notebook | Temporal GNN training | Planned |

The laptop GPU is AMD (no CUDA), so GPU work runs on Kaggle.

---

## 1. Host machine

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| WSL | WSL2, kernel 6.6.87.2-microsoft-standard-WSL2 |
| Linux distribution | Ubuntu 24.04.4 LTS |
| CPU / RAM available to WSL | 16 logical cores / 11 GB |
| Repository path | Windows `D:\earlymule-tg`, WSL `/mnt/d/earlymule-tg` |

## 2. AMLSim environment

AMLSim (IBM) needs Java 8, Maven and Python 3.7 with old libraries (`networkx 1.11`), and its pipeline is driven by bash scripts. It therefore runs inside WSL, isolated from the main Python environment.

### 2.1. Component versions

| Component | Version | Source |
|---|---|---|
| AMLSim | branch `earlymule-fixes` on top of upstream commit `7338a4b` (2022-02-14) | https://github.com/Verzol/AMLSim (fork of https://github.com/IBM/AMLSim) |
| Java | OpenJDK 1.8.0_504 | apt `openjdk-8-jdk` |
| Maven | 3.8.7 | apt `maven` |
| Graphviz | 2.43.0 | apt `graphviz`, `libgraphviz-dev` |
| Conda | Miniforge, conda 26.7.2 | https://github.com/conda-forge/miniforge |
| Python (env `amlsim`) | 3.7.12 | conda-forge |
| numpy | 1.21.6 | pip |
| networkx | 1.11 | pip |
| matplotlib | 3.5.3 | pip |
| pygraphviz | 1.7 | pip |
| powerlaw | 1.5 | pip |
| python-dateutil | 2.9.0.post0 | pip |
| Faker | 18.13.0 | pip |
| MASON | 20, built from source (commit `b799a4a`), `sim.engine` only | https://github.com/eclab/mason |

AMLSim is no longer maintained (last commit February 2022). All data is generated from the `earlymule-fixes` branch of the fork, which starts from that last upstream commit and changes the fan-in/fan-out schedule, how alert members are chosen and the ordering of cycle transactions. What was changed and why is explained in `EARLYMULE-TG.md` at the root of the fork; the full diff is at https://github.com/Verzol/AMLSim/compare/7338a4b...earlymule-fixes.

### 2.2. Install steps

All commands run inside WSL.

**System packages**
```bash
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk maven graphviz libgraphviz-dev unzip
java -version      # openjdk version "1.8.0_504"
mvn -v             # Apache Maven 3.8.7, Java version 1.8.0_504
```

**Conda and the `amlsim` environment**
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
bash /tmp/mf.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
# open a new shell
conda create -y -n amlsim python=3.7 pip
conda activate amlsim
```

**AMLSim source**

AMLSim is a Git submodule of this repository at `AMLSim/`, pinned to the fork commit used to generate the data. Check it out with `core.autocrlf=false`: on a Windows checkout with `autocrlf=true` the `.sh` scripts get CRLF line endings and fail in WSL with `$'\r': command not found`.
```bash
cd /mnt/d/earlymule-tg
git -c core.autocrlf=false submodule update --init AMLSim
git -C AMLSim config core.autocrlf false   # keep LF on later checkouts
pip install -r AMLSim/requirements.txt
```

**MASON dependency**

The AMLSim README points to `cs.gmu.edu/~eclab/projects/mason/` for `mason.20.jar`. As of 2026-10-19 that link no longer serves the file, and the `v20` release on GitHub has no binary attached. AMLSim only imports three MASON classes (`sim.engine.Schedule`, `sim.engine.SimState`, `sim.engine.Steppable`), so the jar is built from the `v20` source, compiling `sim.engine` and the classes it depends on:
```bash
git clone --depth 1 --branch v20 https://github.com/eclab/mason.git /tmp/masonsrc
mkdir -p /tmp/masonbin
cd /tmp/masonsrc/mason
javac -encoding UTF-8 -source 8 -target 8 -d /tmp/masonbin \
  -sourcepath src/main/java src/main/java/sim/engine/*.java
cd /tmp/masonbin && jar cf /mnt/d/earlymule-tg/AMLSim/jars/mason.20.jar .

cd /mnt/d/earlymule-tg/AMLSim
mvn install:install-file -Dfile=jars/mason.20.jar -DgroupId=mason \
  -DartifactId=mason -Dversion=20 -Dpackaging=jar -DgeneratePom=true
```

**Build AMLSim**
```bash
cd /mnt/d/earlymule-tg/AMLSim
bash scripts/build_AMLSim.sh     # BUILD SUCCESS, target/amlsim-1.0.0.jar (about 30 s)
```

### 2.3. Running AMLSim

```bash
cd /mnt/d/earlymule-tg/AMLSim
conda activate amlsim
python scripts/transaction_graph_generator.py conf.json   # 1. generate graph from paramFiles
bash scripts/run_AMLSim.sh conf.json                      # 2. run the Java simulator
python scripts/convert_logs.py conf.json                  # 3. convert logs to CSV outputs
```
Outputs are written to `AMLSim/outputs/` (intermediate files to `AMLSim/tmp/`); both are ignored by the fork's `.gitignore`, so generated data never reaches either repository.

### 2.4. Known issues

| Issue | Cause | Fix |
|---|---|---|
| `mvn` reports `JAVA_HOME` not defined | WSL inherits the Windows `PATH`, which contains `/mnt/c/maven/bin` | Install Maven inside WSL; `/usr/bin/mvn` comes first in `PATH` |
| `$'\r': command not found` when running scripts | CRLF line endings from a Windows checkout | Clone with `-c core.autocrlf=false` |
| `mason.20.jar` cannot be downloaded | Original download page removed | Build from source, see 2.2 |
| WSL paths become `C:/Program Files/Git/mnt/...` | Calling `wsl` from Git Bash triggers MSYS path conversion | Prefix the call with `MSYS_NO_PATHCONV=1`, or run from a WSL shell |

## 3. Main Python environment

All project code (ground truth, features, LightGBM, metrics, tests) runs in the conda env `earlymule`, defined in `environment.yml` at the repository root. It lives in WSL next to the `amlsim` env: conda is already installed there, and Kaggle also runs Linux, so behaviour matches between the laptop and the GPU notebook. It is CPU only; torch and PyTorch Geometric are not installed here because graph models are trained on Kaggle (section 4).

`environment.yml` holds lower bounds only. The versions below were installed on 2026-10-31.

### 3.1. Component versions

| Component | Version |
|---|---|
| Conda | 26.7.2 (same Miniforge as section 2) |
| Python | 3.12.16 |
| numpy | 2.5.4 |
| pandas | 3.1.0 |
| pyarrow | 25.0.0 |
| scikit-learn | 1.9.1 |
| LightGBM | 4.7.0 |
| PyYAML | 6.0.3 |
| matplotlib | 3.11.2 |
| JupyterLab | 4.6.5 |
| pytest | 9.2.0 |
| earlymule (this repo) | 0.1.0, editable install of `src/earlymule` |

### 3.2. Install steps

Inside WSL, from the repository root:
```bash
cd /mnt/d/earlymule-tg
conda env create -f environment.yml    # also runs `pip install -e .`
conda activate earlymule
pytest                                 # 13 passed
```
After editing `environment.yml`:
```bash
conda env update -f environment.yml --prune
```

### 3.3. Data locations

Code never hard-codes machine paths. `earlymule.config.load_paths()` reads `configs/paths.yaml`, where paths are relative to the repository root, and then applies `configs/local.yaml` if that file exists. `configs/local.yaml` is git-ignored and overrides only the keys it defines, for example on Kaggle:
```yaml
amlsim_outputs: /kaggle/input/earlymule-amlsim
processed: /kaggle/working/processed
```
The repository root is taken from the package location; set `EARLYMULE_ROOT` to override it.

## 4. Kaggle GPU environment (planned)

To be filled in once set up: notebook link, GPU type, CUDA, PyTorch and PyG versions.
