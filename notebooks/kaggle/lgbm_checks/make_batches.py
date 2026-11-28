"""Generate one Kaggle CPU notebook folder per batch of the LightGBM checks of the replan (main dataset).

Jobs are (output tag, extra arguments of lightgbm_export, seed): 1% of labelled cases on seeds 3-9 (seeds 0-2 are
the label-scarcity runs), shuffled labels on seed 0, and each of the five typologies left out of training on
seeds 0-2. Each folder gets the template notebook with its own parameter cell and a kernel-metadata.json
(id verzol/earlymule-lgbm-chk-<k>, CPU, the 80K export and the case map as inputs).
Writes notebooks/kaggle/lgbm_checks/batches/<k>/ (git-ignored).
Run: python notebooks/kaggle/lgbm_checks/make_batches.py --commit <sha> [--per-batch 6]
Push: kaggle kernels push -p notebooks/kaggle/lgbm_checks/batches/<k>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = "80K-earlymule-mule"
# Mule typologies of the training cases (case map column `typology`)
TYPOLOGIES = ["cycle", "mule_collect_disperse", "mule_collect_exit", "mule_fan_in_out", "mule_relay_chain"]
CASES = ["--label-cases", "{cases}"]


def jobs() -> list[tuple[str, list[str], int]]:
    """Every check run, the cheap and most-read ones first."""
    out = [("lgbm_shuffled", ["--shuffle-labels"], 0)]
    out += [("lgbm_lf0.01", ["--label-fraction", "0.01", *CASES, "--unlabeled", "negative"], s) for s in range(3, 10)]
    out += [(f"lgbm_wo_{t}", ["--leave-out-typology", t, *CASES], s) for t in TYPOLOGIES for s in range(3)]
    return out


def main() -> None:
    """Write the notebook folder of every batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True, help="commit the notebooks check out (pushed to GitHub)")
    parser.add_argument("--per-batch", type=int, default=6)
    args = parser.parse_args()

    # 1. Template notebook and its parameter cell
    template = json.loads((HERE / "lgbm_checks.ipynb").read_text(encoding="utf-8"))
    param_cell = next(i for i, c in enumerate(template["cells"]) if c.get("id") == "cell-params")
    header = template["cells"][param_cell]["source"][0]

    # 2. One folder per batch: notebook with its jobs, and CPU kernel metadata
    todo = jobs()
    for k, start in enumerate(range(0, len(todo), args.per_batch), start=1):
        batch = todo[start:start + args.per_batch]
        folder = HERE / "batches" / str(k)
        folder.mkdir(parents=True, exist_ok=True)
        nb = json.loads(json.dumps(template))
        nb["cells"][param_cell]["source"] = [header, f"DATASET = {DATASET!r}\n", f"JOBS = {batch!r}\n",
                                             f"COMMIT = {args.commit!r}"]
        (folder / "lgbm_checks.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        kid = f"earlymule-lgbm-chk-{k}"
        meta = {"id": f"verzol/{kid}", "title": f"EarlyMule lgbm chk {k}", "code_file": "lgbm_checks.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "false",
                "enable_tpu": "false", "enable_internet": "true",
                "dataset_sources": [f"verzol/earlymule-{DATASET.lower()}", "verzol/earlymule-80k-label-cases"],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(k, [(t, s) for t, _, s in batch])


if __name__ == "__main__":
    main()
