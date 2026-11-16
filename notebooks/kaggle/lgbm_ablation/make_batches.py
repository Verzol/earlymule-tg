"""Generate one Kaggle CPU notebook folder per batch of the LightGBM feature ablation (EMTG-70).

Splits the 18 ablations of configs/lightgbm_ablation.yaml x seeds into batches of --per-batch runs; batch 1
holds the full feature set and no_churn, so the first notebook checks run time and the full-set result
before the rest is pushed. Each folder gets the template notebook with its own parameter cell and a
kernel-metadata.json (id verzol/earlymule-lgbm-abl-<k>, CPU, the 80K export as input).
Writes notebooks/kaggle/lgbm_ablation/batches/<k>/ (git-ignored).
Run: python notebooks/kaggle/lgbm_ablation/make_batches.py --commit <sha> [--per-batch 6] [--seeds 0 1 2]
     [--skip full no_churn] [--first-id 2]   (later batches: leave out ablations already run, keep kernel ids unique)
Push: kaggle kernels push -p notebooks/kaggle/lgbm_ablation/batches/<k>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
FIRST = ["full", "no_churn"]


def batches(ablations: list[str], seeds: list[int], per_batch: int,
            skip: tuple[str, ...] = ()) -> list[list[tuple[str, int]]]:
    """(ablation, seed) jobs in batches; the FIRST ablations lead, all seeds of an ablation stay together."""
    order = [a for a in FIRST + [a for a in ablations if a not in FIRST] if a not in skip]
    jobs = [(a, s) for a in order for s in seeds]
    return [jobs[i:i + per_batch] for i in range(0, len(jobs), per_batch)]


def main() -> None:
    """Write the notebook folder of every batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True, help="commit the notebooks check out (pushed to GitHub)")
    parser.add_argument("--per-batch", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip", nargs="*", default=[], help="ablations already run")
    parser.add_argument("--first-id", type=int, default=1, help="number of the first kernel id")
    args = parser.parse_args()

    # 1. Ablations from the config, template notebook and its parameter cell
    cfg = yaml.safe_load((HERE.parents[2] / "configs" / "lightgbm_ablation.yaml").read_text(encoding="utf-8"))
    template = json.loads((HERE / "lgbm_ablation.ipynb").read_text(encoding="utf-8"))
    param_cell = next(i for i, c in enumerate(template["cells"]) if c.get("id") == "cell-params")
    header = template["cells"][param_cell]["source"][0]

    # 2. One folder per batch: notebook with its jobs, and CPU kernel metadata
    for k, jobs in enumerate(batches(list(cfg["ablations"]), args.seeds, args.per_batch, tuple(args.skip)),
                             start=args.first_id):
        folder = HERE / "batches" / str(k)
        folder.mkdir(parents=True, exist_ok=True)
        nb = json.loads(json.dumps(template))
        nb["cells"][param_cell]["source"] = [header, f"JOBS = {jobs!r}\n", f"COMMIT = {args.commit!r}"]
        (folder / "lgbm_ablation.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        meta = {"id": f"verzol/earlymule-lgbm-abl-{k}", "title": f"EarlyMule LGBM abl {k}",  # title must slug to the id
                "code_file": "lgbm_ablation.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": "true", "enable_gpu": "false", "enable_tpu": "false", "enable_internet": "true",
                "dataset_sources": ["verzol/earlymule-80k-earlymule-mule"], "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(k, jobs)


if __name__ == "__main__":
    main()
