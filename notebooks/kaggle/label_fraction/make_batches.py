"""Generate one Kaggle CPU notebook folder per batch of the LightGBM label-scarcity runs.

Jobs are (label fraction, unlabeled mode, seed); all seeds of one setting stay in one batch. Each folder gets the
template notebook with its own parameter cell and a kernel-metadata.json (id verzol/earlymule-lgbm-lf-<k>, CPU,
the 80K export and the case map as inputs).
Writes notebooks/kaggle/label_fraction/batches/<k>/ (git-ignored).
Run: python notebooks/kaggle/label_fraction/make_batches.py --commit <sha> [--fractions 0.01 0.05 0.1 0.25]
     [--modes negative drop] [--seeds 0 1 2] [--per-batch 6] [--first-id 1]
Push: kaggle kernels push -p notebooks/kaggle/label_fraction/batches/<k>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Write the notebook folder of every batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True, help="commit the notebooks check out (pushed to GitHub)")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.25])
    parser.add_argument("--modes", nargs="+", default=["negative", "drop"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--per-batch", type=int, default=6)
    parser.add_argument("--first-id", type=int, default=1)
    args = parser.parse_args()

    # 1. Jobs, main mode first; template notebook and its parameter cell
    jobs = [(f, m, s) for m in args.modes for f in args.fractions for s in args.seeds]
    template = json.loads((HERE / "lgbm_label_fraction.ipynb").read_text(encoding="utf-8"))
    param_cell = next(i for i, c in enumerate(template["cells"]) if c.get("id") == "cell-params")
    header = template["cells"][param_cell]["source"][0]

    # 2. One folder per batch: notebook with its jobs, and CPU kernel metadata
    for k, start in enumerate(range(0, len(jobs), args.per_batch), start=args.first_id):
        batch = jobs[start:start + args.per_batch]
        folder = HERE / "batches" / str(k)
        folder.mkdir(parents=True, exist_ok=True)
        nb = json.loads(json.dumps(template))
        nb["cells"][param_cell]["source"] = [header, f"JOBS = {batch!r}\n", f"COMMIT = {args.commit!r}"]
        (folder / "lgbm_label_fraction.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        meta = {"id": f"verzol/earlymule-lgbm-lf-{k}", "title": f"EarlyMule LGBM lf {k}",  # title must slug to the id
                "code_file": "lgbm_label_fraction.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": "true", "enable_gpu": "false", "enable_tpu": "false", "enable_internet": "true",
                "dataset_sources": ["verzol/earlymule-80k-earlymule-mule", "verzol/earlymule-80k-label-cases"],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(k, batch)


if __name__ == "__main__":
    main()
