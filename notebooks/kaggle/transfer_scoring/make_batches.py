"""Generate one Kaggle CPU notebook folder per batch of the transfer-scoring runs (EMTG-72).

Splits the variants of configs/transfer_scoring.yaml x seeds into --batches notebooks of about equal size,
all seeds of a variant spread across batches so every batch mixes cheap and costly variants. Each folder gets
the template notebook with its own parameter cell and a kernel-metadata.json (id
verzol/earlymule-transfer-<k>, CPU, the transfer-task dataset as input).
Writes notebooks/kaggle/transfer_scoring/batches/<k>/ (git-ignored).
Run: python notebooks/kaggle/transfer_scoring/make_batches.py --commit <sha> [--batches 4] [--seeds 0 1 2]
Push: kaggle kernels push -p notebooks/kaggle/transfer_scoring/batches/<k>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Write the notebook folder of every batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True, help="commit the notebooks check out (pushed to GitHub)")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    # 1. Jobs seed-major, dealt round-robin, so the seeds of one variant land in different batches
    cfg = yaml.safe_load((HERE.parents[2] / "configs" / "transfer_scoring.yaml").read_text(encoding="utf-8"))
    jobs = [(v, s) for s in args.seeds for v in cfg["variants"]]
    batches = [jobs[k::args.batches] for k in range(args.batches)]
    template = json.loads((HERE / "transfer_scoring.ipynb").read_text(encoding="utf-8"))
    param_cell = next(i for i, c in enumerate(template["cells"]) if c.get("id") == "cell-params")
    header = template["cells"][param_cell]["source"][0]

    # 2. One folder per batch: notebook with its jobs, and CPU kernel metadata
    for k, batch in enumerate(batches, start=1):
        folder = HERE / "batches" / str(k)
        folder.mkdir(parents=True, exist_ok=True)
        nb = json.loads(json.dumps(template))
        nb["cells"][param_cell]["source"] = [header, f"JOBS = {batch!r}\n", f"COMMIT = {args.commit!r}"]
        (folder / "transfer_scoring.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        meta = {"id": f"verzol/earlymule-transfer-{k}", "title": f"EarlyMule transfer {k}",  # title must slug to the id
                "code_file": "transfer_scoring.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": "true", "enable_gpu": "false", "enable_tpu": "false", "enable_internet": "true",
                "dataset_sources": ["verzol/earlymule-80k-transfer-task"], "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(k, batch)


if __name__ == "__main__":
    main()
