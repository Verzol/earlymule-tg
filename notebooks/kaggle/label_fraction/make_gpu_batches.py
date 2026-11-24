"""Generate the Kaggle GPU notebook folders of the label-scarcity runs of the graph models.

Each batch is a list of stages (module, config, label fraction or None for full labels, optional seeds); the stages of one batch run
one after another, the seeds of a stage in parallel on the two T4 GPUs. Batches are sized to stay well under the
12-hour session limit. Each folder gets the GPU template with its own parameter cell and a kernel-metadata.json
(id verzol/earlymule-tgn-lf-<k>, T4, the 80K export and the case map as inputs).
Writes notebooks/kaggle/label_fraction/batches_gpu/<k>/ (git-ignored).
Run: python notebooks/kaggle/label_fraction/make_gpu_batches.py --commit <sha>
Push: kaggle kernels push -p notebooks/kaggle/label_fraction/batches_gpu/<k>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TGN, STAGT, SAGE = "earlymule.models.tgn", "earlymule.models.stagt", "earlymule.models.graphsage"

# Hours per stage on Kaggle T4 at full labels (slowest seed): transition 3.4, TGN with features 2.8, STA-GT 0.5,
# GraphSAGE 0.1. GraphSAGE runs once with full labels and the 72-feature export (it last ran on the 43-feature one).
BATCHES = [
    [(TGN, "tgn_transition.yaml", 0.01), (TGN, "tgn_transition.yaml", 0.05)],
    [(TGN, "tgn_features.yaml", 0.01), (TGN, "tgn_features.yaml", 0.05), (STAGT, "stagt.yaml", 0.01)],
    [(TGN, "tgn_transition.yaml", 0.1), (STAGT, "stagt.yaml", 0.05), (SAGE, "graphsage.yaml", None)],
    [(TGN, "tgn_features.yaml", 0.1), (STAGT, "stagt.yaml", 0.1)],
    # Batches 2 and 3 ran two STA-GT seeds on one GPU and seed 0 ran out of memory; 0.1 seed 0 in case batch 4 does too
    [(STAGT, "stagt.yaml", 0.01, [0]), (STAGT, "stagt.yaml", 0.05, [0]), (STAGT, "stagt.yaml", 0.1, [0])],
]


def main() -> None:
    """Write the notebook folder of every batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True, help="commit the notebooks check out (pushed to GitHub)")
    args = parser.parse_args()

    # 1. Template notebook and its parameter cell
    template = json.loads((HERE / "tgn_label_fraction.ipynb").read_text(encoding="utf-8"))
    param_cell = next(i for i, c in enumerate(template["cells"]) if c.get("id") == "cell-params")
    header = template["cells"][param_cell]["source"][0]

    # 2. One folder per batch: notebook with its stages, and GPU kernel metadata
    for k, stages in enumerate(BATCHES, start=1):
        folder = HERE / "batches_gpu" / str(k)
        folder.mkdir(parents=True, exist_ok=True)
        nb = json.loads(json.dumps(template))
        nb["cells"][param_cell]["source"] = [header, f"STAGES = {stages!r}\n", "SEEDS = [0, 1, 2]\n",
                                             f"COMMIT = {args.commit!r}"]
        (folder / "tgn_label_fraction.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        meta = {"id": f"verzol/earlymule-tgn-lf-{k}", "title": f"EarlyMule TGN lf {k}",  # title must slug to the id
                "code_file": "tgn_label_fraction.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": ["verzol/earlymule-80k-earlymule-mule", "verzol/earlymule-80k-label-cases"],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(k, stages)


if __name__ == "__main__":
    main()
