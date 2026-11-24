"""Generate the Kaggle GPU notebook folders of the label-scarcity runs of the graph models.

Each batch is a list of stages (module, config, label fraction or None for full labels, optional seeds); the stages of one batch run
one after another, the seeds of a stage in parallel on the two T4 GPUs. Batches are sized to stay well under the
12-hour session limit. Each folder gets the GPU template with its own parameter cell and a kernel-metadata.json
(id verzol/earlymule-tgn-lf-<k>, T4, the 80K export and the case map as inputs).
Writes notebooks/kaggle/label_fraction/batches_gpu/<k>/ (git-ignored).
With --dataset <draw>, writes the two confirmation batches of that draw instead (protocol section 12): full labels
for every graph model of the core set and 5% of cases for the role-transition step, kernels
verzol/earlymule-conf-<draw suffix>-<a|b>, folders batches_gpu/<draw>-<a|b>/; the draw's export carries its case map.
Run: python notebooks/kaggle/label_fraction/make_gpu_batches.py --commit <sha> [--dataset <draw>]
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
# Confirmation draws: about 5 and 3.5 session hours, so both fit one 12-hour session each
CONFIRMATION = {"a": [(TGN, "tgn_transition.yaml", None), (TGN, "tgn_transition.yaml", 0.05),
                      (SAGE, "graphsage.yaml", None)],
                "b": [(TGN, "tgn_features.yaml", None), (STAGT, "stagt.yaml", None)]}
MAIN = "80K-earlymule-mule"


def kernel_meta(kernel: str, title: str, sources: list[str]) -> dict:
    """Kaggle metadata of a private T4 notebook; the title must slug to the kernel id."""
    return {"id": f"verzol/{kernel}", "title": title, "code_file": "tgn_label_fraction.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
            "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": sources,
            "competition_sources": [], "kernel_sources": [], "model_sources": []}


def main() -> None:
    """Write the notebook folder of every batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True, help="commit the notebooks check out (pushed to GitHub)")
    parser.add_argument("--dataset", default=MAIN, help="AMLSim run; another draw than the main one gives its "
                                                        "confirmation batches")
    args = parser.parse_args()

    # 1. Template notebook and its parameter cell
    template = json.loads((HERE / "tgn_label_fraction.ipynb").read_text(encoding="utf-8"))
    param_cell = next(i for i, c in enumerate(template["cells"]) if c.get("id") == "cell-params")
    header = template["cells"][param_cell]["source"][0]

    # 2. Batches: label scarcity on the main dataset, or the two confirmation batches of a draw
    export = f"verzol/earlymule-{args.dataset.lower()}"
    if args.dataset == MAIN:
        batches = {str(k): (stages, f"earlymule-tgn-lf-{k}", f"EarlyMule TGN lf {k}",
                            [export, "verzol/earlymule-80k-label-cases"])
                   for k, stages in enumerate(BATCHES, start=1)}
    else:
        draw = args.dataset.rsplit("-", 1)[-1]  # e.g. g1
        batches = {f"{args.dataset}-{part}": (stages, f"earlymule-conf-{draw}-{part}",
                                               f"EarlyMule conf {draw} {part}", [export])
                   for part, stages in CONFIRMATION.items()}

    # 3. One folder per batch: notebook with its dataset and stages, and GPU kernel metadata
    for name, (stages, kernel, title, sources) in batches.items():
        folder = HERE / "batches_gpu" / name
        folder.mkdir(parents=True, exist_ok=True)
        nb = json.loads(json.dumps(template))
        nb["cells"][param_cell]["source"] = [header, f"DATASET = {args.dataset!r}\n", f"STAGES = {stages!r}\n",
                                             "SEEDS = [0, 1, 2]\n", f"COMMIT = {args.commit!r}"]
        (folder / "tgn_label_fraction.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        meta = kernel_meta(kernel, title, sources)
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(name, kernel, stages)


if __name__ == "__main__":
    main()
