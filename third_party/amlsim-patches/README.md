# AMLSim patches (EarlyMule-TG)

Changes made to [IBM/AMLSim](https://github.com/IBM/AMLSim) for this thesis, kept as `git format-patch` files so the simulator and the datasets can be rebuilt from a clean clone.

- **Base commit:** `7338a4b` (Merge pull request #66 from nelsonjd/iss-65)
- **Branch in the local clone:** `earlymule-fixes`
- **License:** AMLSim is Apache License 2.0. Every modified source file carries a "Modified by EarlyMule-TG" header, and each changed block an `EarlyMule-TG:` comment.

## What the patches change

| Patch | Change | Why |
|---|---|---|
| `0001-*.patch` | `FanInTypology.java`, `FanOutTypology.java`: keep the alert window chosen by the generator | Upstream moved the start step close to step 0 but kept the end step, so fan-in/fan-out alerts spanned hundreds of days |
| | `scripts/transaction_graph_generator.py`: draw typology members from accounts with reliable normal activity | Upstream drew members from all accounts, so about half of the alert accounts had no normal history before the illicit flow |
| `0002-*.patch` | `conf_*.json`, `paramFiles/10K-earlymule`, `paramFiles/80K-earlymule-a100`, `paramFiles/80K-earlymule-a200` | Cycle alerts use `schedule_id=1` (ordered random interval) so relays receive before they send; 10K and 80K settings used in the feasibility study |

## Rebuild

```bash
git clone https://github.com/IBM/AMLSim.git
cd AMLSim
git checkout -b earlymule-fixes 7338a4b
git am ../third_party/amlsim-patches/*.patch
```

Then build and run as described in `docs/setup.md`, e.g. with `conf_80K-earlymule-a200.json`.

## Regenerate the patches

The patches are exported by a local helper script after committing on `earlymule-fixes`:

```bash
git -C AMLSim format-patch 7338a4b..earlymule-fixes --zero-commit --no-signature -o ../third_party/amlsim-patches
```
