# Paper 11 — The Cold-Start Tax

**Title:** The Cold-Start Tax: Warm-Up, Re-Warm, and Residency Policy for Duty-Cycled Edge DNN Inference
**Author:** Manu Nicholas Jacob
**Target venue:** IEEE Internet of Things Journal (IEEEtran journal, single-anonymous, rolling)
**Platform:** Raspberry Pi 5 (Broadcom BCM2712, quad-core ARM Cortex-A76, 2 GB LPDDR4X, microSD), ONNX Runtime 1.24
**Public artifact:** https://github.com/manunicholasjacob/edge-cold-start-tax

## One-paragraph summary
Edge-inference benchmarks report steady-state numbers, but a huge class of IoT devices duty-cycle
— wake, run a few inferences, sleep — so the transient at every wake dominates their latency and
energy. This paper is the first systematic characterization of that transient (the *cold-start
tax*) on an edge SBC, and turns it into deployable methods.

## Headline results (all hardware-measured)
- Cold-start tax **5–23×** steady-state latency across 12 models (compact/dense/INT8 CNNs + ViT).
- The tax is **weight-loading bound**: flash read is 55–81% of the wake; effective microSD BW
  96 MB/s (R²=1.00). **No compute warm-up** — the first inference already runs at steady speed.
- **Size-only predictor** of the cold wake: MAPE 12% (R²=0.998), needing only the model file size.
- **Energy** (Pi 5 PMIC): a cold wake costs 4–19 steady-state inferences; up to 35 J for ViT-Base.
- **Eviction cliff**: on 2 GB, a co-tenant's working set evicts the model and the next wake blows
  up >80× (to 13–32 s via swap thrash). Use-once file I/O does *not* evict (page-cache scan
  resistance).
- **DVFS** ramp adds only ~0–4% (I/O-bound wake). **INT8** shrinks the absolute tax but not its ratio.
- **Methods:** disable graph optimization for short wakes (crossover B*); and **GD-Tax**, a
  tax-aware residency cache that matches/beats LRU/LFU (up to 15% energy) and matches a Belady
  reference — **validated on the Pi itself** (up to 14% measured energy reduction, 3 of 4 configs).

## Contents
- `paper/` — LaTeX (IEEEtran), refs.bib, compiled `main.pdf`, cover letter
- `scripts/` — measurement harness (E1–E7), analysis, figure generation
- `data/` — raw measurement JSONs + computed STATS
- `figures/` — all paper figures
- `submission/` — zipped submission bundle
- `LAB_NOTEBOOK.md` — full campaign log
