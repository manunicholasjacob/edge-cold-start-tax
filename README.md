# The Cold-Start Tax — Duty-Cycled Edge DNN Inference on Raspberry Pi 5

Artifact for the paper **"The Cold-Start Tax: Warm-Up, Re-Warm, and Residency Policy for
Duty-Cycled Edge DNN Inference"** (submitted to *IEEE Internet of Things Journal*).

Edge-inference benchmarks report **steady-state** latency and energy — the numbers a model
reaches *after* it is loaded and warmed. But a huge class of IoT devices **duty-cycle**: they
wake on an event, run a few inferences, and sleep. For them the **transient at every wake**, not
the steady state, dominates latency and energy. This work measures that transient — the
**cold-start tax** — on a Raspberry Pi 5 (ARM Cortex-A76), decomposes it, measures its energy
with on-board PMIC telemetry, exposes the page-cache **eviction cliff** that re-triggers it, and
turns the measurements into a residency policy.

## Headline findings (all measured on real hardware)

| # | Finding |
|---|---------|
| 1 | The cold-start tax is **5–23× the steady-state latency** across 12 models (compact/dense/INT8 CNNs + a ViT). |
| 2 | It is a **weight-loading cost**: flash read is **55–81%** of the cold wake (~94 MB/s microSD). |
| 3 | **There is no compute warm-up** — the first inference already runs at steady-state speed (warm-up loops amortize a cost that doesn't exist). |
| 4 | A cold wake costs the **energy of tens–hundreds of steady-state inferences** (PMIC-measured). |
| 5 | **Use-once file I/O does *not* evict a recently-used model** (Linux active-list scan resistance), but a **co-tenant's anonymous working set does** — a sharp **eviction cliff** re-pays the full tax. |
| 6 | Because the wake is I/O-bound, the **DVFS governor ramp adds only ~0–4%** — max-clock through a wake is wasted energy. |
| 7 | Static **INT8 shrinks the absolute tax but not its ratio** (steady falls in step). |
| 8 | **GD-Tax**, a tax-aware residency cache keyed on measured cold-wake energy penalty, cuts energy/request and tail latency at tight RAM budgets vs LRU/LFU, approaching a Belady oracle. |

## Repository layout

```
scripts/
  cold_common.py      shared harness (input build, cache control, PMIC power, DVFS)
  e1_decomposition.py E1: cold-wake latency decomposition
  e2_warmup.py        E2: warm-up convergence curves
  e3_eviction.py      E3: streaming page-cache pressure (scan resistance)
  e3b_cliff.py        E3b: anonymous-pressure eviction cliff
  e4_energy.py        E4: energy per wake (PMIC, robust pooled-power method)
  e6_dvfs.py          E6: DVFS ramp tax at wake
  vit_supplement.py   ViT-Base runs merged into E1/E2/E6
  footprint.py        resident memory per model (+ input signatures)
  e5_residency.py     E5: GD-Tax residency policy simulation (uses measured constants)
  fig_gen.py          regenerates all paper figures
  compute_stats.py    prints every paper-cited number from the raw JSONs
results/              raw measurement JSONs + STATS.txt
paper/                LaTeX (IEEEtran), refs.bib, figures, compiled PDF
```

## Reproduce

On a Raspberry Pi 5 (Debian, ONNX Runtime ≥ 1.20, `perf`, `sudo` for `drop_caches`), place ONNX
models in `~/coldstart/models/` and run each `eN_*.py`; then pull `results/*.json` and run
`fig_gen.py` and `compute_stats.py` locally. `perf_event_paranoid` should be `-1` and the CPU
governor is controlled by the scripts.

## Platform

Raspberry Pi 5 Model B (Broadcom BCM2712, quad-core Cortex-A76 @ up to 2.4 GHz, **2 GB** LPDDR4X,
32 GB class-10 microSD), Debian 12 / Linux 6.12, ONNX Runtime 1.24 (CPU EP). Energy via
`vcgencmd pmic_read_adc` (per-rail V×I).

## Author

Manu Nicholas Jacob. Part of an edge-AI measurement portfolio on the Pi 5; companion to
*The Memory Wall Governs Edge DNN Inference* and *Latency-Optimal Is Not Energy-Optimal*.
