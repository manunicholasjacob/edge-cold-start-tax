# Lab Notebook — The Cold-Start Tax (Paper 11)

Platform: Raspberry Pi 5, 2 GB, Cortex-A76, Debian 12 / Linux 6.12, ONNX Runtime 1.24 (CPU EP).
Models reused from the memory-wall campaign (`~/memwall/models`): 8 FP32 CNNs, 3 static-INT8
CNNs, ViT-Base/16. Cold start controlled via `sync; echo 3 > /proc/sys/vm/drop_caches`. Clock
pinned to 2.4 GHz (userspace governor) except in the DVFS experiment. Energy via
`vcgencmd pmic_read_adc` (per-rail V×I).

## Experiments
- **E1 — decomposition.** Cold session-create + first inference, split into disk read
  (cold − warm-file), graph build+arena, optimization (opt=all − opt=none), cold compute
  (first − steady). 5–7 trials/model, medians+IQR. Result: tax 5–23×; disk 55–81%; cold compute ≈0.
- **E2 — warm-up curve.** Per-inference latency after session create. first/steady ≈ 1.0×,
  N_conv 0–1 → no compute warm-up.
- **E3 — page-cache pressure (streaming).** Stream up to 2 GB of *use-once* file reads after
  warming the model. Wake stays warm for all models → Linux active-list scan resistance.
- **E3b — eviction cliff (anonymous).** A co-tenant anonymous working set. Sharp cliff: once free
  memory drops below the model's footprint, session-create thrashes swap and the wake explodes to
  13–32 s (>80×). Captured for MobileNetV2, ResNet-18, ResNet-50.
- **E4 — energy per wake (PMIC).** Robust pooled-power method: mean power over many repeated cold/
  warm/steady operations × measured duration (the single-window integral collapses at ~10 Hz
  sampling for short wakes). Cold wake = 4–19 inferences of energy; ViT 35 J; idle 2.1 W.
- **E5 — GD-Tax residency policy (simulation on measured constants).** Greedy-Dual-Size-Frequency
  keyed on cold-wake energy penalty vs reload/LRU/LFU/Belady, Zipf + surveillance workloads,
  averaged over 8 seeds. GD-Tax never underperforms LRU/LFU; up to 15% energy at tight budgets.
- **E6 — DVFS ramp.** schedutil vs pinned min/max at wake. Ramp tax ~0–4% (I/O-bound wake).
- **E7a — storage sensitivity.** microSD vs tmpfs (/dev/shm). Storage = 55–81% of the cold wake;
  tmpfs up to 4× faster → predicts NVMe/eMMC shrinks the tax.
- **E7b — graph-optimization crossover.** cold-wake and steady at opt {none,basic,all}; crossover
  B* below which disabling optimization wins. B*<5 for most; ≈18 for ResNet-18.
- **E8 — on-device GD-Tax (hardware validation).** The residency server implemented and run on
  the Pi 5 itself: real sessions held to a RAM budget, PMIC energy measured over a request trace,
  eviction releases session + file-cache (posix_fadvise DONTNEED). Reload baseline measured
  2.82 J/req vs 2.81 J predicted (<1%). GD-Tax beats LRU in 3/4 configs (up to 14.4% energy, also
  lower latency/misses); trails LRU by 4.2% in the tightest surveillance case (within run-to-run
  variance). Memory-safe (no hog); resident set always < budget < RAM.
- **Analysis models.** Predictive size-only model (flash BW 96 MB/s, R²=1.00; wake MAPE 12%) and
  duty-cycle amortization (need 38–119 inferences/wake for <10% overhead).

## Gotchas / incidents
- ViT-Base input is fully symbolic `[batch, num_channels, height, width]`; a naive NHWC-channel
  heuristic set the width dim to 3 (`{224,3}` conv error). Fix: symbolic non-batch/channel dims
  default to 224; NCHW symbolic channel → 3.
- PMIC sampling is ~10–15 Hz (subprocess-bound); short wakes need pooled-power estimation, not a
  single-window integral. Larger models are well-sampled; quantitative energy claims anchor there.
- **Pi swap-death (power-cycle):** the E3b cliff sweep pushed a 1.65–1.78 GB anonymous hog on the
  2 GB box; combined with a 196 MB ResNet-50 session it drove the system into swap thrashing that
  starved the network stack — the Pi became unreachable and was power-cycled. All on-disk results
  survived and were re-pulled; the ViT supplement and E7 were then run on a fresh boot. The cliff
  (the point of E3b) is the headline result; the pathological high-pressure points were not
  re-run.
- `perf` not used (getrusage + vcgencmd suffice). `sudo` is NOPASSWD for `drop_caches`/governor.
