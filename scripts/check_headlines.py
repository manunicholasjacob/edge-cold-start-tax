"""Recompute the README headline findings from the released JSONs in results/ and fail on drift.

Standard library only. Each check recomputes a README number from the raw per-model fields and
compares it at the precision the README states it.
"""
import json
import os
import sys

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FAIL = []


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def check(label, got, want, ndigits=0):
    ok = round(got, ndigits) == round(want, ndigits)
    print(f"{'ok  ' if ok else 'FAIL'} {label}: recomputed {got:.3f}, README {want}")
    if not ok:
        FAIL.append(label)


def holds(label, cond, detail):
    print(f"{'ok  ' if cond else 'FAIL'} {label}: {detail}")
    if not cond:
        FAIL.append(label)


e1, e2, e4 = load("e1_decomposition.json"), load("e2_warmup.json"), load("e4_energy.json")
e6, e5 = load("e6_dvfs.json"), load("e5_residency.json")

# Finding 1: cold-start tax is 5 to 23x steady state, across 12 models.
tax = {m: v["tax_ratio"]["median"] for m, v in e1.items()}
check("F1 model count", len(tax), 12)
check("F1 lowest tax multiple", min(tax.values()), 5)
check("F1 highest tax multiple", max(tax.values()), 23)

# Finding 2: flash read is 55 to 81% of the cold wake for the FP32 CNNs; ~94 MB/s microSD.
fp32_cnn = [m for m in e1 if not m.endswith("-int8") and not m.startswith("vit")]
frac = [100 * e1[m]["disk_read"]["median"] / e1[m]["total_wake"]["median"] for m in fp32_cnn]
check("F2 FP32 CNN count", len(fp32_cnn), 8)
check("F2 flash-read share low %", min(frac), 55)
check("F2 flash-read share high %", max(frac), 81)
r50 = e1["resnet50"]
check("F2 effective flash read MB/s", r50["file_bytes"] / 1e6 / r50["disk_read"]["median"], 94)

# Finding 3: no compute warm-up; the first inference already runs at steady-state speed.
fos = [v["first_over_steady"] for v in e2.values() if v.get("first_over_steady")]
check("F3 first/steady low", min(fos), 0.92, 2)
check("F3 first/steady high", max(fos), 1.10, 2)
holds("F3 inferences to steady", max(v["n_conv"] for v in e2.values()) <= 1,
      f"max n_conv {max(v['n_conv'] for v in e2.values())}")

# Finding 4: a cold wake costs the energy of 4 to 19 steady-state inferences.
ninf = [v["E_wake_cold_J"] / v["E_inf_steady_J"] for m, v in e4.items()
        if not m.startswith("_") and v.get("E_inf_steady_J")]
check("F4 cold wake in steady inferences, low", min(ninf), 4)
check("F4 cold wake in steady inferences, high", max(ninf), 19)

# Finding 6: the DVFS governor ramp adds only ~0 to 4%.
ramp = [v["ramp_tax_pct"] for v in e6.values()]
check("F6 ramp tax low %", min(ramp), 0)
check("F6 ramp tax high %", max(ramp), 4)

# Finding 8: at tight RAM budgets GD-Tax uses less energy per request than LRU and LFU.
total = e5["total_resident_mb"]
for wl, w in e5["workloads"].items():
    for i in range(len(w["gd_tax"])):
        pct = 100 * w["gd_tax"][i]["B_mb"] / total
        if pct > 55:
            continue
        gd = w["gd_tax"][i]["energy_per_req_J"]
        best = min(w["lru"][i]["energy_per_req_J"], w["lfu"][i]["energy_per_req_J"])
        holds(f"F8 {wl} at {pct:.0f}% budget", gd < best,
              f"GD-Tax {gd:.3f} J/req vs best of LRU/LFU {best:.3f}")

if FAIL:
    print(f"\n{len(FAIL)} check(s) drifted: {', '.join(FAIL)}")
    sys.exit(1)
print("\nall README headline findings recompute from results/")
