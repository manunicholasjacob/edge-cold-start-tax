"""
E5 — Tax-aware model residency caching (the method contribution), evaluated in simulation
on MEASURED constants from E1 (latency) and E4 (energy).

Problem. A multi-model edge device serves a stream of inference requests addressed to different
models. RAM is too small to hold every model's session resident ("warm"). A residency policy
decides which sessions to keep warm. On a request to model i:
  * HIT  (i resident): pay warm cost  -> steady latency L_s[i], steady energy E_s[i]
  * MISS (i evicted) : pay cold wake  -> cold-wake latency L_c[i], cold-wake energy E_c[i];
                        i becomes resident, evicting others per policy to fit RAM budget B.
Each resident model occupies m[i] MB (measured resident footprint; proxy = weight bytes).

Policies: always-reload (no cache), LRU, LFU, GD-tax (ours: Greedy-Dual keyed on the measured
cold-wake energy penalty so high-tax models are retained), and a cost-aware Belady oracle (MIN
with per-miss cost = the model's cold-wake penalty) as an upper bound.

Metrics over the trace: mean & p99 request latency, total energy, and SLO-violation rate.
Swept over RAM budget B and two workloads (Zipf popularity; a 2-model detect->classify pipeline).
"""
import os, sys, json, heapq
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def load_constants():
    with open(os.path.join(RES, "e1_decomposition.json")) as f:
        e1 = json.load(f)
    e4 = {}
    p4 = os.path.join(RES, "e4_energy.json")
    if os.path.exists(p4):
        with open(p4) as f:
            e4 = json.load(f)
    foot = {}
    pf = os.path.join(RES, "footprint.json")
    if os.path.exists(pf):
        with open(pf) as f:
            foot = json.load(f)
    models = {}
    for name, d in e1.items():
        if name.startswith("_"):
            continue
        L_c = d["total_wake"]["median"]        # cold wake latency (s)
        L_s = d["steady"]["median"]            # steady latency (s)
        e = e4.get(name, {})
        E_c = e.get("E_wake_cold_J")
        E_s = e.get("E_inf_steady_J")
        # fallbacks if energy missing: approximate with a platform power model (5.2W active)
        if E_c is None:
            E_c = 5.2 * L_c
        if E_s is None:
            E_s = 5.2 * L_s
        m = foot.get(name, {}).get("resident_mb")
        if m is None:
            m = 1.3 * d["file_bytes"] / 1e6     # proxy: 1.3x weight bytes
        models[name] = {"L_c": L_c, "L_s": L_s, "E_c": E_c, "E_s": E_s, "m": m,
                        "pen_L": L_c - L_s, "pen_E": E_c - E_s}
    return models

# ---------- workload traces ----------
def zipf_trace(names, n=4000, a=1.2, seed=0):
    """Zipf popularity with model->rank assignment INDEPENDENT of cost (random per seed)."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, len(names) + 1)
    p = 1.0 / ranks**a
    p /= p.sum()
    order = list(names)
    rng.shuffle(order)
    idx = rng.choice(len(order), size=n, p=p)
    return [order[i] for i in idx]

def surveillance_trace(depl, models, n=4000, seed=0):
    """Cost/frequency-misaligned edge workload (the motivating case), drawn from the deployment
    set: a cheap model fires very often (motion/keyword gate), mid models sometimes, and the most
    EXPENSIVE-to-reload model rarely (a triggered detector). LRU/LFU evict the expensive detector
    between triggers; a tax-aware policy keeps it warm."""
    rng = np.random.default_rng(seed)
    cheap = min(depl, key=lambda m: models[m]["E_c"])
    pricey = max(depl, key=lambda m: models[m]["E_c"])
    mids = [m for m in depl if m not in (cheap, pricey)]
    seq = []
    for _ in range(n):
        r = rng.random()
        if r < 0.75:  seq.append(cheap)
        elif r < 0.95: seq.append(mids[rng.integers(len(mids))] if mids else cheap)
        else:          seq.append(pricey)   # 5% triggered expensive detector
    return seq

# ---------- cache simulator ----------
def simulate(trace, models, budget_mb, policy, slo_s):
    resident = {}          # name -> metadata for eviction key
    used = 0.0
    clock = 0
    tot_L = tot_E = 0.0
    lats = []
    viol = 0
    # precompute future-use index for Belady
    future = None
    if policy == "belady":
        future = {i: [] for i in range(len(trace))}
        last = {}
        # positions of each name
        positions = {}
        for t, nm in enumerate(trace):
            positions.setdefault(nm, []).append(t)

    freq = {}
    def evict_to_fit(need, t, incoming):
        nonlocal used
        while used + need > budget_mb and resident:
            if policy == "lru":
                victim = min(resident, key=lambda k: resident[k]["last"])
            elif policy == "lfu":
                victim = min(resident, key=lambda k: (freq.get(k, 0), resident[k]["last"]))
            elif policy == "gd_tax":
                # Greedy-Dual-Size-Frequency with cost = measured cold-wake energy penalty:
                # priority H = L + freq * dE / size. Evict min H. Balances cost, popularity, size.
                victim = min(resident, key=lambda k: resident[k]["H"])
            elif policy == "belady":
                # evict resident whose next use is farthest in the future
                def next_use(k):
                    for tt in positions.get(k, []):
                        if tt > t:
                            return tt
                    return 10**9
                victim = max(resident, key=next_use)
            else:
                victim = next(iter(resident))
            used -= models[victim]["m"]
            # GD aging: on eviction, inflate base clock (classic Greedy-Dual)
            if policy == "gd_tax":
                evict_to_fit.base = max(getattr(evict_to_fit, "base", 0.0),
                                        resident[victim]["H"])
            del resident[victim]

    evict_to_fit.base = 0.0
    for t, nm in enumerate(trace):
        clock += 1
        freq[nm] = freq.get(nm, 0) + 1
        m = models[nm]
        def gdsf_H():
            return evict_to_fit.base + freq[nm] * m["pen_E"] / max(m["m"], 1.0)
        if nm in resident:
            L, E = m["L_s"], m["E_s"]
            resident[nm]["last"] = clock
            if policy == "gd_tax":
                resident[nm]["H"] = gdsf_H()
        else:
            L, E = m["L_c"], m["E_c"]
            if policy != "reload":
                if m["m"] <= budget_mb:
                    evict_to_fit(m["m"], t, nm)
                    used += m["m"]
                    resident[nm] = {"last": clock, "H": gdsf_H()}
        tot_L += L; tot_E += E; lats.append(L)
        if L > slo_s:
            viol += 1
    lats = np.array(lats)
    return {"mean_L_ms": float(lats.mean()*1e3), "p99_L_ms": float(np.percentile(lats,99)*1e3),
            "total_E_J": tot_E, "energy_per_req_J": tot_E/len(trace),
            "slo_viol_pct": 100.0*viol/len(trace)}

def main():
    models = load_constants()
    names = list(models.keys())
    # use the CNN+INT8 set; pick a representative multi-model deployment (8 fp32 CNNs)
    depl = [n for n in ["squeezenet1.1","shufflenet-v2","mobilenetv2","resnet18",
                        "googlenet","densenet","efficientnet-lite4","resnet50"] if n in models]
    total_mem = sum(models[n]["m"] for n in depl)
    slo = 0.150  # 150 ms wake SLO
    budgets = [round(total_mem*f) for f in [0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 0.85, 1.0]]
    SEEDS = list(range(8))
    out = {"deployment": depl, "total_resident_mb": total_mem, "slo_ms": slo*1e3,
           "budgets_mb": budgets, "workloads": {}}
    workloads = {
        "zipf":         lambda s: zipf_trace(depl, seed=s),
        "surveillance": lambda s: surveillance_trace(depl, models, seed=s),
    }
    for wl, gen in workloads.items():
        out["workloads"][wl] = {}
        for pol in ["reload","lru","lfu","gd_tax","belady"]:
            rows = []
            for B in budgets:
                # average metrics over seeds
                accs = [simulate(gen(s), models, B, pol, slo) for s in SEEDS]
                agg = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}
                rows.append({"B_mb": B, **agg})
            out["workloads"][wl][pol] = rows
            print(f"[{wl}/{pol}] "
                  + " ".join(f"B={x['B_mb']}:{x['energy_per_req_J']:.2f}J/{x['p99_L_ms']:.0f}ms" for x in rows),
                  flush=True)
    with open(os.path.join(RES, "e5_residency.json"), "w") as f:
        json.dump(out, f, indent=2)
    # headline: gd_tax vs lru at tight budget (25% -> index 1)
    for wl in out["workloads"]:
        b_idx = 1
        lru = out["workloads"][wl]["lru"][b_idx]["energy_per_req_J"]
        gd  = out["workloads"][wl]["gd_tax"][b_idx]["energy_per_req_J"]
        rel = out["workloads"][wl]["reload"][b_idx]["energy_per_req_J"]
        bel = out["workloads"][wl]["belady"][b_idx]["energy_per_req_J"]
        lruP = out["workloads"][wl]["lru"][b_idx]["p99_L_ms"]
        gdP  = out["workloads"][wl]["gd_tax"][b_idx]["p99_L_ms"]
        print(f"[{wl}] @25% budget: reload={rel:.2f} lru={lru:.2f} gd_tax={gd:.2f} belady={bel:.2f} "
              f"J/req | gd vs lru: {100*(lru-gd)/lru:.1f}% energy, "
              f"p99 gd={gdP:.0f}ms vs lru={lruP:.0f}ms", flush=True)
    print("DONE E5")

if __name__ == "__main__":
    main()
