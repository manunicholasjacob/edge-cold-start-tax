"""
E2 — Warm-up convergence curves.
Drop caches, create session, run K inferences recording each latency. Characterize the
transient: N_conv = first index after which all inferences are within +/-5% of steady
(trailing-median). warmup_excess_s = sum(lat_i - steady) over the transient (extra wall time
paid before reaching steady state). Repeat TRIALS; report per-iteration median curve + stats.
Frequency pinned 2.4 GHz to isolate warm-up from DVFS.
"""
import os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cold_common as cc

TRIALS = 5
K = 60
BAND = 0.05
MODELS = ["squeezenet1.1","shufflenet-v2","mobilenetv2","resnet18","googlenet",
          "densenet","efficientnet-lite4","resnet50","vit-base",
          "squeezenet1.1-int8","mobilenetv2-int8","resnet50-int8"]

def one_curve(name):
    path = cc.model_path(name)
    cc.drop_caches(3); gc.collect()
    sess = cc.new_session(path, intra_threads=4, graph_opt="all")
    feeds = cc.make_inputs(sess)
    lat = []
    for _ in range(K):
        a = cc.now(); sess.run(None, feeds); lat.append(cc.now()-a)
    del sess; gc.collect()
    return np.array(lat)

def analyze(curve):
    steady = float(np.median(curve[-15:]))
    # N_conv: smallest i such that curve[i:] all within band of steady
    n_conv = len(curve)
    for i in range(len(curve)):
        if np.all(np.abs(curve[i:] - steady) <= BAND*steady):
            n_conv = i; break
    excess = float(np.sum(np.clip(curve[:max(n_conv,1)] - steady, 0, None)))
    return steady, n_conv, excess

def main():
    cc.set_freq_khz(2400000)
    out = {}
    for name in MODELS:
        curves = []
        for k in range(TRIALS):
            try:
                curves.append(one_curve(name))
            except Exception as e:
                print(f"[{name}] trial {k} ERROR {e}", flush=True)
        if not curves:
            continue
        L = min(len(c) for c in curves)
        arr = np.stack([c[:L] for c in curves])
        med_curve = np.median(arr, axis=0)
        steady, n_conv, excess = analyze(med_curve)
        first = float(med_curve[0])
        out[name] = {
            "median_curve_ms": [round(float(x)*1e3,4) for x in med_curve],
            "steady_ms": steady*1e3,
            "first_infer_ms": first*1e3,
            "first_over_steady": first/steady if steady>0 else None,
            "n_conv": int(n_conv),
            "warmup_excess_ms": excess*1e3,
            "trials": len(curves),
        }
        print(f"[{name}] steady={steady*1e3:.2f}ms first={first*1e3:.1f}ms "
              f"({first/steady:.1f}x) N_conv={n_conv} excess={excess*1e3:.1f}ms", flush=True)
        with open(os.path.expanduser("~/coldstart/results/e2_warmup.json"),"w") as f:
            json.dump(out, f, indent=2)
    cc.set_governor("schedutil")
    print("DONE E2", flush=True)

if __name__ == "__main__":
    main()
