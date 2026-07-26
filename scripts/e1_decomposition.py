"""
E1 — Cold-start latency decomposition.

For each model, measure and attribute the first-inference (cold) latency into components:
  disk_read    = session_create(COLD, cache dropped) - session_create(WARM-FILE, in page cache)
  build+alloc  = session_create(WARM-FILE)                 (graph load + optimize + arena alloc)
  cold_compute = first_infer - steady_infer                (cold i/d-caches, lazy allocations)
  steady       = median of trailing warm inferences
Also records major/minor page faults (rusage) for the COLD path -> disk-read attribution,
and the cold-start ratio (first_infer_total_wake / steady) which is the headline "tax".

Design choices:
  * frequency PINNED to 2.4 GHz (userspace) so DVFS ramp is NOT part of this experiment
    (DVFS interaction is isolated separately in E6).
  * trials repeated; we report median + IQR.
"""
import os, sys, json, gc, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cold_common as cc

TRIALS = 7
WARM_ITERS = 40      # warm inferences to reach + measure steady state
STEADY_TAIL = 15     # median of last N warm inferences = steady state
MODELS = ["squeezenet1.1","shufflenet-v2","mobilenetv2","resnet18","googlenet",
          "densenet","efficientnet-lite4","resnet50","vit-base",
          "squeezenet1.1-int8","mobilenetv2-int8","resnet50-int8"]

def measure_once(name, threads=4):
    path = cc.model_path(name)

    # ---------- COLD: drop caches, session-create pays full disk read ----------
    cc.drop_caches(3)
    gc.collect()
    maj0, min0 = cc.faults()
    t0 = cc.now()
    sess_cold = cc.new_session(path, intra_threads=threads, graph_opt="all")
    t1 = cc.now()
    feeds = cc.make_inputs(sess_cold)
    t2 = cc.now()
    sess_cold.run(None, feeds)     # first (cold) inference
    t3 = cc.now()
    maj1, min1 = cc.faults()
    sess_create_cold = t1 - t0
    first_infer      = t3 - t2
    majflt = maj1 - maj0
    minflt = min1 - min0

    # warm inferences -> steady state
    warm = []
    for _ in range(WARM_ITERS):
        a = cc.now(); sess_cold.run(None, feeds); warm.append(cc.now()-a)
    steady = float(np.median(warm[-STEADY_TAIL:]))
    warm_first = warm[0]
    del sess_cold; gc.collect()

    # ---------- WARM-FILE: file already in page cache, session-create has no disk I/O ----------
    cc.warm_file(path)
    gc.collect()
    t0 = cc.now()
    sess_warm = cc.new_session(path, intra_threads=threads, graph_opt="all")
    t1 = cc.now()
    sess_create_warmfile = t1 - t0
    del sess_warm; gc.collect()

    # ---------- graph-optimize contribution: build with opt=none vs all (warm file) ----------
    cc.warm_file(path)
    t0 = cc.now(); s_none = cc.new_session(path, intra_threads=threads, graph_opt="none"); t1 = cc.now()
    del s_none; gc.collect()
    sess_create_noopt = t1 - t0

    disk_read   = max(sess_create_cold - sess_create_warmfile, 0.0)
    build_alloc = sess_create_warmfile
    optimize    = max(sess_create_warmfile - sess_create_noopt, 0.0)
    cold_compute= max(first_infer - steady, 0.0)
    total_wake  = sess_create_cold + first_infer
    return {
        "sess_create_cold": sess_create_cold,
        "sess_create_warmfile": sess_create_warmfile,
        "sess_create_noopt": sess_create_noopt,
        "first_infer": first_infer,
        "steady": steady,
        "warm_first": warm_first,
        "disk_read": disk_read,
        "build_alloc": build_alloc,
        "optimize": optimize,
        "cold_compute": cold_compute,
        "total_wake": total_wake,
        "tax_ratio": total_wake / steady if steady > 0 else None,
        "majflt": majflt, "minflt": minflt,
    }

def agg(key, runs):
    vals = [r[key] for r in runs if r[key] is not None]
    return {"median": float(np.median(vals)), "iqr": float(np.subtract(*np.percentile(vals,[75,25]))),
            "min": float(np.min(vals)), "max": float(np.max(vals))}

def main():
    cc.set_freq_khz(2400000)   # pin 2.4 GHz; isolate DVFS out of E1
    print("pinned freqs:", cc.cur_freqs_khz(), flush=True)
    out = {}
    for name in MODELS:
        runs = []
        for k in range(TRIALS):
            try:
                r = measure_once(name)
                runs.append(r)
                print(f"[{name}] trial {k}: wake={r['total_wake']*1e3:.1f}ms "
                      f"steady={r['steady']*1e3:.2f}ms tax={r['tax_ratio']:.1f}x "
                      f"majflt={r['majflt']}", flush=True)
            except Exception as e:
                print(f"[{name}] trial {k} ERROR: {e}", flush=True)
        if not runs:
            continue
        keys = ["sess_create_cold","sess_create_warmfile","sess_create_noopt","first_infer",
                "steady","warm_first","disk_read","build_alloc","optimize","cold_compute",
                "total_wake","tax_ratio","majflt","minflt"]
        out[name] = {"file_bytes": cc.file_size(name), "n": len(runs),
                     **{k: agg(k, runs) for k in keys}}
        with open(os.path.expanduser("~/coldstart/results/e1_decomposition.json"), "w") as f:
            json.dump(out, f, indent=2)
    cc.set_governor("schedutil")  # restore
    print("DONE E1", flush=True)

if __name__ == "__main__":
    main()
