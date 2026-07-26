"""
E3b — The eviction cliff under anonymous (co-tenant working-set) memory pressure.

Companion to the streaming-pressure result (e3_streaming): use-once file reads do NOT evict a
recently-used model, because Linux protects active-list pages against scan (scan resistance).
A co-tenant's *anonymous* working set is different: to satisfy it the kernel must reclaim file
cache, including the model's pages. We warm+use the target (active), then allocate and touch
S MB of anonymous memory (a co-tenant), then measure the target's next wake. Sweeping S toward
the 2 GB limit exposes a sharp cliff where session-create jumps from warm to cold as the model's
pages are evicted. OOM-guarded (skip a point if MemAvailable would go dangerously low).
Frequency pinned 2.4 GHz.
"""
import os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cold_common as cc

TRIALS = 3
S_LIST = [0, 600, 1000, 1300, 1450, 1550, 1650, 1720, 1780]  # MB anonymous co-tenant
MODELS = ["mobilenetv2","resnet18","resnet50","densenet","efficientnet-lite4"]

def avail_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1])//1024
    return 0

def hog(mb):
    if mb <= 0: return None
    n = mb*1024*1024
    buf = bytearray(n)
    for i in range(0, n, 4096): buf[i] = 1  # touch every page -> backed by RAM
    return buf

def measure(name, S):
    path = cc.model_path(name)
    cc.drop_caches(3); gc.collect()
    cc.warm_file(path)
    # touch the target once so its pages are on the active list (a "recently used" model)
    s0 = cc.new_session(path, intra_threads=4, graph_opt="all")
    s0.run(None, cc.make_inputs(s0)); del s0; gc.collect()
    cc.warm_file(path)  # ensure file pages resident+active
    buf = hog(S)                                    # co-tenant anonymous pressure
    av = avail_mb()
    t0 = cc.now()
    sess = cc.new_session(path, intra_threads=4, graph_opt="all")
    t1 = cc.now()
    sess.run(None, cc.make_inputs(sess))
    t2 = cc.now()
    del sess; del buf; gc.collect()
    return {"sess_create_ms": (t1-t0)*1e3, "wake_ms": (t2-t0)*1e3, "avail_mb": av}

def main():
    cc.set_freq_khz(2400000)
    out = {}
    for name in MODELS:
        fb = cc.file_size(name)/1e6
        out[name] = {"file_mb": fb, "curve": []}
        for S in S_LIST:
            runs = []
            for _ in range(TRIALS):
                try:
                    r = measure(name, S)
                    runs.append(r)
                except MemoryError:
                    print(f"[{name}] S={S} OOM-skip", flush=True); break
                except Exception as e:
                    print(f"[{name}] S={S} ERR {e}", flush=True)
            if not runs: continue
            wake = float(np.median([r["wake_ms"] for r in runs]))
            sc   = float(np.median([r["sess_create_ms"] for r in runs]))
            av   = float(np.median([r["avail_mb"] for r in runs]))
            out[name]["curve"].append({"S_mb":S,"wake_ms":wake,"sess_create_ms":sc,"avail_mb":av,"n":len(runs)})
            print(f"[{name}] file={fb:.0f}MB S={S:>4} avail={av:.0f} sess_create={sc:.0f}ms wake={wake:.0f}ms", flush=True)
            with open(os.path.expanduser("~/coldstart/results/e3b_cliff.json"),"w") as f:
                json.dump(out, f, indent=2)
    cc.set_governor("schedutil")
    print("DONE E3b", flush=True)

if __name__ == "__main__":
    main()
