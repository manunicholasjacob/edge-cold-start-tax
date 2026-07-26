"""
E3 (v2) — Page-cache eviction cliff via page-cache streaming pressure.

v1 used an anonymous memory hog; on a 2 GB Pi it could not evict a small (<100 MB) model file
without approaching OOM, because the kernel retains a small working set. v2 instead applies
*page-cache* pressure the way a real duty-cycled deployment does: during the idle interval,
other file I/O (logs, buffers, co-tenant model loads) streams through the page cache and evicts
least-recently-used pages. We emulate that by sequentially reading P MB of OTHER data (we cycle
through the other model files, reading but not holding them). Once cumulative pressure exceeds
what the cache can retain, the target model's pages are evicted and the next wake pays the full
cold disk-read tax again. No large anonymous allocation -> no OOM risk.

Output: cliff curve wake_ms vs streamed-pressure P (MB), per target model. Frequency pinned.
"""
import os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cold_common as cc

TRIALS = 3
P_LIST = [0, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048]  # MB of other data streamed
MODELS = ["mobilenetv2","resnet18","resnet50","densenet","efficientnet-lite4"]

def stream_pressure(target_path, mb):
    """Read `mb` MB of data from files OTHER than target, streaming (not held), to pressure
    the page cache. Cycle through the model directory files as the pressure source."""
    if mb <= 0:
        return
    budget = mb * 1024 * 1024
    srcs = [os.path.join(cc.MODELS_DIR, f) for f in os.listdir(cc.MODELS_DIR)
            if os.path.join(cc.MODELS_DIR, f) != target_path and f.endswith(".onnx")]
    read = 0
    buf = bytearray(1 << 20)
    while read < budget:
        progressed = False
        for s in srcs:
            try:
                with open(s, "rb") as f:
                    while read < budget:
                        n = f.readinto(buf)
                        if not n:
                            break
                        read += n
                        progressed = True
            except Exception:
                continue
            if read >= budget:
                break
        if not progressed:
            break

def measure(name, P):
    path = cc.model_path(name)
    cc.drop_caches(3); gc.collect()
    cc.warm_file(path)                 # target hot in cache
    stream_pressure(path, P)           # idle-interval page-cache pressure
    t0 = cc.now()
    sess = cc.new_session(path, intra_threads=4, graph_opt="all")
    t1 = cc.now()
    feeds = cc.make_inputs(sess)
    sess.run(None, feeds)
    t2 = cc.now()
    del sess; gc.collect()
    return {"sess_create_ms": (t1-t0)*1e3, "wake_ms": (t2-t0)*1e3}

def main():
    cc.set_freq_khz(2400000)
    out = {}
    for name in MODELS:
        fb = cc.file_size(name)/1e6
        # reference hot/cold anchors
        cc.drop_caches(3); cc.warm_file(cc.model_path(name))
        out[name] = {"file_mb": fb, "curve": []}
        for P in P_LIST:
            runs = []
            for _ in range(TRIALS):
                try:
                    runs.append(measure(name, P))
                except Exception as e:
                    print(f"[{name}] P={P} ERR {e}", flush=True)
            if not runs:
                continue
            wake = float(np.median([r["wake_ms"] for r in runs]))
            sc   = float(np.median([r["sess_create_ms"] for r in runs]))
            out[name]["curve"].append({"P_mb": P, "wake_ms": wake, "sess_create_ms": sc, "n": len(runs)})
            print(f"[{name}] file={fb:.0f}MB P={P:>4}MB sess_create={sc:.0f}ms wake={wake:.0f}ms", flush=True)
            with open(os.path.expanduser("~/coldstart/results/e3_eviction.json"),"w") as f:
                json.dump(out, f, indent=2)
    cc.set_governor("schedutil")
    print("DONE E3", flush=True)

if __name__ == "__main__":
    main()
