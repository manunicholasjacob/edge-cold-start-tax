"""Measure resident memory footprint per model (warm session held) + dump input signatures.
Resident footprint = process RSS increase from just-before session-create to after first
inference, i.e. the RAM a kept-warm session actually occupies. Feeds E5's memory budgets."""
import os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cold_common as cc

MODELS = list(cc.ZOO.keys())

def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024.0
    return None

def main():
    out = {}
    for name in MODELS:
        path = cc.model_path(name)
        try:
            cc.warm_file(path); gc.collect()
            base = rss_mb()
            sess = cc.new_session(path, intra_threads=4, graph_opt="all")
            # input signature
            sig = [{"name": i.name, "shape": [str(x) for x in i.shape], "type": i.type}
                   for i in sess.get_inputs()]
            feeds = cc.make_inputs(sess)
            try:
                sess.run(None, feeds); ran = True
            except Exception as e:
                ran = False; sig_err = str(e)[:120]
            peak = rss_mb()
            resident = peak - base
            del sess; gc.collect()
            rec = {"file_mb": cc.file_size(name)/1e6, "resident_mb": round(resident, 1),
                   "inputs": sig, "ran": ran}
            if not ran:
                rec["run_error"] = sig_err
            out[name] = rec
            print(f"[{name}] resident={resident:.0f}MB inputs={sig} ran={ran}", flush=True)
        except Exception as e:
            print(f"[{name}] ERR {e}", flush=True)
    with open(os.path.expanduser("~/coldstart/results/footprint.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("DONE FOOTPRINT", flush=True)

if __name__ == "__main__":
    main()
