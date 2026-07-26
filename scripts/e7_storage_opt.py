"""
E7 — Depth experiments to strengthen the causal and actionable story.

E7a Storage sensitivity (tmpfs vs microSD). We claim the cold-start tax is dominated by loading
weights from flash. To establish causality (not correlation with model size) we load the SAME
model from RAM-backed tmpfs (/dev/shm, effectively infinite storage bandwidth) versus the microSD
card. The tmpfs wake is the irreducible build+optimize+compute floor; the microSD-minus-tmpfs gap
is the storage contribution. This also predicts what a faster medium (NVMe/eMMC) would achieve.

E7b Graph-optimization crossover. Higher ONNX Runtime optimization makes session creation slower
but steady inference faster. For a duty-cycled device running only B inferences per wake there is
a crossover B* below which DISABLING optimization minimizes total wake+work time. We measure cold
wake and steady latency at optimization levels {none, basic, all} and compute B* per model.

Frequency pinned 2.4 GHz.
"""
import os, sys, json, gc, shutil, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cold_common as cc

TRIALS = 5
SHM = "/dev/shm"
# exclude vit-base from tmpfs (346MB in RAM on a 2GB box is risky); include the rest <=110MB
STORAGE_MODELS = ["squeezenet1.1","shufflenet-v2","mobilenetv2","resnet18","googlenet",
                  "densenet","efficientnet-lite4","resnet50","resnet50-int8"]
OPT_MODELS = ["mobilenetv2","resnet18","resnet50","densenet","efficientnet-lite4","squeezenet1.1"]

def wake_from(path, opt="all"):
    t0 = cc.now()
    sess = cc.new_session(path, intra_threads=4, graph_opt=opt)
    t1 = cc.now()
    feeds = cc.make_inputs(sess)
    sess.run(None, feeds)
    t2 = cc.now()
    # steady
    lat = []
    for _ in range(30):
        a = cc.now(); sess.run(None, feeds); lat.append(cc.now()-a)
    steady = float(np.median(lat[-15:]))
    del sess; gc.collect()
    return {"sess_create": t1-t0, "first_infer": t2-t1, "wake": t2-t0, "steady": steady}

def storage_experiment():
    out = {}
    for name in STORAGE_MODELS:
        sd_path = cc.model_path(name)
        shm_path = os.path.join(SHM, os.path.basename(sd_path))
        try:
            shutil.copy(sd_path, shm_path)
        except Exception as e:
            print(f"[stor {name}] copy ERR {e}", flush=True); continue
        sd, shm = [], []
        for _ in range(TRIALS):
            cc.drop_caches(3); gc.collect()
            sd.append(wake_from(sd_path, "all")["wake"])
            # tmpfs: drop_caches does not evict tmpfs pages (always RAM-resident)
            gc.collect()
            shm.append(wake_from(shm_path, "all")["wake"])
        try: os.remove(shm_path)
        except Exception: pass
        sdm, shmm = float(np.median(sd)), float(np.median(shm))
        out[name] = {"file_mb": cc.file_size(name)/1e6, "wake_sd_ms": sdm*1e3,
                     "wake_tmpfs_ms": shmm*1e3, "storage_ms": (sdm-shmm)*1e3,
                     "storage_frac": (sdm-shmm)/sdm if sdm>0 else None}
        print(f"[stor {name}] sd={sdm*1e3:.0f}ms tmpfs={shmm*1e3:.0f}ms "
              f"storage={100*(sdm-shmm)/sdm:.0f}%", flush=True)
        with open(os.path.expanduser("~/coldstart/results/e7a_storage.json"),"w") as f:
            json.dump(out, f, indent=2)
    return out

def opt_experiment():
    out = {}
    for name in OPT_MODELS:
        path = cc.model_path(name)
        rec = {"file_mb": cc.file_size(name)/1e6, "levels": {}}
        for opt in ["none","basic","all"]:
            cold, steady = [], []
            for _ in range(TRIALS):
                cc.drop_caches(3); gc.collect()
                r = wake_from(path, opt)
                cold.append(r["wake"]); steady.append(r["steady"])
            rec["levels"][opt] = {"cold_wake_ms": float(np.median(cold))*1e3,
                                  "steady_ms": float(np.median(steady))*1e3}
        # crossover B*: total(opt=all) < total(opt=none) when
        # cold_all + B*L_all < cold_none + B*L_none  => B* = (cold_all-cold_none)/(L_none-L_all)
        ca = rec["levels"]["all"]["cold_wake_ms"]; cn = rec["levels"]["none"]["cold_wake_ms"]
        la = rec["levels"]["all"]["steady_ms"];   ln = rec["levels"]["none"]["steady_ms"]
        rec["Bstar_all_vs_none"] = ((ca-cn)/(ln-la)) if (ln-la) > 1e-6 else None
        out[name] = rec
        bs = rec["Bstar_all_vs_none"]
        print(f"[opt {name}] cold none={cn:.0f}/all={ca:.0f}ms steady none={ln:.1f}/all={la:.1f}ms "
              f"B*={bs:.1f}" if bs else f"[opt {name}] B*=n/a", flush=True)
        with open(os.path.expanduser("~/coldstart/results/e7b_optlevel.json"),"w") as f:
            json.dump(out, f, indent=2)
    return out

def main():
    cc.set_freq_khz(2400000)
    print("=== E7a storage ===", flush=True); storage_experiment()
    print("=== E7b optlevel ===", flush=True); opt_experiment()
    cc.set_governor("schedutil")
    print("DONE E7", flush=True)

if __name__ == "__main__":
    main()
