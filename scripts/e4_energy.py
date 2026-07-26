"""
E4 (v2) — Energy per wake (PMIC), robust to short windows.

v1 integrated power inside a single short wake window; at the ~50 Hz PMIC sampling ceiling a
sub-100 ms wake captures too few samples and the trapezoidal integral collapses. v2 instead
estimates the MEAN POWER of each phase by pooling samples across MANY repeated operations, then
multiplies by the phase's measured median duration:
    E_phase = P_phase * t_phase,  P_phase = mean(power samples falling inside any phase window).
Phases: cold wake (drop_caches each time), warm wake (file pre-cached), steady inference.
Idle power measured over a quiet window. Rails: total board, VDD_CORE (CPU), DDR.
Frequency pinned 2.4 GHz.
"""
import os, sys, json, gc, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cold_common as cc

MODELS = ["squeezenet1.1","shufflenet-v2","mobilenetv2","resnet18","googlenet",
          "densenet","efficientnet-lite4","resnet50","vit-base",
          "squeezenet1.1-int8","mobilenetv2-int8","resnet50-int8"]

def iters_for(name):
    mb = cc.file_size(name)/1e6
    if mb > 200: return 6
    if mb > 60:  return 12
    return 20

def pool_power(ps, windows, rail=1):
    """mean of power samples (rail index into sample tuple: 1=total,2=core,3=ddr) inside windows."""
    vals = []
    for (t, tot, core, ddr) in ps.samples:
        for (a, b) in windows:
            if a <= t <= b:
                vals.append((tot, core, ddr)[rail-1]); break
    return float(np.mean(vals)) if vals else None, len(vals)

def measure_idle(ps, secs=4.0):
    t0 = cc.now()
    while cc.now()-t0 < secs: time.sleep(0.03)
    p,_ = pool_power(ps, [(t0, cc.now())], 1)
    pc,_ = pool_power(ps, [(t0, cc.now())], 2)
    return p, pc

def cold_or_warm(ps, path, cold, M):
    windows, durs = [], []
    for _ in range(M):
        if cold: cc.drop_caches(3)
        else:    cc.warm_file(path)
        gc.collect(); time.sleep(0.05)
        t0 = cc.now()
        sess = cc.new_session(path, intra_threads=4, graph_opt="all")
        feeds = cc.make_inputs(sess)
        sess.run(None, feeds)
        t1 = cc.now()
        windows.append((t0, t1)); durs.append(t1-t0)
        del sess; gc.collect()
    P, n = pool_power(ps, windows, 1)
    Pc, _ = pool_power(ps, windows, 2)
    t = float(np.median(durs))
    return {"P_W": P, "P_core_W": Pc, "t_s": t, "E_J": (P*t if P else None),
            "n_samples": n, "M": M}

def steady(ps, path, secs=6.0):
    cc.warm_file(path)
    sess = cc.new_session(path, intra_threads=4, graph_opt="all")
    feeds = cc.make_inputs(sess)
    for _ in range(3): sess.run(None, feeds)  # reach steady
    lat = []
    t0 = cc.now()
    while cc.now()-t0 < secs:
        a = cc.now(); sess.run(None, feeds); lat.append(cc.now()-a)
    t1 = cc.now()
    P, n = pool_power(ps, [(t0, t1)], 1)
    Pc, _ = pool_power(ps, [(t0, t1)], 2)
    del sess; gc.collect()
    L = float(np.median(lat))
    return {"P_W": P, "P_core_W": Pc, "lat_s": L, "E_inf_J": (P*L if P else None), "n_samples": n}

def main():
    cc.set_freq_khz(2400000)
    ps = cc.PowerSampler(period=0.012); ps.start()
    time.sleep(1.0)
    P_idle, P_idle_core = measure_idle(ps)
    out = {"_meta": {"P_idle_W": P_idle, "P_idle_core_W": P_idle_core}}
    print(f"P_idle={P_idle:.3f}W core={P_idle_core:.3f}W", flush=True)
    for name in MODELS:
        path = cc.model_path(name)
        M = iters_for(name)
        try:
            cold = cold_or_warm(ps, path, True, M)
            warm = cold_or_warm(ps, path, False, M)
            st   = steady(ps, path)
        except Exception as e:
            print(f"[{name}] ERR {e}", flush=True); continue
        rec = {"file_mb": cc.file_size(name)/1e6,
               "P_idle_W": P_idle,
               "E_wake_cold_J": cold["E_J"], "t_wake_cold_s": cold["t_s"], "P_cold_W": cold["P_W"],
               "E_wake_warm_J": warm["E_J"], "t_wake_warm_s": warm["t_s"], "P_warm_W": warm["P_W"],
               "E_inf_steady_J": st["E_inf_J"], "lat_steady_s": st["lat_s"], "P_steady_W": st["P_W"],
               "n_cold": cold["n_samples"], "n_warm": warm["n_samples"], "n_steady": st["n_samples"]}
        if rec["E_inf_steady_J"] and rec["E_wake_cold_J"]:
            rec["cold_wake_in_inferences"] = rec["E_wake_cold_J"]/rec["E_inf_steady_J"]
            rec["cold_over_warm_energy"] = (rec["E_wake_cold_J"]/rec["E_wake_warm_J"]
                                            if rec["E_wake_warm_J"] else None)
        out[name] = rec
        print(f"[{name}] E_cold={rec['E_wake_cold_J']:.2f}J E_warm={rec['E_wake_warm_J']:.2f}J "
              f"E_inf={rec['E_inf_steady_J']*1e3:.0f}mJ P_cold={rec['P_cold_W']:.1f}W "
              f"cold={rec.get('cold_wake_in_inferences',0):.0f}inf (nc={rec['n_cold']})", flush=True)
        with open(os.path.expanduser("~/coldstart/results/e4_energy.json"),"w") as f:
            json.dump(out, f, indent=2)
    ps.stop(); cc.set_governor("schedutil")
    print("DONE E4", flush=True)

if __name__ == "__main__":
    main()
