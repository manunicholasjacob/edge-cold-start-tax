"""Run E1 decomposition + E2 warm-up + E6 DVFS for vit-base (after input-shape fix) and merge
into the existing result JSONs, so ViT-Base (transformer) joins the CNN/INT8 zoo."""
import os, sys, json, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cold_common as cc
from e1_decomposition import measure_once, agg
from e2_warmup import one_curve, analyze

NAME = "vit-base"
RES = os.path.expanduser("~/coldstart/results")

def merge(fname, key, rec):
    path = os.path.join(RES, fname)
    d = {}
    if os.path.exists(path):
        with open(path) as f: d = json.load(f)
    d[key] = rec
    with open(path, "w") as f: json.dump(d, f, indent=2)

def e6_vit():
    path = cc.model_path(NAME)
    def wake():
        cc.drop_caches(3); gc.collect(); time.sleep(0.2)
        t0 = cc.now()
        s = cc.new_session(path, intra_threads=4, graph_opt="all")
        s.run(None, cc.make_inputs(s)); t1 = cc.now(); del s; gc.collect()
        return t1-t0
    rec = {}
    for pol in ["schedutil","pinned_min","pinned_max"]:
        lat = []
        for _ in range(4):
            if pol=="schedutil":
                cc.set_governor("schedutil");
                t0=cc.now()
                while cc.now()-t0<2.0: time.sleep(0.05)
            elif pol=="pinned_min": cc.set_freq_khz(1500000)
            else: cc.set_freq_khz(2400000)
            lat.append(wake())
        rec[pol] = {"wake_ms": float(np.median(lat))*1e3}
    rt = rec["schedutil"]["wake_ms"]-rec["pinned_max"]["wake_ms"]
    rec["ramp_tax_ms"]=rt; rec["ramp_tax_pct"]=100*rt/rec["pinned_max"]["wake_ms"]
    return rec

def main():
    cc.set_freq_khz(2400000)
    print("pinned", cc.cur_freqs_khz(), flush=True)
    # E1
    runs=[]
    for k in range(5):
        try:
            r=measure_once(NAME); runs.append(r)
            print(f"[E1 {NAME}] t{k} wake={r['total_wake']*1e3:.0f}ms steady={r['steady']*1e3:.0f}ms tax={r['tax_ratio']:.1f}x",flush=True)
        except Exception as e: print(f"[E1 {NAME}] t{k} ERR {e}",flush=True)
    if runs:
        keys=["sess_create_cold","sess_create_warmfile","sess_create_noopt","first_infer","steady",
              "warm_first","disk_read","build_alloc","optimize","cold_compute","total_wake","tax_ratio","majflt","minflt"]
        merge("e1_decomposition.json", NAME,
              {"file_bytes":cc.file_size(NAME),"n":len(runs),**{k:agg(k,runs) for k in keys}})
        print("merged E1",flush=True)
    # E2
    curves=[]
    for k in range(3):
        try: curves.append(one_curve(NAME))
        except Exception as e: print(f"[E2 {NAME}] t{k} ERR {e}",flush=True)
    if curves:
        L=min(len(c) for c in curves); arr=np.stack([c[:L] for c in curves]); med=np.median(arr,axis=0)
        steady,n_conv,excess=analyze(med); first=float(med[0])
        merge("e2_warmup.json", NAME,
              {"median_curve_ms":[round(float(x)*1e3,4) for x in med],"steady_ms":steady*1e3,
               "first_infer_ms":first*1e3,"first_over_steady":first/steady if steady>0 else None,
               "n_conv":int(n_conv),"warmup_excess_ms":excess*1e3,"trials":len(curves)})
        print(f"[E2 {NAME}] steady={steady*1e3:.0f}ms first={first*1e3:.0f}ms N_conv={n_conv}",flush=True)
    # E6
    try:
        rec=e6_vit(); merge("e6_dvfs.json", NAME, rec)
        print(f"[E6 {NAME}] schedutil={rec['schedutil']['wake_ms']:.0f} pinned_max={rec['pinned_max']['wake_ms']:.0f} ramp_tax={rec['ramp_tax_pct']:.1f}%",flush=True)
    except Exception as e: print(f"[E6 {NAME}] ERR {e}",flush=True)
    cc.set_governor("schedutil")
    print("DONE VIT",flush=True)

if __name__ == "__main__":
    main()
