"""
E8 — ON-DEVICE GD-Tax residency, measured on real hardware (converts the E5 simulation into a
demonstration). A real multi-model server holds ONNX Runtime sessions resident up to a RAM budget
and serves a request trace; on a miss it builds the session (real disk read + graph build + first
inference) and, on eviction, releases both the session and the model file's page-cache residency
(posix_fadvise DONTNEED) as a memory-pressured device would. We measure real per-request energy
(PMIC) and latency under reload / LRU / LFU / GD-Tax.

Memory-safe by construction: resident footprint never exceeds the budget (< RAM); there is NO
anonymous hog (unlike E3b), so no thrash/OOM risk. Frequency pinned 2.4 GHz.
"""
import os, sys, json, gc, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cold_common as cc

RES = os.path.expanduser("~/coldstart/results")
DEPL = ["squeezenet1.1","mobilenetv2","googlenet","densenet","resnet18","resnet50"]
N_REQ = 250
BUDGETS_MB = [200, 320]
POLICIES = ["reload","lru","lfu","gd_tax"]

def load_costs():
    e4 = json.load(open(os.path.join(RES,"e4_energy.json")))
    foot = json.load(open(os.path.join(RES,"footprint.json")))
    C = {}
    for m in DEPL:
        penE = e4[m]["E_wake_cold_J"] - (e4[m]["E_inf_steady_J"] or 0)
        C[m] = {"penE": max(penE,1e-3), "m": foot[m]["resident_mb"] or (cc.file_size(m)/1e6)}
    return C

def zipf_trace(names, n, a=1.2, seed=0):
    rng=np.random.default_rng(seed); r=np.arange(1,len(names)+1); p=1/r**a; p/=p.sum()
    order=list(names); rng.shuffle(order); idx=rng.choice(len(order),size=n,p=p)
    return [order[i] for i in idx]

def surveillance_trace(names, costs, n, seed=0):
    rng=np.random.default_rng(seed)
    cheap=min(names,key=lambda m:costs[m]["penE"]); pricey=max(names,key=lambda m:costs[m]["penE"])
    mids=[m for m in names if m not in (cheap,pricey)]
    seq=[]
    for _ in range(n):
        r=rng.random()
        if r<0.75: seq.append(cheap)
        elif r<0.95: seq.append(mids[rng.integers(len(mids))])
        else: seq.append(pricey)
    return seq

def evict_file_cache(path):
    try:
        fd=os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception:
        pass

def run(policy, budget, trace, costs, ps):
    resident={}   # name -> {sess, last, H}
    used=0.0; clock=0; base=0.0
    freq={}; feeds_cache={}
    lat=[]; misses=0
    t_start=cc.now()
    for nm in trace:
        clock+=1; freq[nm]=freq.get(nm,0)+1
        path=cc.model_path(nm); m=costs[nm]["m"]
        if nm in resident and policy!="reload":
            a=cc.now(); resident[nm]["sess"].run(None, feeds_cache[nm]); lat.append(cc.now()-a)
            resident[nm]["last"]=clock
            if policy=="gd_tax": resident[nm]["H"]=base+freq[nm]*costs[nm]["penE"]/max(m,1)
        else:
            misses+=1
            a=cc.now()
            sess=cc.new_session(path, intra_threads=4, graph_opt="all")
            feeds=cc.make_inputs(sess); sess.run(None,feeds); lat.append(cc.now()-a)
            if policy!="reload" and m<=budget:
                # evict to fit
                while used+m>budget and resident:
                    if policy=="lru": v=min(resident,key=lambda k:resident[k]["last"])
                    elif policy=="lfu": v=min(resident,key=lambda k:(freq.get(k,0),resident[k]["last"]))
                    else: v=min(resident,key=lambda k:resident[k]["H"]); base=max(base,resident[v]["H"])
                    used-=costs[v]["m"]; evict_file_cache(cc.model_path(v))
                    del resident[v]["sess"]; del resident[v]; gc.collect()
                used+=m; feeds_cache[nm]=feeds
                resident[nm]={"sess":sess,"last":clock,
                              "H":base+freq[nm]*costs[nm]["penE"]/max(m,1)}
            else:
                evict_file_cache(path); del sess; gc.collect()
    t_end=cc.now()
    e=ps.energy_between(t_start,t_end)
    lat=np.array(lat)
    # free remaining
    for k in list(resident):
        del resident[k]["sess"];
    resident.clear(); gc.collect()
    return {"E_total_J": e["E_J"] if e else None, "dur_s": t_end-t_start,
            "E_per_req_J": (e["E_J"]/len(trace)) if e else None,
            "mean_lat_ms": float(lat.mean()*1e3), "p95_lat_ms": float(np.percentile(lat,95)*1e3),
            "misses": misses, "n": len(trace)}

def main():
    cc.set_freq_khz(2400000)
    costs=load_costs()
    total=sum(costs[m]["m"] for m in DEPL)
    print(f"deployment total resident = {total:.0f} MB; budgets={BUDGETS_MB}", flush=True)
    ps=cc.PowerSampler(period=0.02); ps.start(); time.sleep(1.0)
    out={"deployment":DEPL,"total_mb":total,"n_req":N_REQ,"runs":{}}
    traces={"zipf":zipf_trace(DEPL,N_REQ,seed=1),
            "surveillance":surveillance_trace(DEPL,costs,N_REQ,seed=1)}
    for wl,trace in traces.items():
        out["runs"][wl]={}
        for B in BUDGETS_MB:
            out["runs"][wl][str(B)]={}
            for pol in POLICIES:
                cc.drop_caches(3); gc.collect(); time.sleep(0.3)
                r=run(pol,B,trace,costs,ps)
                out["runs"][wl][str(B)][pol]=r
                print(f"[{wl} B={B} {pol:7s}] E/req={r['E_per_req_J']:.3f}J "
                      f"mean={r['mean_lat_ms']:.0f}ms p95={r['p95_lat_ms']:.0f}ms miss={r['misses']}", flush=True)
                json.dump(out,open(os.path.join(RES,"e8_ondevice.json"),"w"),indent=2)
    ps.stop(); cc.set_governor("schedutil")
    # headline
    for wl in out["runs"]:
        for B in BUDGETS_MB:
            d=out["runs"][wl][str(B)]
            if "lru" in d and "gd_tax" in d and d["lru"]["E_per_req_J"]:
                sav=100*(d["lru"]["E_per_req_J"]-d["gd_tax"]["E_per_req_J"])/d["lru"]["E_per_req_J"]
                print(f">> {wl} B={B}: gd_tax vs lru energy {sav:+.1f}%  "
                      f"(reload {d['reload']['E_per_req_J']:.3f} -> gd_tax {d['gd_tax']['E_per_req_J']:.3f} J/req)", flush=True)
    print("DONE E8", flush=True)

if __name__=="__main__":
    main()
