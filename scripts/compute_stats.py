"""Compute all paper-ready headline numbers from the result JSONs."""
import os, json, numpy as np
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
def L(n):
    p=os.path.join(RES,n)
    return json.load(open(p)) if os.path.exists(p) else None

e1=L("e1_decomposition.json"); e2=L("e2_warmup.json"); e4=L("e4_energy.json")
e3s=L("e3_streaming.json"); e3b=L("e3b_cliff.json"); e6=L("e6_dvfs.json"); e5=L("e5_residency.json")
foot=L("footprint.json")
out=[]
def p(s): out.append(s); print(s)

if e1:
    tax={m:e1[m]["tax_ratio"]["median"] for m in e1}
    p(f"[E1] tax range: {min(tax.values()):.1f}x ({min(tax,key=tax.get)}) .. {max(tax.values()):.1f}x ({max(tax,key=tax.get)})")
    fr=[]
    for m in e1:
        tw=e1[m]["total_wake"]["median"]; dr=e1[m]["disk_read"]["median"]
        fr.append(100*dr/tw)
    p(f"[E1] disk-read fraction of wake: {min(fr):.0f}%..{max(fr):.0f}% (mean {np.mean(fr):.0f}%)")
    # cold_compute fraction
    cc=[100*e1[m]["cold_compute"]["median"]/e1[m]["total_wake"]["median"] for m in e1]
    p(f"[E1] cold-compute fraction: {max(cc):.1f}% max (mean {np.mean(cc):.2f}%)")
    # effective SD read bandwidth from resnet50
    if "resnet50" in e1:
        mb=e1["resnet50"]["file_bytes"]/1e6; dr=e1["resnet50"]["disk_read"]["median"]
        p(f"[E1] effective flash read BW (resnet50): {mb/dr:.0f} MB/s")
    # int8 vs fp32 tax
    for a,b in [("mobilenetv2","mobilenetv2-int8"),("resnet50","resnet50-int8"),("squeezenet1.1","squeezenet1.1-int8")]:
        if a in e1 and b in e1:
            p(f"[E1] {a}: tax {tax[a]:.1f}x wake {e1[a]['total_wake']['median']*1e3:.0f}ms | "
              f"{b}: tax {tax[b]:.1f}x wake {e1[b]['total_wake']['median']*1e3:.0f}ms")
if e2:
    fos=[e2[m]["first_over_steady"] for m in e2 if e2[m].get("first_over_steady")]
    p(f"[E2] first/steady ratio: {min(fos):.2f}..{max(fos):.2f} (mean {np.mean(fos):.2f}) -> no compute warm-up")
    nconv=[e2[m]["n_conv"] for m in e2]
    p(f"[E2] N_conv (inferences to steady): median {int(np.median(nconv))}, max {max(nconv)}")
if e4:
    ms=[m for m in e4 if not m.startswith("_")]
    ninf=[e4[m].get("cold_wake_in_inferences") for m in ms if e4[m].get("cold_wake_in_inferences")]
    if ninf: p(f"[E4] cold wake = {min(ninf):.0f}..{max(ninf):.0f} steady inferences of energy")
    p(f"[E4] P_idle={e4['_meta']['P_idle_W']:.2f}W")
    for m in ms:
        r=e4[m]
        p(f"     {m}: E_cold={r['E_wake_cold_J']:.2f}J E_warm={r.get('E_wake_warm_J')}J "
          f"E_inf={ (r['E_inf_steady_J']*1e3) if r.get('E_inf_steady_J') else None}mJ "
          f"P_cold={r.get('P_cold_W')}W")
if e3b:
    for m in e3b:
        cur=e3b[m]["curve"]
        base=cur[0]["wake_ms"]; top=max(c["wake_ms"] for c in cur)
        # cliff: first S where wake > 2*base
        cliff=None
        for c in cur:
            if c["wake_ms"]>1.8*base: cliff=c["S_mb"]; break
        p(f"[E3b] {m} ({e3b[m]['file_mb']:.0f}MB): warm={base:.0f}ms peak={top:.0f}ms cliff@S={cliff}MB")
if e3s:
    for m in e3s:
        cur=e3s[m]["curve"]
        w=[c["wake_ms"] for c in cur]
        p(f"[E3-stream] {m}: wake {min(w):.0f}..{max(w):.0f}ms across {cur[0]['P_mb']}..{cur[-1]['P_mb']}MB streamed (flat=scan-resistant)")
if e6:
    rt=[e6[m]["ramp_tax_pct"] for m in e6]
    p(f"[E6] DVFS ramp tax: {min(rt):.1f}%..{max(rt):.1f}% (mean {np.mean(rt):.1f}%) -> wake is I/O-bound")
if e5:
    for wl in e5["workloads"]:
        b=1
        rel=e5["workloads"][wl]["reload"][b]; lru=e5["workloads"][wl]["lru"][b]
        gd=e5["workloads"][wl]["gd_tax"][b]; bel=e5["workloads"][wl]["belady"][b]
        p(f"[E5/{wl}] @{100*rel['B_mb']/e5['total_resident_mb']:.0f}% budget: "
          f"reload={rel['energy_per_req_J']:.2f} lru={lru['energy_per_req_J']:.2f} "
          f"gd_tax={gd['energy_per_req_J']:.2f} belady={bel['energy_per_req_J']:.2f} J/req | "
          f"gd vs lru {100*(lru['energy_per_req_J']-gd['energy_per_req_J'])/lru['energy_per_req_J']:.0f}% save, "
          f"p99 gd={gd['p99_L_ms']:.0f}ms lru={lru['p99_L_ms']:.0f}ms reload={rel['p99_L_ms']:.0f}ms")
open(os.path.join(RES,"STATS.txt"),"w").write("\n".join(out))
print("\nwrote STATS.txt")
