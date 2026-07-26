"""Generate all paper figures from result JSONs. Run locally (matplotlib)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figs")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})
C = {"disk":"#2166ac","build":"#4393c3","opt":"#92c5de","compute":"#d6604d",
     "a":"#1b7837","b":"#762a83","c":"#e08214","d":"#c51b7d","e":"#01665e","f":"#8c510a"}

def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

SHORT = {"squeezenet1.1":"SqueezeNet","shufflenet-v2":"ShuffleNetV2","mobilenetv2":"MobileNetV2",
         "resnet18":"ResNet-18","googlenet":"GoogLeNet","densenet":"DenseNet",
         "efficientnet-lite4":"EffNet-Lite4","resnet50":"ResNet-50","vit-base":"ViT-Base",
         "squeezenet1.1-int8":"SqueezeNet-i8","mobilenetv2-int8":"MobileNetV2-i8",
         "resnet50-int8":"ResNet-50-i8"}
ORDER = ["squeezenet1.1","shufflenet-v2","mobilenetv2","resnet18","googlenet","densenet",
         "efficientnet-lite4","resnet50","vit-base","squeezenet1.1-int8","mobilenetv2-int8","resnet50-int8"]

def fig1_tax(e1):
    ms = [m for m in ORDER if m in e1]
    tax = [e1[m]["tax_ratio"]["median"] for m in ms]
    order = np.argsort(tax)[::-1]
    ms = [ms[i] for i in order]; tax = [tax[i] for i in order]
    fig, ax = plt.subplots(figsize=(7,3.2))
    cols = ["#762a83" if "int8" in m else ("#e08214" if m=="vit-base" else "#2166ac") for m in ms]
    ax.bar(range(len(ms)), tax, color=cols)
    for i,t in enumerate(tax):
        ax.text(i, t+0.2, f"{t:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([SHORT[m] for m in ms], rotation=40, ha="right")
    ax.set_ylabel("Cold-start tax\n(first wake / steady latency)")
    ax.set_title("The cold-start tax: first-wake latency is 5–23$\\times$ steady-state")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2166ac",label="FP32 CNN"),Patch(color="#762a83",label="INT8"),
                       Patch(color="#e08214",label="ViT")], frameon=False, fontsize=8)
    fig.savefig(os.path.join(FIG,"fig1_tax.png")); plt.close(fig)

def fig2_decomp(e1):
    ms = [m for m in ORDER if m in e1]
    disk = np.array([e1[m]["disk_read"]["median"] for m in ms])*1e3
    opt  = np.array([e1[m]["optimize"]["median"] for m in ms])*1e3
    build= np.array([e1[m]["build_alloc"]["median"] for m in ms])*1e3 - opt
    comp = np.array([e1[m]["cold_compute"]["median"] for m in ms])*1e3
    build = np.clip(build,0,None)
    fig, ax = plt.subplots(figsize=(7,3.4))
    x = range(len(ms))
    ax.bar(x, disk, color=C["disk"], label="weight load (disk read)")
    ax.bar(x, build, bottom=disk, color=C["build"], label="graph build + arena alloc")
    ax.bar(x, opt, bottom=disk+build, color=C["opt"], label="graph optimization")
    ax.bar(x, comp, bottom=disk+build+opt, color=C["compute"], label="cold compute (1st infer)")
    ax.set_xticks(list(x)); ax.set_xticklabels([SHORT[m] for m in ms], rotation=40, ha="right")
    ax.set_ylabel("Cold-wake time (ms)")
    ax.set_title("Where the cold-start tax goes: weight loading dominates")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(os.path.join(FIG,"fig2_decomp.png")); plt.close(fig)

def fig3_warmup(e2):
    show = [m for m in ["resnet50","mobilenetv2","densenet","squeezenet1.1"] if m in e2]
    fig, ax = plt.subplots(figsize=(6,3.2))
    cs = [C["a"],C["b"],C["c"],C["d"]]
    for m,c in zip(show,cs):
        curve = np.array(e2[m]["median_curve_ms"])
        steady = e2[m]["steady_ms"]
        ax.plot(range(1,len(curve)+1), curve/steady, marker="o", ms=3, color=c, label=SHORT[m])
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlim(0.5, 20.5)
    ax.set_xlabel("Inference index after session creation")
    ax.set_ylabel("Latency / steady-state")
    ax.set_title("No compute warm-up: latency is at steady state by the 1st inference")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(FIG,"fig3_warmup.png")); plt.close(fig)

def fig4_cliff(e3b, e3s=None):
    fig, axs = plt.subplots(1, 2 if e3s else 1, figsize=(8.6 if e3s else 6.4, 3.4), squeeze=False)
    cs = [C["a"],C["b"],C["c"],C["d"],C["e"],C["f"]]
    ax = axs[0][0]
    for (m,c) in zip([k for k in ORDER if k in e3b], cs):
        cur = e3b[m]["curve"]
        S = [r["S_mb"] for r in cur]; w = [r["wake_ms"] for r in cur]
        ax.plot(S, w, marker="o", ms=3, color=c, label=f"{SHORT[m]} ({e3b[m]['file_mb']:.0f}MB)")
    ax.set_xlabel("Co-tenant anonymous footprint (MB)")
    ax.set_ylabel("Next-wake latency (ms)")
    ax.set_title("(a) Eviction cliff: anonymous pressure")
    ax.legend(frameon=False, fontsize=7)
    if e3s:
        ax2 = axs[0][1]
        for (m,c) in zip([k for k in ORDER if k in e3s], cs):
            cur = e3s[m]["curve"]
            P = [r["P_mb"] for r in cur]; w = [r["wake_ms"] for r in cur]
            ax2.plot(P, w, marker="s", ms=3, color=c, label=SHORT[m])
        ax2.set_xlabel("Use-once file reads streamed (MB)")
        ax2.set_ylabel("Next-wake latency (ms)")
        ax2.set_title("(b) Scan resistance: streaming pressure")
        ax2.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(FIG,"fig4_cliff.png")); plt.close(fig)

def fig7b_storage(e7a):
    ms=[m for m in ORDER if m in e7a]
    tm=np.array([e7a[m]["wake_tmpfs_ms"] for m in ms])
    st=np.array([e7a[m]["storage_ms"] for m in ms])
    fig,ax=plt.subplots(figsize=(6.8,3.3))
    x=range(len(ms))
    ax.bar(x, tm, color="#4393c3", label="build+optimize+compute (tmpfs)")
    ax.bar(x, st, bottom=tm, color="#2166ac", label="storage (microSD $-$ tmpfs)")
    ax.set_xticks(list(x)); ax.set_xticklabels([SHORT[m] for m in ms], rotation=40, ha="right")
    ax.set_ylabel("Cold-wake time (ms)")
    ax.set_title("Storage is the cause: microSD vs RAM-backed tmpfs")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(os.path.join(FIG,"fig7b_storage.png")); plt.close(fig)

def fig5_energy(e4):
    ms = [m for m in ORDER if m in e4 and not m.startswith("_")]
    size = np.array([e4[m]["file_mb"] for m in ms])
    ninf = np.array([e4[m].get("cold_wake_in_inferences",np.nan) for m in ms])
    fig, ax = plt.subplots(1,2, figsize=(8.4,3.3))
    ax[0].bar(range(len(ms)), [e4[m]["E_wake_cold_J"] for m in ms], color=C["disk"], label="cold wake")
    ax[0].bar(range(len(ms)), [e4[m]["E_wake_warm_J"] or 0 for m in ms], color=C["compute"], label="warm wake")
    ax[0].set_xticks(range(len(ms))); ax[0].set_xticklabels([SHORT[m] for m in ms], rotation=45, ha="right", fontsize=7)
    ax[0].set_ylabel("Wake energy (J)"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("(a) Energy per wake: cold vs warm")
    ax[1].scatter(size, ninf, color=C["b"])
    for m,s,n in zip(ms,size,ninf):
        if not np.isnan(n): ax[1].annotate(SHORT[m],(s,n),fontsize=6,xytext=(3,2),textcoords="offset points")
    ax[1].set_xlabel("Model size (MB)"); ax[1].set_ylabel("Cold wake energy\n(= N steady inferences)")
    ax[1].set_title("(b) A cold wake costs several–tens of inferences")
    fig.savefig(os.path.join(FIG,"fig5_energy.png")); plt.close(fig)

def fig6_dvfs(e6):
    ms = [m for m in ORDER if m in e6]
    su = [e6[m]["schedutil"]["wake_ms"] for m in ms]
    pmax=[e6[m]["pinned_max"]["wake_ms"] for m in ms]
    fig, ax = plt.subplots(figsize=(6.4,3.2))
    x = np.arange(len(ms)); w=0.38
    ax.bar(x-w/2, su, w, color=C["c"], label="schedutil (ramps from idle clock)")
    ax.bar(x+w/2, pmax, w, color=C["a"], label="pinned 2.4 GHz")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in ms], rotation=40, ha="right")
    ax.set_ylabel("Cold-wake latency (ms)")
    ax.set_title("DVFS ramp tax: on-demand governor lengthens the cold wake")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(FIG,"fig6_dvfs.png")); plt.close(fig)

def fig7_policy(e5):
    fig, axs = plt.subplots(1,2, figsize=(8.6,3.4))
    pol_style = {"reload":("Always-reload","#999999","o"),"lru":("LRU","#4393c3","s"),
                 "lfu":("LFU","#92c5de","^"),"gd_tax":("GD-Tax (ours)","#c51b7d","D"),
                 "belady":("Belady oracle","#1b7837","*")}
    for ax, wl, ylab, key, title in [
        (axs[0],"zipf","Energy per request (J)","energy_per_req_J","(a) Zipf workload — energy"),
        (axs[1],"zipf","p99 wake latency (ms)","p99_L_ms","(b) Zipf workload — tail latency")]:
        w = e5["workloads"][wl]
        tot = e5["total_resident_mb"]
        for pol,(lab,col,mk) in pol_style.items():
            rows = w[pol]
            xs = [100*r["B_mb"]/tot for r in rows]; ys=[r[key] for r in rows]
            ax.plot(xs, ys, marker=mk, color=col, label=lab, ms=5)
        ax.set_xlabel("RAM budget (% of all-resident)")
        ax.set_ylabel(ylab); ax.set_title(title)
    axs[0].legend(frameon=False, fontsize=7.5)
    fig.savefig(os.path.join(FIG,"fig7_policy.png")); plt.close(fig)

def fig10_ondevice(e8):
    wls=list(e8["runs"].keys())
    budgets=list(next(iter(e8["runs"].values())).keys())
    pol_lab={"reload":("Always-reload","#999999"),"lru":("LRU","#4393c3"),
             "lfu":("LFU","#92c5de"),"gd_tax":("GD-Tax (ours)","#c51b7d")}
    fig,axs=plt.subplots(1,len(wls),figsize=(8.6,3.4),squeeze=False)
    for j,wl in enumerate(wls):
        ax=axs[0][j]; x=np.arange(len(budgets)); w=0.2
        for i,(pol,(lab,col)) in enumerate(pol_lab.items()):
            ys=[e8["runs"][wl][b][pol]["E_per_req_J"] for b in budgets]
            ax.bar(x+(i-1.5)*w, ys, w, color=col, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([f"{b} MB" for b in budgets])
        ax.set_xlabel("RAM budget"); ax.set_ylabel("Energy per request (J), measured")
        ax.set_title(f"({chr(97+j)}) {wl} workload (on-device)")
    axs[0][0].legend(frameon=False,fontsize=7.5)
    fig.savefig(os.path.join(FIG,"fig10_ondevice.png")); plt.close(fig)

def main():
    e1=load("e1_decomposition.json"); e2=load("e2_warmup.json")
    e3b=load("e3b_cliff.json"); e3s=load("e3_streaming.json")
    e4=load("e4_energy.json"); e5=load("e5_residency.json"); e6=load("e6_dvfs.json")
    if e1: fig1_tax(e1); fig2_decomp(e1); print("fig1,2 ok")
    if e2: fig3_warmup(e2); print("fig3 ok")
    if e3b: fig4_cliff(e3b, e3s); print("fig4 ok")
    e7a=load("e7a_storage.json")
    if e7a: fig7b_storage(e7a); print("fig7b ok")
    if e4: fig5_energy({k:v for k,v in e4.items() if not k.startswith("_")}); print("fig5 ok")
    if e6: fig6_dvfs(e6); print("fig6 ok")
    if e5: fig7_policy(e5); print("fig7 ok")
    e8=load("e8_ondevice.json")
    if e8: fig10_ondevice(e8); print("fig10 ok")
    print("FIGS DONE")

if __name__ == "__main__":
    main()
