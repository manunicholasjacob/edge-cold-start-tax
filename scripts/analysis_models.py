"""
Analysis models (run locally on measured constants):
  (1) Predictive model of the cold-start tax from model file size + a platform constant.
      t_cold_wake ~= file_MB / BW_flash + t_build(file). We fit the disk term (BW) and a linear
      build+optimize term, and report R^2 / MAPE. Gives a deploy-time predictor needing only file
      size.
  (2) Duty-cycle amortization: overhead fraction of a wake vs inferences-per-wake B, in time and
      energy, and the B needed to push overhead below 10%. Plus the single-model keep-warm vs
      reload crossover.
Produces fig8_predict.png, fig9_amortize.png and prints stats.
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figs")
def L(n):
    p=os.path.join(RES,n); return json.load(open(p)) if os.path.exists(p) else None
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"savefig.bbox":"tight"})

e1=L("e1_decomposition.json"); e4=L("e4_energy.json")
SHORT={"squeezenet1.1":"SqueezeNet","shufflenet-v2":"ShuffleNetV2","mobilenetv2":"MobileNetV2",
       "resnet18":"ResNet-18","googlenet":"GoogLeNet","densenet":"DenseNet",
       "efficientnet-lite4":"EffNet-Lite4","resnet50":"ResNet-50","vit-base":"ViT-Base",
       "squeezenet1.1-int8":"SqueezeNet-i8","mobilenetv2-int8":"MobileNetV2-i8","resnet50-int8":"ResNet-50-i8"}
stats=[]
def P(s): stats.append(s); print(s)

# ---------- (1) predictive model ----------
def predictive():
    ms=[m for m in e1 if not m.startswith("_")]
    fmb=np.array([e1[m]["file_bytes"]/1e6 for m in ms])
    disk=np.array([e1[m]["disk_read"]["median"] for m in ms])       # s
    wake=np.array([e1[m]["total_wake"]["median"] for m in ms])      # s
    # disk term: linear through origin  disk = fmb / BW
    BW = float(np.sum(fmb*disk)/np.sum(disk*disk))**0  # placeholder to avoid div0; compute properly below
    # proper: disk = fmb / BW -> BW = sum(fmb^2)/sum(fmb*disk) via least squares disk = (1/BW) fmb
    slope = float(np.sum(fmb*disk)/np.sum(fmb*fmb))    # disk ≈ slope*fmb ; slope=1/BW
    BW = 1.0/slope
    disk_pred = slope*fmb
    ss_res=np.sum((disk-disk_pred)**2); ss_tot=np.sum((disk-disk.mean())**2)
    r2_disk=1-ss_res/ss_tot
    # full wake model: wake = a*fmb + b  (a captures disk+per-MB build, b fixed overhead)
    A=np.vstack([fmb,np.ones_like(fmb)]).T
    coef,_,_,_=np.linalg.lstsq(A,wake,rcond=None)
    wake_pred=A@coef
    mape=float(np.mean(np.abs((wake-wake_pred)/wake))*100)
    ss_res2=np.sum((wake-wake_pred)**2); r2_wake=1-ss_res2/np.sum((wake-wake.mean())**2)
    P(f"[predict] flash read BW = {BW:.0f} MB/s (R^2 disk={r2_disk:.3f})")
    P(f"[predict] wake = {coef[0]*1e3:.2f} ms/MB * file_MB + {coef[1]*1e3:.0f} ms ; "
      f"R^2={r2_wake:.3f} MAPE={mape:.1f}%")
    fig,ax=plt.subplots(1,2,figsize=(8.4,3.3))
    ax[0].scatter(fmb,disk*1e3,color="#2166ac")
    xs=np.linspace(0,fmb.max()*1.05,50); ax[0].plot(xs,slope*xs*1e3,color="#b2182b",
        label=f"{BW:.0f} MB/s (R²={r2_disk:.2f})")
    ax[0].set_xlabel("Model file size (MB)"); ax[0].set_ylabel("Weight-load time (ms)")
    ax[0].set_title("(a) Weight loading is flash-bandwidth bound"); ax[0].legend(frameon=False,fontsize=8)
    ax[1].scatter(wake*1e3,wake_pred*1e3,color="#1b7837")
    lim=[0,wake.max()*1e3*1.05]; ax[1].plot(lim,lim,ls="--",color="k",alpha=0.6)
    ax[1].set_xlabel("Measured cold wake (ms)"); ax[1].set_ylabel("Predicted (ms)")
    ax[1].set_title(f"(b) Size-only predictor (MAPE {mape:.0f}%)")
    fig.savefig(os.path.join(FIG,"fig8_predict.png")); plt.close(fig)
    return {"BW_MBps":BW,"r2_disk":r2_disk,"wake_slope_ms_per_mb":coef[0]*1e3,
            "wake_intercept_ms":coef[1]*1e3,"r2_wake":r2_wake,"mape":mape}

# ---------- (2) amortization ----------
def amortize():
    ms=[m for m in e1 if not m.startswith("_")]
    fig,ax=plt.subplots(1,2,figsize=(8.6,3.3))
    show=["squeezenet1.1","mobilenetv2","resnet18","resnet50","densenet"]
    cols=["#1b7837","#762a83","#e08214","#c51b7d","#01665e"]
    B=np.arange(1,201)
    b10_list=[]
    for m,c in zip(show,cols):
        cw=e1[m]["total_wake"]["median"]; s=e1[m]["steady"]["median"]
        # total time to deliver B results after a cold wake = cw + (B-1)*s ; useful = B*s
        overhead_frac=(cw - s)/(cw + (B-1)*s)
        ax[0].plot(B,overhead_frac*100,color=c,label=SHORT[m])
        b10=next((int(bb) for bb,of in zip(B,overhead_frac) if of<=0.10),None)
        b10_list.append((m,b10))
    ax[0].axhline(10,ls=":",color="grey"); ax[0].set_xscale("log")
    ax[0].set_xlabel("Inferences per wake, B"); ax[0].set_ylabel("Cold-start overhead (% of wake time)")
    ax[0].set_title("(a) Time overhead amortization"); ax[0].legend(frameon=False,fontsize=7.5)
    # energy amortization
    if e4:
        for m,c in zip(show,cols):
            if m not in e4: continue
            Ec=e4[m]["E_wake_cold_J"]; Ei=e4[m].get("E_inf_steady_J")
            if not Ei: continue
            of=(Ec - Ei)/(Ec + (B-1)*Ei)
            ax[1].plot(B,of*100,color=c,label=SHORT[m])
        ax[1].axhline(10,ls=":",color="grey"); ax[1].set_xscale("log")
        ax[1].set_xlabel("Inferences per wake, B"); ax[1].set_ylabel("Cold-start overhead (% of wake energy)")
        ax[1].set_title("(b) Energy overhead amortization"); ax[1].legend(frameon=False,fontsize=7.5)
    fig.savefig(os.path.join(FIG,"fig9_amortize.png")); plt.close(fig)
    P(f"[amortize] B to reach <10% time overhead: "+", ".join(f"{SHORT[m]}={b}" for m,b in b10_list))
    return {"b10":{m:b for m,b in b10_list}}

def main():
    if not e1: print("no e1"); return
    pm=predictive(); am=amortize()
    out={"predictive":pm,"amortize":am}
    json.dump(out,open(os.path.join(RES,"analysis_models.json"),"w"),indent=2)
    open(os.path.join(RES,"ANALYSIS.txt"),"w").write("\n".join(stats))
    print("wrote analysis_models.json, figs 8&9")

if __name__=="__main__":
    main()
