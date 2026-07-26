"""
E6 — DVFS ramp interaction at wake.
A duty-cycled device wakes from idle: the on-demand governor (schedutil) starts near the
minimum clock and ramps up while the *critical* first inference runs at a low frequency.
We compare cold-wake latency and energy under three clock policies:
  schedutil : default on-demand governor (cold clock, ramps during wake)
  pinned_min: userspace @ 1.5 GHz
  pinned_max: userspace @ 2.4 GHz
The "ramp tax" = wake latency(schedutil) - wake latency(pinned_max). Energy compares whether
running the wake at a lower clock trades latency for energy.
"""
import os, sys, json, gc, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cold_common as cc

TRIALS = 6
MODELS = ["mobilenetv2","resnet18","resnet50","densenet","efficientnet-lite4","vit-base"]

def idle_to_settle(secs=2.0):
    """Idle so schedutil drops the clock back down before a cold wake."""
    t0 = cc.now()
    while cc.now()-t0 < secs:
        time.sleep(0.05)

def wake(ps, path):
    gc.collect()
    time.sleep(0.2)
    t0 = cc.now()
    sess = cc.new_session(path, intra_threads=4, graph_opt="all")
    feeds = cc.make_inputs(sess)
    sess.run(None, feeds)
    t1 = cc.now()
    e = ps.energy_between(t0, t1)
    f_start = cc.cur_freqs_khz()  # freq right after wake (post-ramp)
    del sess; gc.collect()
    return (t1-t0), (e["E_J"] if e else None)

def run_policy(ps, name, policy):
    path = cc.model_path(name)
    lat, en = [], []
    for _ in range(TRIALS):
        if policy == "schedutil":
            cc.set_governor("schedutil"); idle_to_settle(2.0)
        elif policy == "pinned_min":
            cc.set_freq_khz(1500000)
        elif policy == "pinned_max":
            cc.set_freq_khz(2400000)
        cc.drop_caches(3)
        l, e = wake(ps, path)
        lat.append(l)
        if e: en.append(e)
    return float(np.median(lat)), (float(np.median(en)) if en else None)

def main():
    ps = cc.PowerSampler(period=0.02); ps.start()
    time.sleep(1.0)
    out = {}
    for name in MODELS:
        rec = {}
        for pol in ["schedutil","pinned_min","pinned_max"]:
            l, e = run_policy(ps, name, pol)
            rec[pol] = {"wake_ms": l*1e3, "wake_E_J": e}
        rt = rec["schedutil"]["wake_ms"] - rec["pinned_max"]["wake_ms"]
        rec["ramp_tax_ms"] = rt
        rec["ramp_tax_pct"] = 100.0*rt/rec["pinned_max"]["wake_ms"]
        out[name] = rec
        print(f"[{name}] schedutil={rec['schedutil']['wake_ms']:.0f}ms "
              f"pinned_max={rec['pinned_max']['wake_ms']:.0f}ms "
              f"ramp_tax={rt:.0f}ms ({rec['ramp_tax_pct']:.0f}%)", flush=True)
        with open(os.path.expanduser("~/coldstart/results/e6_dvfs.json"),"w") as f:
            json.dump(out, f, indent=2)
    ps.stop()
    cc.set_governor("schedutil")
    print("DONE E6", flush=True)

if __name__ == "__main__":
    main()
