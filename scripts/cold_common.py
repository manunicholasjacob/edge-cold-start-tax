"""
cold_common.py — shared harness for the Cold-Start Tax campaign (Raspberry Pi 5 / Cortex-A76).

Provides:
  - model input construction (derives shape/dtype from the ONNX graph)
  - page-cache control (drop_caches; per-file warm)
  - high-resolution timing
  - rusage-based page-fault accounting (major = disk reads, minor = zero-fill/COW)
  - PMIC power sampler (background thread; per-rail V*I -> Watts)
  - CPU frequency / governor control
All timing uses time.perf_counter(); all sub-shell privileged ops go through sudo (NOPASSWD).
"""
import os, sys, time, json, subprocess, threading, resource, gc
import numpy as np

MODELS_DIR = os.path.expanduser("~/coldstart/models")

# ---- model zoo (name -> filename). fp32 first, int8 variants tagged. ----
ZOO = {
    "squeezenet1.1":   "squeezenet1.1-7.onnx",
    "shufflenet-v2":   "shufflenet-v2-10.onnx",
    "mobilenetv2":     "mobilenetv2-12.onnx",
    "resnet18":        "resnet18-v1-7.onnx",
    "googlenet":       "googlenet-12.onnx",
    "densenet":        "densenet-12.onnx",
    "efficientnet-lite4":"efficientnet-lite4-11.onnx",
    "resnet50":        "resnet50-v1-7.onnx",
    "vit-base":        "vit-base-16.onnx",
    # int8 (static, SDOT) variants
    "squeezenet1.1-int8": "squeezenet1.1-7-int8s.onnx",
    "mobilenetv2-int8":   "mobilenetv2-12-int8s.onnx",
    "resnet50-int8":      "resnet50-v1-7-int8s.onnx",
}

def model_path(name):
    return os.path.join(MODELS_DIR, ZOO[name])

def file_size(name):
    return os.path.getsize(model_path(name))

# ---- privileged helpers (sudo is NOPASSWD on this Pi) ----
def drop_caches(level=3):
    """1=pagecache, 2=dentries+inodes, 3=both. Full cold start uses 3."""
    subprocess.run(["sync"], check=True)
    subprocess.run(["sudo", "sh", "-c", f"echo {level} > /proc/sys/vm/drop_caches"], check=True)

def warm_file(path):
    """Pull a file into the page cache (read-only), so session-create pays no disk I/O."""
    with open(path, "rb") as f:
        while f.read(1 << 20):
            pass

def set_governor(gov):
    subprocess.run(["sudo", "sh", "-c",
        f"for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo {gov} > $c; done"],
        check=True)

def set_freq_khz(khz):
    """Requires userspace governor. Pins all cores to khz."""
    set_governor("userspace")
    subprocess.run(["sudo", "sh", "-c",
        f"for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed; do echo {khz} > $c; done"],
        check=True)

def cur_freqs_khz():
    out = subprocess.check_output(
        ["sh", "-c", "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"]).decode()
    return [int(x) for x in out.split()]

# ---- input construction ----
def make_inputs(sess):
    feeds = {}
    rng = np.random.default_rng(1234)
    for inp in sess.get_inputs():
        raw = inp.shape
        shape = []
        for i, d in enumerate(raw):
            if isinstance(d, int) and d > 0:
                shape.append(d)
            elif i == 0:
                shape.append(1)                       # batch
            elif len(raw) == 4 and i == 1:
                shape.append(3)                       # NCHW channel (symbolic)
            else:
                shape.append(224)                     # symbolic spatial
        # NHWC image tensors declare channels last; only override when concrete dims say so
        if len(shape) == 4 and shape[1] not in (1, 3) and shape[3] in (1, 3):
            pass  # already concrete NHWC, leave as-is
        t = inp.type
        if "float16" in t:
            dt = np.float16
        elif "float" in t:
            dt = np.float32
        elif "int64" in t:
            dt = np.int64
        elif "int32" in t:
            dt = np.int32
        elif "uint8" in t:
            dt = np.uint8
        else:
            dt = np.float32
        if np.issubdtype(dt, np.floating):
            feeds[inp.name] = rng.standard_normal(shape).astype(dt)
        else:
            feeds[inp.name] = np.ones(shape, dtype=dt)
    return feeds

# ---- rusage faults ----
def faults():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_majflt, r.ru_minflt  # major (disk), minor (no-disk)

# ---- PMIC power sampler ----
# rails we care about; VDD_CORE = CPU core, DDR_* = DRAM. We integrate total board-ish
# power from the rails exposed by pmic_read_adc (V*I per rail).
class PowerSampler:
    def __init__(self, period=0.02):
        self.period = period
        self._stop = threading.Event()
        self.samples = []  # (t, total_W, core_W, ddr_W)
        self._thr = None

    @staticmethod
    def _read():
        out = subprocess.check_output(["vcgencmd", "pmic_read_adc"]).decode()
        volts, amps = {}, {}
        for line in out.strip().splitlines():
            line = line.strip()
            # format: NAME_V volt(n)=1.23V   or  NAME_A current(n)=0.45A
            if "volt(" in line:
                name = line.split()[0][:-2]  # strip trailing _V
                val = float(line.split("=")[1].rstrip("V"))
                volts[name] = val
            elif "current(" in line:
                name = line.split()[0][:-2]  # strip trailing _A
                val = float(line.split("=")[1].rstrip("A"))
                amps[name] = val
        total = core = ddr = 0.0
        for k in volts:
            if k in amps:
                p = volts[k] * amps[k]
                total += p
                if k.startswith("VDD_CORE"):
                    core += p
                elif k.startswith("DDR"):
                    ddr += p
        return total, core, ddr

    def _loop(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                total, core, ddr = self._read()
                self.samples.append((t, total, core, ddr))
            except Exception:
                pass
            time.sleep(self.period)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2)

    def energy_between(self, t0, t1):
        """Trapezoidal integral of total power over [t0,t1] -> Joules; plus mean rails."""
        pts = [(t, tot, c, d) for (t, tot, c, d) in self.samples if t0 <= t <= t1]
        if len(pts) < 2:
            return None
        E = Ecore = Eddr = 0.0
        for i in range(1, len(pts)):
            dt = pts[i][0] - pts[i-1][0]
            E    += 0.5*(pts[i][1]+pts[i-1][1])*dt
            Ecore+= 0.5*(pts[i][2]+pts[i-1][2])*dt
            Eddr += 0.5*(pts[i][3]+pts[i-1][3])*dt
        dur = pts[-1][0]-pts[0][0]
        return {"E_J": E, "E_core_J": Ecore, "E_ddr_J": Eddr,
                "P_avg_W": E/dur if dur>0 else None,
                "P_core_W": Ecore/dur if dur>0 else None,
                "n": len(pts), "dur_s": dur}

def new_session(path, intra_threads=4, graph_opt="all"):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads
    so.inter_op_num_threads = 1
    lvl = {"all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
           "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
           "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}[graph_opt]
    so.graph_optimization_level = lvl
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])

def now():
    return time.perf_counter()
