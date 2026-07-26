# Cover Letter — IEEE Internet of Things Journal

**Manuscript:** *The Cold-Start Tax: Warm-Up, Re-Warm, and Residency Policy for Duty-Cycled Edge DNN Inference*
**Author:** Manu Nicholas Jacob

Dear Editors,

Please consider the attached manuscript for publication in the *IEEE Internet of Things Journal*.

Edge-inference research and practice almost universally report **steady-state** latency and
energy — the numbers a model reaches after it has been loaded and warmed. Yet a very large class
of IoT devices does not run inference in a hot loop; they **duty-cycle** — waking on an event or
timer, running a few inferences, and returning to a low-power state. For these devices the
transient at every wake, not the steady state, dominates both latency and energy. This manuscript
provides, to our knowledge, the first systematic characterization of that transient — the
**cold-start tax** — on a representative edge platform (Raspberry Pi 5, ARM Cortex-A76), and
turns the measurements into deployable methods.

**Contributions and why they fit IoT-J:**
1. A hardware measurement of the cold-start tax (**5–23×** the steady-state latency across twelve
   models) and a decomposition showing it is a **weight-loading** cost (55–81% of the wake), not
   a compute warm-up — the first inference already runs at steady-state speed.
2. A **size-only predictive model** of the tax (flash bandwidth R²=1.00; cold-wake latency MAPE
   12%), and a causal control (tmpfs vs microSD) confirming the tax is storage-bound.
3. The **energy of a wake** measured from the Pi 5's on-board PMIC (a cold wake costs 4–19
   steady-state inferences of energy; up to 35 J for a vision transformer).
4. An **eviction cliff**: on a 2 GB device the tax re-triggers when a co-tenant's working set
   evicts the model's pages, degrading the wake catastrophically (>80×) via swap thrashing —
   while use-once file I/O is shown not to (page-cache scan resistance).
5. Two deployable methods: a graph-optimization policy for short wakes, and **GD-Tax**, a
   tax-aware model-residency cache that consistently beats LRU/LFU and matches an offline Belady
   reference on measured constants.

The work is directly relevant to the IoT-J readership building energy- and memory-constrained
intelligent endpoints, and it argues concretely for reporting and optimizing the **cold wake**
rather than the steady state. All results are reproducible from an open, scripted artifact
(code, raw measurements, and figures) on commodity hardware.

This manuscript is original, has not been published previously, and is not under consideration
elsewhere. The author has no conflicts of interest to declare.

Thank you for your consideration.

Sincerely,
Manu Nicholas Jacob
