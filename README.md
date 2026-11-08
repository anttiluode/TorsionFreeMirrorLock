# Torsion-Free Mirror Lock

**Hiding information from the doors of a network without any symmetry, and what it costs.**

[Friday's mirror lock](https://github.com/anttiluode/Friday#writing-behind-the-mirror-added-7-november-2026) stores bits behind a reflection S with S² = 1. Its dark sector is the image of the idempotent (1 − S)/2. In group-ring language, that is the *torsion* way to get a hidden sector: an element of finite order makes the projector for free. That is exactly the case Kaplansky's conjectures exclude.

Mghirbi and Douglas ([arXiv:2611.04599](https://arxiv.org/abs/2611.04599)) build non-trivial idempotents in groups with no finite-order element at all. Their cancellation comes from counting and a congruence, not from a symmetry, and Section 8 shows that over the reals it collides with the property the construction needs. This repository asks the wave-network version of that question:

> Can a network hide information from its public ports with **no symmetry at all**, and can that hiding survive the network's own plasticity?

**Short answer (corrected 8 November after review):**

- **The hiding exists.** It appears generically once the interior has more weights than the ports can constrain, and it is exact at every frequency. Nothing is dynamically dark: all 14 modes reach the ports. What is hidden is *which* internal weighting produced them. This is parameter non-identifiability, not an unobservable state.
- **Two kinds of plasticity are exactly safe:** any learning on port-to-port edges, and a common gain on the whole hidden network whose rate is computed from public signals (synaptic scaling). A plastic system combining the two keeps the lock exact for 500 steps, and a scale-invariant key still decodes every word (F2).
- **The strain rules I tested leak once they write any edge with an interior endpoint.** They then shrink the distance between stored words, but slowly: after 50 writes all 64 words are still perfectly recoverable by a recalibrated reader. It took about 300 writes (at 10% noise) to 1000 writes (at 1% noise) before recovery failed (F1). I first called this "erasure", which was wrong. A different interior rule could behave better; I only tested these.

**About the name.** It is evocative, not a theorem. The paper's torsion-freeness is a property of a group. Here, the maps relating hidden states do not even compose: T² applied to a stored graph is no longer a valid weighting of it (F4). The accurate description is **storage in the fibers of a boundary transfer map**.

*Built by Claude (Opus 5.5) on 7 November 2026, from Antti Luode's Friday repository and his request to explore the "paper-shaped" version. Reviewed by Astra (GPT-6) on 8 November; its corrections and the follow-up are in [Review and follow-up](#review-and-follow-up-8-november). Ideal linear model; CPU only.*

## The picture

Think of a house where the public stands only in the hallway (the port nodes). It can shout and listen at any pitch, and it can reshape the hallway (edges between ports). The rooms behind the doors (the interior) can be arranged in a continuum of different ways that sound *exactly* the same from the hallway. No mirror makes them sound the same. There are simply more ways to build the rooms than there are things the hallway can hear.

Reshaping the hallway never tells you which arrangement you are in. Touching a single hinge on a door to the rooms does.

## Setup

- **Graph:** 4 ports and 10 interior nodes, with 64 edges of three kinds:
  - 6 **boundary** edges (port–port);
  - 20 **port-interior** edges (each port wired to 5 random interior nodes);
  - 38 **bulk** edges (interior–interior, density 0.8).
- **Weights:** random in [0.5, 1.5], so the weighted graph has no automorphism.
- **Dynamics:** dx/dt = −iLx − γx + drive, with γ = 0.2, read in steady state. A drive at frequency ω gives A = G(ω)b, with G(ω) = (γ + i(L − ω))⁻¹.
- **Public:** drives and reads only port nodes, so everything it can ever learn is G_PP(ω), the port block. Fibers are built from 40 frequencies and verified on 97 held-out ones.
- **Port map:** the hidden weights (port-interior and bulk) mapped to G_PP(ω). Its **fibers** are weight settings no port experiment can tell apart.
- **Key:** holds every port at zero (clamping) and drives and listens at interior nodes. With the ports clamped, the interior evolves under L_II alone, so this reading cannot depend on the boundary weights.
- **Public plasticity:** SliderV3's strain rule, E = 0.48·tanh(|A_a − A_c|²/0.25), additive or multiplicative, applied to a chosen set of edge kinds.

## Results

All numbers come from `python run_all.py` and are in [results/results.json](results/results.json).

### E1. Hidden directions come from counting, not symmetry

| Bulk density | Hidden weights | Rank of the port map | Fiber dimension |
|---|---|---|---|
| 1.0 | 65 | 49 | 16 |
| 0.9 | 62 | 49 | 13 |
| 0.8 | 58 | 49 | **9** |
| 0.7 | 55 | 49 | 6 |
| 0.6 | 48 | 48 | 0 |

The ports constrain about 49 numbers, whatever the interior. Every hidden weight beyond that adds a direction they cannot see. The gap in singular values is clean: the dropped ones are at most 3e-15, and the smallest kept ones are 1e-7 to 1e-4. That tail of nearly-dark but visible directions is Friday's point 3 again: visible is not the same as readable.

### E2. A walked pair: invisible at the ports, visible to the key

I moved the hidden weights along the fiber (largest single weight change 0.33) and measured:

| Check | Value |
|---|---|
| Port difference at 97 held-out frequencies | **3e-13** |
| Clamped-key difference | **0.17** |
| Sorted weight lists differ by | 0.09 (so not a relabelling) |
| Full Laplacian spectra differ by | 2e-12 (isospectral) |

**What the hidden variable is.** There is an orthogonal matrix T on the interior with T L_II Tᵀ = L_II′ (residual 1e-12) and T L_IP = L_IP′ (residual 4e-12). This is Kalman's realization theorem: the port transfer function fixes the interior only up to a change of basis. The fiber is that basis-change orbit, intersected with the matrices that are still Laplacians of the same graph.

T is not a permutation. 19 of its 100 entries exceed 0.1. Its eigen-angles are 0.0027π, 0.033π, 0.059π and 0.115π (each in a ± pair, plus two at zero), and they vary continuously along the walk.

~~These T are generically of infinite order. That is the literal sense in which this lock is torsion-free.~~ **Retracted.** Rounded eigen-angles cannot establish infinite order. More importantly, T does not act on the admissible set: T²L T⁻² puts weight up to 0.036 on edges that do not exist and creates positive off-diagonals (F4). The fiber is not an orbit of any group of graph maps.

### E3. The no-go: only the boundary can be edited without leaking

I applied the *same* edit to both fiber partners and measured the port difference afterwards:

| Edit | Port difference after |
|---|---|
| +0.05 on one boundary edge | 3e-13 (all 5 tried) |
| +0.05 on one port-interior edge | 2e-3 to 1e-2 |
| +0.05 on one bulk edge | 2e-3 to 8e-3 |
| Scale every weight by 1.05 | 3e-13 |

The leak happens even though the edit is predetermined and reads nothing. It is a property of the edge, not of a learning rule.

The space of shared first-order edits that keeps a pair equivalent has dimension 20 for one pair. Across 2, 3 and 4 partners it drops to **6**, and that space lies **100% in the 6 boundary edges**.

**Why the boundary is exactly safe.** An edit Δ on port–port edges changes only the port block, and then

G′_PP = (G_PP⁻¹ + iΔ_PP)⁻¹,

a function of G_PP alone (Schur complement / Kron reduction; tested in `test_fiberlock.py`). Scaling the whole hidden network by a common factor is also safe (F2). Fiber partners are related by S = diag(I_P, T) with T𝟏 = 𝟏, so they have the same port degrees. Scaling every hidden edge by α maps L_II → αL_II and L_IP → αL_IP, and shifts L_PP by the same amount in both partners. Scaling only the bulk, or only the port-interior edges, leaks (1.2% and 4.6% for α = 1.3).

These are *additive, shared* edits. Scaling is not one of them, since its increment depends on the current weights, so the "only the boundary" count above does not cover it.

With a complete bulk graph there is one more safe edit: the same shift on every bulk edge, which is c(n·I − J) on the interior and commutes with every basis change of this kind. That is global again.

This is *loosely* reminiscent of the paper's Section 8, where an exact cancellation not built on symmetry conflicts with a property the construction needs. It is an analogy only. Section 8 proves an obstruction for one specific construction and implies nothing about plasticity in networks.

### E4. Public plasticity: boundary-only is exact; the tested interior rules leak

Same 300 random public writes (1 to 4 ports, random complex amplitudes and frequencies), applied to both fiber partners. Each cell shows port difference / key difference:

| Write scope | After 1 write | After 10 | After 50 | After 300 |
|---|---|---|---|---|
| Boundary only (additive or multiplicative) | 4e-13 / 0.166 | 6e-13 / 0.166 | 5e-13 / 0.166 | **6e-14 / 0.166** |
| Boundary + port-interior, additive | 4.5e-3 / 0.166 | 4e-2 / 0.163 | 6e-2 / 0.101 | 1e-3 / 0.023 |
| Everything, additive | 5e-3 / 0.165 | 6e-2 / 0.157 | 5e-2 / 0.022 | 1e-4 / 0.008 |

(Multiplicative versions behave the same way; see the JSON.)

Bulk-reaching plasticity first **reveals** the secret at the ports (up to about 7%). It then **contracts** the difference: both partners drift toward the same weights, and the key difference falls from 0.17 to about 0.01.

~~It then erases it.~~ Corrected: with a recalibrated reader, the information survives much longer than this suggests (F1).

### E5. Storing words

I chose six orthonormal fiber directions and walked to all 64 sign patterns (walk distance 0.8):

| | Result |
|---|---|
| Largest port difference across the 64 words | **3e-12** |
| Smallest pairwise key difference | 0.069 |
| Key decoding (nearest template), noise 0 / 0.3 / 1.0 / 3.0 × signal RMS | 100% / 100% / 99.96% / 76% |
| After 300 boundary-only public writes: port difference across words | 3e-13 |
| After 300 boundary-only public writes: decoding with the original templates | 100% |
| After 50 public writes reaching the bulk: port difference across words | **0.15** (now publicly distinguishable) |
| After 50 public writes reaching the bulk: decoding with the original templates | 3% |
| ...the same, with templates recalibrated to the new weights (Astra; F1) | **100%** |

The 3% shows the *old reader* is miscalibrated, not that the words are gone (see F1).

The noise tolerance comes from reading 100 interior node pairs at 40 frequencies: 4,000 complex numbers for 6 bits. F3 shows how far a compact reader gets.

### E6. The key cannot write locally

One SliderV3 write driven at an interior node changes the public port response by 15–20%. Local writes move *off* the fiber, so the public sees that something changed. The words above were written by fiber walks, which use the model and are not local.

## The two locks side by side

| | Mirror lock (Friday) | Torsion-free lock (here) |
|---|---|---|
| Why the ports are blind | graph symmetry S, S² = 1 | dimension counting (Kalman non-uniqueness); relating maps form no group |
| Hidden variable | one sign per ring (Z₂) | a point on a 9-dimensional manifold |
| Dark modes? | yes, 15 in the symmetric graph | none; all 14 modes reach the ports |
| Exact port invisibility | yes | yes |
| Public plasticity on boundary edges | safe | safe (Schur complement) |
| Public plasticity in the bulk | safe (equivariance) | common hidden gain: safe; tested strain rules: leak, then slowly contract |
| Key read | public drive, listen at interior twins | clamp ports, listen inside |
| Key write | local (drive one twin) | non-local (fiber walk) |
| Survives noise in the weights? | no (breaks the symmetry) | no (moves off the fiber) |

Symmetry buys robustness to every equivariant rule. Without symmetry you get more hidden capacity, and so far only two kinds of plasticity are known to keep it exactly: port-to-port learning and a common hidden gain.

## Honest limits

- **Ideal linear model.** Independent drift in the interior weights, from noise, temperature or fabrication, moves the graph off its fiber and leaks. Common gain drift does not (F2).
- **The key needs privileges.** Clamping the ports is an operation on public nodes. A key that only listens inside, without clamping, would see boundary changes mixed in.
- **Writing needs the model.** Words are written by fiber walks computed from full knowledge of the weights. I found no local rule that moves along a fiber, but I did not search systematically.
- **Recalibration is an oracle.** F1's recalibrated reader is given labeled readings for all 64 words after the history. That proves the information survives. It does not show that an autonomous reader can track the drift. Because the dynamics are deterministic and the public history is public, a reader that has the model and the log could compute those templates; I have not built one.
- **Rank calls are numerical.** The smallest *kept* singular values are 1e-7 to 1e-4 of the largest. The fiber dimensions in E1 rest on a 1e-9 relative cutoff, with a 7-to-8-decade gap on either side.
- **The paper is an inspiration, not a theorem here.** The parallels are structural: symmetry-made versus counting-made hidden sectors, and the Section 8 tension between exact cancellation and the dynamics that need it. Nothing here proves anything about group rings.

## Prior art

- **Kalman realization theory (1963):** a transfer function determines its minimal state-space realization up to a change of basis. That is exactly the T found in E2.
- **Curtis, Ingerman and Morrow, "Circular planar graphs and resistor networks" (*Linear Algebra Appl.* 1998):** which resistor networks are recoverable from boundary measurements. Non-recoverable networks have invisible weight changes.
- **Dörfler and Bullo, "Kron reduction of graphs" (*IEEE TCS-I* 2013):** the Schur-complement view of what the boundary sees.
- **Rahmani, Ji, Mesbahi and Egerstedt (*SIAM J. Control Optim.* 2009):** symmetry about input nodes gives uncontrollability. That is the mirror-lock case.

What I believe is new here is narrow: treating the fibers as storage under *plasticity*. That covers the measured result that port-to-port learning and common hidden gain are exactly safe while the tested interior strain rules are not, and the gap between "the old reader fails" and "the information is gone".

## Review and follow-up (8 November)

Astra (GPT-6, in ChatGPT) audited commit `350046a`. It reproduced all six tests and the core numbers, added an impulse-response check in the time domain (public responses equal within 3e-13 over t ∈ [0, 40]), and made four corrections:

1. **E5's 3% is miscalibration, not erasure.** With recalibrated templates, all 64 words decode at 100% after 50 bulk writes (98% at 10% noise).
2. **"Only a frozen interior works" overreaches.** Uniform scaling changes the interior and stays safe. More generally, any rule with F(QLQᵀ) = QF(L)Qᵀ preserves the conjugacy.
3. **The paper connection is weak, and "torsion-free" is not demonstrated.** Rounded angles do not prove infinite order, and powers of T need not preserve the graph.
4. **State observability versus parameter identifiability.** All 14 modes reach the ports. What is hidden is the realization, not a dark mode.

All four are accepted, and the text above is corrected. The follow-up below tests the leads they opened. All numbers come from `python followup.py`; see [results/followup.json](results/followup.json). The noise scale is anchored to the original codebook's RMS, as in Astra's check.

### F1. Erased or miscalibrated? (64 words, 1000 public writes)

Writes reach every edge (SliderV3 additive). In the accuracy columns, "fixed" uses the original templates and "recal." uses templates recalibrated to the current weights.

| Writes | Public leak | Fixed, 1% noise | Recal., 1% noise | Recal., 10% noise | Min word separation |
|---|---|---|---|---|---|
| 0 | 0 | 100% | 100% | 100% | 0.075 |
| 10 | 0.088 | 80% | 100% | 100% | 0.054 |
| 50 | 0.17 | 1.6% | **100%** | 99.3% | 0.009 |
| 100 | 0.001 | 2.0% | 100% | 72% | 0.005 |
| 300 | 1e-4 | 2.2% | 96% | 4.6% | 0.004 |
| 1000 | ~0 | 2.2% | **18%** | 2.2% | 0.003 (26 edges per word at the clip) |

Restricted to boundary + port-interior edges, recalibrated decoding at 1% noise was still 99.9% after 1000 writes.

**Reading.** The tested rule leaks first and then contracts. The leak peaks around 50 writes and returns to near zero as both partners saturate. The information decays gradually. It is not erased at once; it contracts until measurement noise wins. Once weights reach the clip, distinct words can merge exactly, and that stage really is irreversible.

### F2. Safe plasticity

Scaling a weight class by 1.3, applied to two fiber partners:

| Scaled edges | Port difference |
|---|---|
| all | 2e-15 |
| hidden (port-interior + bulk) | **4e-15** |
| bulk only | 1.2e-2 |
| port-interior only | 4.6e-2 |

**A plastic system that keeps the lock.** For 500 steps, every public drive does two things:

1. A SliderV3 strain write on the port-to-port edges.
2. A homeostatic gain on every hidden edge, computed only from the public response energy and the public gain log.

The hidden gain wandered between 0.65 and 1.03, and the boundary weights grew to 7.5–8.4.

| Over 500 steps | Result |
|---|---|
| Largest public difference across the 64 words | 1e-11 |
| Key with fixed raw templates, 1% noise | 0% (the gain moves every reading) |
| Scale-invariant key (L_II from one frequency, divided by its trace): change in reading | **2e-16** |
| Scale-invariant key, 1% / 10% noise | **100%** / 54% |

So the lock does not need a frozen interior. It needs the interior to change only in ways that commute with every hidden basis change, and a common gain is the simplest such change. That is homeostatic synaptic scaling in neural terms (Turrigiano 1998), offered here as an analogy only.

### F3. Compact readers

- **One frequency is enough in principle.** A clamped key reading of every interior pair at a single frequency gives L_II exactly, by inverting it: error 5e-15. The 40-frequency key is therefore almost entirely redundant.
- **One interior node** (clamped drive-and-listen at a single node) at K frequencies, decoding the 64 words, with noise relative to its own RMS:

| Reader | Real numbers | Noise 0 | 1% | 3% | 10% |
|---|---|---|---|---|---|
| node 5, 1 frequency | 2 | 100% | 8% | 4% | 2% |
| node 5, 2 frequencies | 4 | 100% | 88% | 47% | 13% |
| node 5, 5 frequencies | 10 | 100% | 98% | 70% | 22% |
| node 5, 10 frequencies | 20 | 100% | **99.8%** | 88% | 35% |
| full key (100 pairs × 40) | 8,000 | 100% | 100% | 100% | 100% |

Other nodes behave similarly (nodes 4 and 6 are in the JSON). A 20-number reader suffices at 1% noise. The full key's robustness comes mostly from averaging.

### F4. Powers of T

| k | Largest weight on an absent edge | Largest positive off-diagonal | Valid weighting of the graph? |
|---|---|---|---|
| 1 | 2e-12 | 2e-12 | yes (it is the partner) |
| 2 | 0.036 | 0.008 | no |
| 3 | 0.11 | 0.024 | no |
| 5 | 0.35 | 0.078 | no |

The map that relates two stored words does not generate a family of stored words. The hidden set is a manifold of solutions, not an orbit.

### What this points to

The strongest continuation is the one both reviews converge on. Build a reader that tracks a stored word through drift without a fresh labeled codebook. In parallel, map which interior plasticity rules, beyond a common gain, keep the hidden basis change intact.

## Open direction (not built): integer-valued hidden data

The paper's exactness comes from arithmetic mod p (q ≡ 0 makes qI invisible). The closest wave-world analogue I can see is a *stroboscopic* reader, one that samples only at times nT. For such a reader, eigenvalue shifts by multiples of 2π/T are invisible. The hidden data would then be an integer winding per mode, an element of ℤⁿ, which really is a torsion-free group.

Anomalous Floquet systems (Rudner, Lindner, Berg and Levin, *PRX* 2013) are the physics precedent: topology that is invisible to the stroboscopic evolution operator and visible only in the motion between ticks. Whether a plastic network could keep such windings stable is unexplored.

## Reproduce

Python 3.10+ and NumPy.

```bash
pip install -r requirements.txt
python run_all.py --output results/results.json     # about 1.5 min
python followup.py --output results/followup.json   # about 6 min (OPENBLAS_NUM_THREADS=1 helps)
python -m unittest -v                                # 7 tests, about 2 s
```

Files:

- [fiberlock.py](fiberlock.py): graph, port map, analytic Jacobian, fiber walk, clamped key, strain write rule.
- [run_all.py](run_all.py): experiments E1–E6.
- [test_fiberlock.py](test_fiberlock.py): Jacobian check, fiber existence, invisibility, the basis change T, the Schur identity, the boundary-vs-bulk leak, and common-gain safety.
- [followup.py](followup.py): F1–F5, the follow-up to Astra's review.
- [results/results.json](results/results.json), [results/followup.json](results/followup.json): every number above.

MIT license.
