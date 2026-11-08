"""Follow-up to Astra's review (8 Nov 2026). Run: python followup.py --output results/followup.json (~2 min).

F1  Erased or miscalibrated? Track public leak, fixed-template decoding, recalibrated decoding and
    codeword separation through long bulk-reaching plasticity.
F2  Which plasticity is safe? Partial scalings, and a plastic system (public boundary learning +
    homeostatic scaling of the hidden network driven by public signals) that keeps the lock exact.
F3  Compact readers. One frequency of the clamped key gives L_II exactly; one interior node at a
    few frequencies decodes the 64 words.
F4  Do the relating maps T generate anything? Powers of T applied to a stored graph.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from fiberlock import GAMMA, Graph, public_history, rel, strain_write

HELD = np.linspace(0.03, 11.0, 97)
EVERYTHING = ("boundary", "port-interior", "bulk")


def setup(seed=0, bits=6, distance=0.8):
    g = Graph(bulk_density=0.8, seed=seed)
    m0 = np.random.default_rng(seed + 1).uniform(0.5, 1.5, g.E)
    rng = np.random.default_rng(seed + 200)
    V, _, _ = g.null_space(m0, g.hidden)
    dirs = V @ np.linalg.qr(rng.normal(size=(V.shape[1], bits)))[0]
    words = list(itertools.product([0, 1], repeat=bits))
    book = []
    for w in words:
        d = np.zeros(g.E)
        d[g.hidden] = dirs @ (2*np.array(w) - 1)
        book.append(g.fiber_walk(m0, d, distance, steps=25))
    return g, m0, np.array(book)


def accuracy(readings, templates, noise, scale, reads=40, seed=77):
    r = np.random.default_rng(seed)
    ok = 0
    for i, x in enumerate(readings):
        y = x + noise*scale*r.normal(size=(reads, x.size))
        d = np.linalg.norm(templates[None, :, :] - y[:, None, :], axis=2)
        ok += int(np.sum(np.argmin(d, axis=1) == i))
    return ok/(len(readings)*reads)


def min_separation(keys):
    n = len(keys)
    return float(min(rel(keys[i], keys[j]) for i in range(n) for j in range(i+1, n)))


def f1_erasure(g, book, steps=1000, checkpoints=(0, 10, 50, 100, 300, 1000)):
    out = {}
    hist = public_history(g, np.random.default_rng(300), steps)
    keys0 = np.array([g.clamped_key(m) for m in book])
    scale = np.sqrt(np.mean(keys0**2))
    for scope_name, scope in [("everything", EVERYTHING), ("boundary + port-interior", ("boundary", "port-interior"))]:
        cur = book.copy()
        rows = {}
        for k in range(steps + 1):
            if k in checkpoints:
                keys = np.array([g.clamped_key(m) for m in cur])
                ports = [g.port_map(m, HELD) for m in cur]
                rows[str(k)] = {
                    "public_leak_max": max(rel(p, ports[0]) for p in ports[1:]),
                    "fixed_templates_noise_0.01": accuracy(keys, keys0, 0.01, scale),
                    "recalibrated_noise_0.01": accuracy(keys, keys, 0.01, scale),
                    "recalibrated_noise_0.1": accuracy(keys, keys, 0.1, scale),
                    "min_pairwise_key_separation": min_separation(keys),
                    "edges_at_clip": int(np.sum((cur <= 0.0201) | (cur >= 9.999))),
                    "mean_weight": float(cur.mean()),
                }
            if k < steps:
                drive, f = hist[k]
                cur = np.array([strain_write(g, m, drive, f, scope) for m in cur])
        out[scope_name] = rows
    return out


def f2_safe_plasticity(g, m0, book):
    m1 = book[1]
    kinds = {"all edges": np.ones(g.E, bool),
             "hidden edges (port-interior + bulk)": g.kind != "boundary",
             "bulk only": g.kind == "bulk",
             "port-interior only": g.kind == "port-interior"}
    scalings = {}
    for name, mask in kinds.items():
        s = np.where(mask, 1.3, 1.0)
        scalings[name] = rel(g.port_map(book[0]*s, HELD), g.port_map(m1*s, HELD))

    # plastic system: SliderV3 strain learning on boundary edges + homeostatic gain on the hidden network
    hist = public_history(g, np.random.default_rng(400), 500)
    keys0 = np.array([g.clamped_key(m) for m in book])
    scale = np.sqrt(np.mean(keys0**2))
    hidden = g.kind != "boundary"
    cur = book.copy()
    running, log_gain = None, 0.0
    gains = []
    for drive, f in hist:
        cur = np.array([strain_write(g, m, drive, f, ("boundary",)) for m in cur])
        # public observable: response energy at the driven ports (identical for every word)
        b = np.zeros(g.N, complex)
        for node, amp in drive.items():
            b[node] = amp
        energy = float(np.linalg.norm((g.resolvent(cur[0], f) @ b)[g.ports])**2/np.linalg.norm(b)**2)
        running = energy if running is None else 0.9*running + 0.1*energy
        # gain rises when the response is above its running mean, with a pull that keeps it bounded;
        # every input to this line is public, so all 64 words receive the same gain
        step = 0.04*np.tanh((energy - running)/running) - 0.05*log_gain
        log_gain += step
        cur[:, hidden] *= np.exp(step)
        gains.append(float(np.exp(log_gain)))
    keys = np.array([g.clamped_key(m) for m in cur])
    ports = [g.port_map(m, HELD) for m in cur]
    lii0 = np.array([normalized_lii(g, m) for m in book])
    lii = np.array([normalized_lii(g, m) for m in cur])
    sys_ = {"steps": 500,
            "final_hidden_gain": gains[-1],
            "hidden_gain_range_over_history": [float(min(gains)), float(max(gains))],
            "boundary_weight_range": [float(cur[:, ~hidden].min()), float(cur[:, ~hidden].max())],
            "public_leak_max": max(rel(p, ports[0]) for p in ports[1:]),
            "fixed_raw_templates_noise_0.01": accuracy(keys, keys0, 0.01, scale),
            "scale_invariant_reader_noise_0.01": accuracy(lii, lii0, 0.01, np.sqrt(np.mean(lii0**2))),
            "scale_invariant_reader_noise_0.1": accuracy(lii, lii0, 0.1, np.sqrt(np.mean(lii0**2))),
            "scale_invariant_reading_change": float(np.abs(lii - lii0).max())}
    return {"scale_by_1.3_port_difference": scalings, "plastic_system": sys_}


def lii_from_one_frequency(g, m, w=1.0):
    """Clamped key at ONE frequency, all interior pairs: invert to get L_II exactly."""
    L = g.laplacian(m)
    Lii = L[np.ix_(g.interior, g.interior)]
    G = np.linalg.inv(GAMMA*np.eye(g.I) + 1j*(Lii - w*np.eye(g.I)))     # what the key measures
    return w*np.eye(g.I) - 1j*(np.linalg.inv(G) - GAMMA*np.eye(g.I)), Lii


def normalized_lii(g, m):
    Lr, _ = lii_from_one_frequency(g, m)
    Lr = Lr.real
    return (Lr/np.trace(Lr)).ravel()


def f3_compact(g, book):
    Lr, Lii = lii_from_one_frequency(g, book[0])
    out = {"one_frequency_L_II_recovery_error": float(np.abs(Lr - Lii).max())}
    rows = {}
    for node in g.interior[:3]:
        for K in [1, 2, 3, 5, 10]:
            freqs = np.linspace(0.5, 8.0, K)
            keys = np.array([g.clamped_key(m, freqs, nodes=[int(node)]) for m in book])
            sc = np.sqrt(np.mean(keys**2))
            rows[f"node {int(node)}, {K} freq ({2*K} real numbers)"] = {
                "noise_0": accuracy(keys, keys, 0.0, sc, reads=1),
                "noise_0.01": accuracy(keys, keys, 0.01, sc),
                "noise_0.03": accuracy(keys, keys, 0.03, sc),
                "noise_0.1": accuracy(keys, keys, 0.1, sc),
                "min_separation": min_separation(keys)}
    out["single_node_reader"] = rows
    return out


def f4_powers(g, m0, m1):
    L0, L1 = g.laplacian(m0), g.laplacian(m1)
    I, P = g.interior, g.ports
    A0, A1, B0, B1 = L0[np.ix_(I, I)], L1[np.ix_(I, I)], L0[np.ix_(I, P)], L1[np.ix_(I, P)]
    w0, V0 = np.linalg.eigh(A0)
    w1, V1 = np.linalg.eigh(A1)
    S = np.sign(np.sum((V1.T @ B1)*(V0.T @ B0), axis=1))
    T = V1 @ np.diag(S) @ V0.T
    absent = np.ones((g.N, g.N), bool)
    for a, b in g.edges:
        absent[a, b] = absent[b, a] = False
    np.fill_diagonal(absent, False)
    rows = {}
    Tk = np.eye(g.I)
    for k in range(1, 6):
        Tk = T @ Tk
        Sk = np.eye(g.N)
        Sk[np.ix_(I, I)] = Tk
        Lk = Sk @ L0 @ Sk.T
        off = Lk - np.diag(np.diag(Lk))
        rows[str(k)] = {"largest_entry_on_absent_edges": float(np.abs(Lk[absent]).max()),
                        "largest_positive_offdiagonal": float(off.max()),
                        "is_a_valid_weighting_of_the_graph": bool(np.abs(Lk[absent]).max() < 1e-9 and off.max() < 1e-9)}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    g, m0, book = setup()
    res = {"codebook": {"words": len(book), "min_weight": float(book.min())}}
    res["F1_erasure_vs_miscalibration"] = f1_erasure(g, book)
    res["F2_safe_plasticity"] = f2_safe_plasticity(g, m0, book)
    res["F3_compact_readers"] = f3_compact(g, book)
    res["F4_powers_of_T"] = f4_powers(g, book[0], book[1])
    eig, V = np.linalg.eigh(g.laplacian(m0))
    res["F5_state_observability"] = {"modes": g.N, "smallest_port_norm_of_an_eigenvector":
                                     float(np.linalg.norm(V[:g.P], axis=0).min()),
                                     "smallest_eigenvalue_gap": float(np.diff(eig).min())}
    text = json.dumps(res, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
