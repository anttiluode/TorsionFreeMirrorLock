"""Every measurement in the README. Run: python run_all.py --output results/results.json (~2-3 min, CPU)."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from fiberlock import Graph, public_history, rel, strain_write

HELD = np.linspace(0.03, 11.0, 97)        # held-out frequencies, never used to build a fiber
BOUNDARY = ("boundary",)
PORT_EDGES = ("boundary", "port-interior")
EVERYTHING = ("boundary", "port-interior", "bulk")


def e1_counting(seed):
    """Fiber dimension = hidden weights - rank of the port map. No symmetry anywhere."""
    rows = []
    for dens in [1.0, 0.9, 0.8, 0.7, 0.6]:
        g = Graph(bulk_density=dens, seed=seed)
        m = np.random.default_rng(seed + 1).uniform(0.5, 1.5, g.E)
        V, s, r = g.null_space(m, g.hidden)
        s = s/s[0]
        rows.append({"bulk_density": dens, "edges": g.E, "hidden_weights": len(g.hidden),
                     "port_rank": r, "fiber_dimension": V.shape[1],
                     "smallest_kept_singular_value": float(s[r-1]),
                     "largest_dropped_singular_value": float(s[r]) if r < len(s) else None})
    return rows


def e2_pair(g, m0, rng):
    """One walked pair: invisible at the ports, visible to the key, related by a hidden basis change."""
    d = np.zeros(g.E)
    d[g.hidden] = rng.normal(size=len(g.hidden))
    m1 = g.fiber_walk(m0, d, distance=1.0)
    L0, L1 = g.laplacian(m0), g.laplacian(m1)
    I, P = g.interior, g.ports
    A0, A1 = L0[np.ix_(I, I)], L1[np.ix_(I, I)]
    B0, B1 = L0[np.ix_(I, P)], L1[np.ix_(I, P)]
    w0, V0 = np.linalg.eigh(A0)
    w1, V1 = np.linalg.eigh(A1)
    S = np.sign(np.sum((V1.T @ B1)*(V0.T @ B0), axis=1))
    T = V1 @ np.diag(S) @ V0.T
    ang = np.sort(np.abs(np.angle(np.linalg.eigvals(T))))/np.pi
    return m1, {
        "largest_weight_change": float(np.abs(m1 - m0).max()),
        "smallest_weight": float(m1.min()),
        "port_difference_held_out": rel(g.port_map(m0, HELD), g.port_map(m1, HELD)),
        "clamped_key_difference": rel(g.clamped_key(m0), g.clamped_key(m1)),
        "weight_multiset_difference": float(np.abs(np.sort(m0) - np.sort(m1)).max()),
        "full_laplacian_isospectral": float(np.abs(np.linalg.eigvalsh(L0) - np.linalg.eigvalsh(L1)).max()),
        "basis_change_T": {"orthogonality_residual": float(np.abs(T @ T.T - np.eye(g.I)).max()),
                           "T_L_II_T_residual": float(np.abs(T @ A0 @ T.T - A1).max()),
                           "T_L_IP_residual": float(np.abs(T @ B0 - B1).max()),
                           "largest_entry": float(np.abs(T).max()),
                           "entries_above_0.1": int(np.sum(np.abs(T) > 0.1)),
                           "eigen_angles_over_pi": [round(float(a), 4) for a in ang]},
    }


def e3_shared_edits(g, m0, m1, rng):
    """Which edits, applied identically to two fiber partners, keep them indistinguishable?"""
    single = {}
    for kind in ["boundary", "port-interior", "bulk"]:
        vals = []
        for k in np.where(g.kind == kind)[0][:5]:
            d = np.zeros(g.E)
            d[k] = 0.05
            vals.append(rel(g.port_map(m0 + d, HELD), g.port_map(m1 + d, HELD)))
        single[kind] = [float(v) for v in vals]
    J0 = g.port_jacobian(m0)
    Ds, dims = [], []
    for _ in range(4):
        d = np.zeros(g.E)
        d[g.hidden] = rng.normal(size=len(g.hidden))
        Ds.append(J0 - g.port_jacobian(g.fiber_walk(m0, d, 0.6)))
        s = np.linalg.svd(np.vstack(Ds), compute_uv=False)
        dims.append(int(np.sum(s/s[0] <= 1e-9) + max(0, g.E - len(s))))
    Vt = np.linalg.svd(np.vstack(Ds))[2]
    s = np.linalg.svd(np.vstack(Ds), compute_uv=False)
    ns = Vt[int(np.sum(s/s[0] > 1e-9)):]
    return {"single_edge_edit_plus_0.05": single,
            "global_scale_x1.05": rel(g.port_map(1.05*m0, HELD), g.port_map(1.05*m1, HELD)),
            "edits_preserving_all_partners_dim_vs_partners": dims,
            "boundary_share_of_that_space": float(np.linalg.norm(ns[:, g.kind == "boundary"])**2/max(len(ns), 1)),
            "boundary_edges": int(np.sum(g.kind == "boundary"))}


def e4_leak(g, m0, m1, rng, steps=300):
    hist = public_history(g, rng, steps)
    out = {}
    for name, scope in [("boundary only", BOUNDARY), ("boundary + port-interior", PORT_EDGES),
                        ("everything", EVERYTHING)]:
        for mode in ["additive", "multiplicative"]:
            a, b = m0.copy(), m1.copy()
            row = {}
            for k, (drive, f) in enumerate(hist, start=1):
                a = strain_write(g, a, drive, f, scope, mode=mode)
                b = strain_write(g, b, drive, f, scope, mode=mode)
                if k in (1, 10, 50, steps):
                    row[str(k)] = {"port_difference": rel(g.port_map(a, HELD), g.port_map(b, HELD)),
                                   "key_difference": rel(g.clamped_key(a), g.clamped_key(b))}
            out[f"{name} / {mode}"] = row
    return out


def e5_words(g, m0, rng, bits=6, distance=0.8, noise_levels=(0.0, 0.1, 0.3, 1.0, 3.0), reads=40):
    V, _, _ = g.null_space(m0, g.hidden)
    Q = np.linalg.qr(rng.normal(size=(V.shape[1], bits)))[0]
    dirs = V @ Q                                             # bits orthonormal fiber directions
    words = list(itertools.product([0, 1], repeat=bits))
    configs = []
    for w in words:
        d = np.zeros(g.E)
        d[g.hidden] = dirs @ (2*np.array(w) - 1)
        configs.append(g.fiber_walk(m0, d, distance, steps=25))
    ports = [g.port_map(m, HELD) for m in configs]
    keys = np.array([g.clamped_key(m) for m in configs])
    scale = np.sqrt(np.mean(keys**2))
    pairwise = [rel(keys[i], keys[j]) for i in range(len(keys)) for j in range(i+1, len(keys))]

    def decode_accuracy(readings, templates, noise):
        ok = tot = 0
        r = np.random.default_rng(7)
        for i, x in enumerate(readings):
            for _ in range(reads):
                y = x + noise*scale*r.normal(size=x.shape)
                ok += int(np.argmin(np.linalg.norm(templates - y, axis=1)) == i)
                tot += 1
        return ok/tot

    res = {"bits": bits, "words": len(words), "walk_distance": distance,
           "max_port_difference_across_words": max(rel(p, ports[0]) for p in ports[1:]),
           "min_pairwise_key_difference": float(min(pairwise)),
           "key_decode_accuracy_vs_noise": {str(n): decode_accuracy(keys, keys, n) for n in noise_levels}}

    # public traffic on every stored word
    hist = public_history(g, rng, 300)
    for name, scope, steps in [("after 300 boundary-only public writes", BOUNDARY, 300),
                               ("after 50 public writes reaching the bulk", EVERYTHING, 50)]:
        cur = [m.copy() for m in configs]
        for drive, f in hist[:steps]:
            cur = [strain_write(g, m, drive, f, scope) for m in cur]
        p = [g.port_map(m, HELD) for m in cur]
        k = np.array([g.clamped_key(m) for m in cur])
        res[name] = {"max_port_difference_across_words": max(rel(x, p[0]) for x in p[1:]),
                     "key_decode_with_original_templates_noise_0.01": decode_accuracy(k, keys, 0.01)}
    return res


def e6_local_key_write(g, m0, rng):
    """A local write driven at an interior node moves off the fiber: the public sees that something changed."""
    out = {}
    for node in g.interior[:3]:
        m = strain_write(g, m0, {int(node): 1.0}, 1.3, ("port-interior", "bulk"))
        out[int(node)] = rel(g.port_map(m0, HELD), g.port_map(m, HELD))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed + 100)
    g = Graph(bulk_density=0.8, seed=args.seed)
    m0 = np.random.default_rng(args.seed + 1).uniform(0.5, 1.5, g.E)
    res = {"graph": {"ports": g.P, "interior": g.I, "edges": g.E,
                     "edge_kinds": {k: int(np.sum(g.kind == k)) for k in ["boundary", "port-interior", "bulk"]},
                     "bulk_density": 0.8, "gamma": 0.2, "seed": args.seed}}
    res["E1_counting"] = e1_counting(args.seed)
    m1, res["E2_pair"] = e2_pair(g, m0, rng)
    res["E3_shared_edits"] = e3_shared_edits(g, m0, m1, rng)
    res["E4_public_plasticity"] = e4_leak(g, m0, m1, rng)
    res["E5_words"] = e5_words(g, m0, rng)
    res["E6_local_key_write_port_change"] = e6_local_key_write(g, m0, rng)
    text = json.dumps(res, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
