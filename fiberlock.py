"""Core of the torsion-free mirror lock: a weighted graph, its port map, and the fibers of that map.

A graph has P public port nodes and I interior nodes. Edge weights are positive. The medium is
dx/dt = -i L x - gamma x + drive, read in steady state, so a drive b at frequency w gives
A = G(w) b with G(w) = (gamma + i (L - w))^-1.

The public may drive and read only port nodes, so everything it can ever learn is the port block
G_PP(w). The *port map* sends a weight vector m to G_PP(w) at every w. Its fibers are sets of
weight vectors that no port experiment, at any frequency, can tell apart. Friday's mirror lock is
the special case where two fiber points are related by a graph symmetry. Here the graph has no
symmetry at all.
"""
import numpy as np

GAMMA = 0.2
SWEEP = np.linspace(0.05, 9.0, 40)              # frequencies used to define and walk the fiber


class Graph:
    """Fixed topology. Ports are nodes 0..P-1; interior nodes are P..N-1."""

    def __init__(self, P=4, I=10, port_degree=5, seed=0, bulk_density=1.0):
        rng = np.random.default_rng(seed)
        self.P, self.I, self.N = P, I, P + I
        boundary = [(a, b) for a in range(P) for b in range(a + 1, P)]
        port_interior = [(a, P + int(j)) for a in range(P)
                         for j in sorted(rng.choice(I, port_degree, replace=False))]
        bulk = [(P + a, P + b) for a in range(I) for b in range(a + 1, I)]
        if bulk_density < 1.0:
            keep = rng.random(len(bulk)) < bulk_density
            bulk = [e for e, k in zip(bulk, keep) if k]
        self.edges = boundary + port_interior + bulk
        self.kind = np.array(["boundary"]*len(boundary) + ["port-interior"]*len(port_interior)
                             + ["bulk"]*len(bulk))
        self.E = len(self.edges)
        self.U = np.zeros((self.N, self.E))           # signed incidence, L = U diag(m) U^T
        for k, (a, b) in enumerate(self.edges):
            self.U[a, k], self.U[b, k] = 1.0, -1.0
        self.ports = np.arange(P)
        self.interior = np.arange(P, self.N)
        self.hidden = np.where(self.kind != "boundary")[0]   # weights the public cannot touch directly

    def laplacian(self, m):
        return (self.U * m) @ self.U.T

    def resolvent(self, m, w, L=None):
        L = self.laplacian(m) if L is None else L
        return np.linalg.inv(GAMMA*np.eye(self.N) + 1j*(L - w*np.eye(self.N)))

    def port_map(self, m, freqs=SWEEP):
        """Stacked real/imag of the upper triangle of G_PP(w) for every w."""
        L = self.laplacian(m)
        iu = np.triu_indices(self.P)
        out = [self.resolvent(m, w, L)[np.ix_(self.ports, self.ports)][iu] for w in freqs]
        out = np.concatenate(out)
        return np.concatenate([out.real, out.imag])

    def port_jacobian(self, m, cols=None, freqs=SWEEP):
        """d(port_map)/dm, analytic: dG = -G (i dL) G, dL/dm_e = u_e u_e^T."""
        cols = np.arange(self.E) if cols is None else cols
        L = self.laplacian(m)
        iu = np.triu_indices(self.P)
        blocks = []
        for w in freqs:
            G = self.resolvent(m, w, L)
            GU = G @ self.U[:, cols]                    # N x len(cols)
            GUp = GU[self.ports]                        # P x len(cols)
            # dG_PP[a,b] / dm_e = -i (G u_e)_a (G u_e)_b   (G symmetric)
            d = -1j*GUp[iu[0]]*GUp[iu[1]]
            blocks.append(d)
        J = np.concatenate(blocks)
        return np.concatenate([J.real, J.imag])

    def null_space(self, m, cols, tol=1e-9):
        J = self.port_jacobian(m, cols)
        _, s, Vt = np.linalg.svd(J, full_matrices=True)
        rank = int(np.sum(s > tol*s[0]))
        return Vt[rank:].T, s, rank

    def project_to_fiber(self, m, target, cols, iters=30, tol=1e-13):
        """Gauss-Newton on the hidden weights until the port map equals target."""
        m = m.copy()
        for _ in range(iters):
            r = self.port_map(m) - target
            if np.abs(r).max() < tol:
                break
            J = self.port_jacobian(m, cols)
            m[cols] -= np.linalg.lstsq(J, r, rcond=None)[0]
        return m, float(np.abs(self.port_map(m) - target).max())

    def fiber_walk(self, m0, direction, distance, steps=40):
        """Move along the fiber of m0 through the hidden weights, roughly along `direction`.

        Each step projects the requested direction onto the current null space, takes a small
        step, then corrects back onto the fiber. Non-local by construction: it uses the model.
        """
        cols = self.hidden
        target = self.port_map(m0)
        m = m0.copy()
        h = distance/steps
        for _ in range(steps):
            V, _, _ = self.null_space(m, cols)
            v = V @ (V.T @ direction[cols])
            n = np.linalg.norm(v)
            if n < 1e-12:
                break
            m[cols] += h*v/n
            m, _ = self.project_to_fiber(m, target, cols)
        return m

    def clamped_key(self, m, freqs=SWEEP, nodes=None):
        """Key reader: hold every port at zero, drive and listen at interior nodes.

        With the ports clamped the interior evolves under L_II alone, so this reading never
        depends on the boundary (port-port) weights.
        """
        nodes = self.interior if nodes is None else nodes
        L = self.laplacian(m)
        Lii = L[np.ix_(self.interior, self.interior)]
        idx = [int(np.where(self.interior == n)[0][0]) for n in nodes]
        out = []
        for w in freqs:
            Gi = np.linalg.inv(GAMMA*np.eye(self.I) + 1j*(Lii - w*np.eye(self.I)))
            out.append(Gi[np.ix_(idx, idx)].ravel())
        out = np.concatenate(out)
        return np.concatenate([out.real, out.imag])


def rel(x, y):
    return float(np.linalg.norm(x - y)/np.sqrt((np.linalg.norm(x)**2 + np.linalg.norm(y)**2)/2))


def strain_write(g, m, drive, freq, scope, rate=0.06, mode="additive"):
    """SliderV3 Fourier/strain rule restricted to a set of edge kinds.

    scope: iterable of edge kinds the rule may change ("boundary", "port-interior", "bulk").
    The rule reads the strain |A_a - A_b| on each edge it writes, so a rule that writes an edge
    with an interior endpoint reads an interior amplitude.
    """
    b = np.zeros(g.N, complex)
    for node, amp in drive.items():
        b[node] = amp
    A = g.resolvent(m, freq) @ b
    A = A/max(np.abs(A[g.ports]).max(), 1e-12)        # normalized by what the public can see
    m = m.copy()
    allowed = np.isin(g.kind, list(scope))
    for k in np.where(allowed)[0]:
        a, c = g.edges[k]
        E = 0.48*np.tanh(abs(A[a] - A[c])**2/0.25)
        m[k] = m[k] + rate*E if mode == "additive" else m[k]*(1 + rate*E)
    return np.clip(m, 0.02, 10.0)


def public_history(g, rng, steps):
    hist = []
    for _ in range(steps):
        k = rng.integers(1, g.P + 1)
        ports = rng.choice(g.P, size=k, replace=False)
        hist.append(({int(p): complex(rng.normal(), rng.normal()) for p in ports},
                     float(rng.uniform(0.1, 8.5))))
    return hist
