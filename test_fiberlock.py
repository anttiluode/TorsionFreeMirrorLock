"""Regression checks for the claims in the README (about 15 s)."""
import unittest

import numpy as np

from fiberlock import Graph, public_history, rel, strain_write

HELD = np.linspace(0.03, 11.0, 61)


class FiberLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = Graph(bulk_density=0.8, seed=0)
        cls.m0 = np.random.default_rng(1).uniform(0.5, 1.5, cls.g.E)
        d = np.zeros(cls.g.E)
        d[cls.g.hidden] = np.random.default_rng(2).normal(size=len(cls.g.hidden))
        cls.m1 = cls.g.fiber_walk(cls.m0, d, distance=0.8, steps=25)

    def test_analytic_jacobian_matches_finite_differences(self):
        g, m = self.g, self.m0
        J = g.port_jacobian(m)
        for k in [0, 10, 40]:
            e = np.zeros(g.E)
            e[k] = 1e-6
            fd = (g.port_map(m + e) - g.port_map(m - e))/2e-6
            np.testing.assert_allclose(J[:, k], fd, atol=1e-6)

    def test_graph_has_a_fiber_and_no_symmetry_is_needed(self):
        g = self.g
        V, s, r = g.null_space(self.m0, g.hidden)
        self.assertEqual(V.shape[1], 9)
        self.assertGreater(np.abs(np.sort(self.m0) - np.sort(self.m1)).max(), 0.01)   # not a relabelling

    def test_fiber_partners_are_invisible_at_ports_but_visible_to_key(self):
        g = self.g
        self.assertLess(rel(g.port_map(self.m0, HELD), g.port_map(self.m1, HELD)), 1e-9)
        self.assertGreater(rel(g.clamped_key(self.m0), g.clamped_key(self.m1)), 0.05)

    def test_partners_are_a_hidden_orthogonal_change_of_interior_basis(self):
        g = self.g
        L0, L1 = g.laplacian(self.m0), g.laplacian(self.m1)
        I, P = g.interior, g.ports
        A0, A1, B0, B1 = L0[np.ix_(I, I)], L1[np.ix_(I, I)], L0[np.ix_(I, P)], L1[np.ix_(I, P)]
        w0, V0 = np.linalg.eigh(A0)
        w1, V1 = np.linalg.eigh(A1)
        np.testing.assert_allclose(w0, w1, atol=1e-9)
        S = np.sign(np.sum((V1.T @ B1)*(V0.T @ B0), axis=1))
        T = V1 @ np.diag(S) @ V0.T
        np.testing.assert_allclose(T @ A0 @ T.T, A1, atol=1e-9)
        np.testing.assert_allclose(T @ B0, B1, atol=1e-9)

    def test_boundary_edit_acts_through_the_port_block_only(self):
        """Schur complement: G'_PP = (G_PP^-1 + i Delta_PP)^-1 for an edit on port-port edges."""
        g, m = self.g, self.m0
        d = np.zeros(g.E)
        d[:6] = [0.3, -0.1, 0.2, 0.05, 0.4, -0.2]
        for w in [0.7, 3.1]:
            Gpp = g.resolvent(m, w)[np.ix_(g.ports, g.ports)]
            Dpp = g.laplacian(d)[np.ix_(g.ports, g.ports)]
            predicted = np.linalg.inv(np.linalg.inv(Gpp) + 1j*Dpp)
            actual = g.resolvent(m + d, w)[np.ix_(g.ports, g.ports)]
            np.testing.assert_allclose(actual, predicted, atol=1e-11)

    def test_boundary_plasticity_never_leaks_and_bulk_plasticity_does(self):
        g = self.g
        hist = public_history(g, np.random.default_rng(5), 40)
        a, b = self.m0.copy(), self.m1.copy()
        c, d = self.m0.copy(), self.m1.copy()
        for drive, f in hist:
            a, b = (strain_write(g, x, drive, f, ("boundary",)) for x in (a, b))
            c, d = (strain_write(g, x, drive, f, ("boundary", "port-interior", "bulk")) for x in (c, d))
        self.assertLess(rel(g.port_map(a, HELD), g.port_map(b, HELD)), 1e-9)
        self.assertGreater(rel(g.port_map(c, HELD), g.port_map(d, HELD)), 1e-3)


if __name__ == "__main__":
    unittest.main()
