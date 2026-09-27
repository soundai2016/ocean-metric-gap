"""Verification of the normal-mode solver and array mismatch metric."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from acoustic_example import solve_modes, solve_modes_fd, pressure, reference_speed, depth_rms, perturbation, first_order_delta_k, array_mismatch

class AcousticTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.z = np.linspace(0.0, 200.0, 801)
        cls.c = reference_speed(cls.z)
        cls.modes = solve_modes(cls.z, cls.c, 100.0)
        cls.r = np.linspace(500.0, 30000.0, 61)
        cls.receivers = np.arange(20.0, 181.0, 5.0)
        cls.field = pressure(cls.modes, cls.r, 50.0, cls.receivers)

    def test_constant_speed_analytic_wavenumbers(self):
        modes = solve_modes(self.z, np.full_like(self.z, 1500.0), 100.0)
        n = np.arange(1, len(modes.k) + 1)
        exact = np.sqrt((2 * np.pi * 100 / 1500.0) ** 2 - (n * np.pi / 200.0) ** 2)
        np.testing.assert_allclose(modes.k, exact, rtol=1e-10, atol=1e-13)

    def test_constant_speed_analytic_modes(self):
        modes = solve_modes(self.z, np.full_like(self.z, 1500.0), 100.0)
        exact = np.sqrt(2 / 200.0) * np.sin(np.pi * self.z[:, None] * np.arange(1, len(modes.k) + 1) / 200.0)
        overlaps = np.trapezoid(modes.phi * exact, self.z, axis=0)
        np.testing.assert_allclose(overlaps ** 2, 1.0, rtol=1e-10, atol=1e-10)

    def test_normalization_and_pressure_release(self):
        np.testing.assert_allclose(np.trapezoid(self.modes.phi ** 2, self.z, axis=0), 1.0, atol=1e-08)
        np.testing.assert_array_equal(self.modes.phi[[0, -1]], 0.0)

    def test_zero_perturbation(self):
        np.testing.assert_allclose(array_mismatch(self.field, self.field), 0.0, atol=1e-14)

    def test_common_phase_and_amplitude_invariance(self):
        np.testing.assert_allclose(array_mismatch(self.field, self.field * (2.3 * np.exp(1.7j))), 0.0, atol=1e-14)

    def test_equal_depth_rms(self):
        localized = perturbation(self.z, 45.0, 15.0, 0.5)
        uniform = np.full_like(self.z, 0.5)
        self.assertAlmostEqual(depth_rms(localized, self.z), depth_rms(uniform, self.z), places=13)

    def test_central_difference_modal_derivative(self):
        dc = perturbation(self.z, 45.0, 15.0, 0.001)
        positive = solve_modes(self.z, self.c + dc, 100.0)
        negative = solve_modes(self.z, self.c - dc, 100.0)
        derivative = (positive.k - negative.k) / 2
        prediction = first_order_delta_k(self.modes, self.c, dc)
        np.testing.assert_allclose(derivative, prediction, rtol=2e-06, atol=1e-12)

    def test_galerkin_basis_refinement(self):
        dc = perturbation(self.z, 45.0, 15.0, 0.5)
        base192 = solve_modes(self.z, self.c, 100.0, 192)
        pert192 = solve_modes(self.z, self.c + dc, 100.0, 192)
        pert128 = solve_modes(self.z, self.c + dc, 100.0, 128)
        d128 = array_mismatch(self.field, pressure(pert128, self.r, 50.0, self.receivers))
        d192 = array_mismatch(pressure(base192, self.r, 50.0, self.receivers), pressure(pert192, self.r, 50.0, self.receivers))
        self.assertLess(float(np.max(abs(d128 - d192))), 1e-06)

    def test_depth_sampling_refinement(self):
        curves = []
        for dz in (0.5, 0.25, 0.125):
            z = np.linspace(0.0, 200.0, round(200 / dz) + 1)
            c = reference_speed(z)
            m = solve_modes(z, c, 100.0)
            p = solve_modes(z, c + perturbation(z, 45.0, 15.0, 0.5), 100.0)
            curves.append(array_mismatch(pressure(m, self.r, 50.0, self.receivers), pressure(p, self.r, 50.0, self.receivers)))
        self.assertLess(max((float(np.max(abs(c - curves[-1]))) for c in curves[:-1])), 1e-05)

    def test_independent_finite_difference_check(self):
        z = np.linspace(0.0, 200.0, 4001)
        c = reference_speed(z)
        spectral = solve_modes(z, c, 100.0)
        fd = solve_modes_fd(z, c, 100.0)
        self.assertEqual(len(spectral.k), len(fd.k))
        np.testing.assert_allclose(fd.k, spectral.k, rtol=0.0004, atol=1e-08)

    def test_reciprocity_at_equal_range(self):
        a = pressure(self.modes, np.array([10000.0]), 50.0, np.array([75.0]))
        b = pressure(self.modes, np.array([10000.0]), 75.0, np.array([50.0]))
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)

    def test_steering_snr_identity(self):
        pert = solve_modes(self.z, self.c + perturbation(self.z, 45.0, 15.0, 0.5), 100.0)
        q = pressure(pert, self.r, 50.0, self.receivers)
        snr_ratio = abs(np.sum(q.conj() * self.field, axis=1)) ** 2 / (np.sum(abs(q) ** 2, axis=1) * np.sum(abs(self.field) ** 2, axis=1))
        np.testing.assert_allclose(snr_ratio, 1 - array_mismatch(self.field, q), atol=1e-14)

    def test_metric_bounds(self):
        pert = solve_modes(self.z, self.c + 0.5, 100.0)
        d = array_mismatch(self.field, pressure(pert, self.r, 50.0, self.receivers))
        self.assertTrue(np.all((0 <= d) & (d <= 1)))

    def test_invalid_grids_and_speeds(self):
        for z, c in [(self.z[::-1], self.c), (self.z, np.zeros_like(self.z)), (self.z, np.full_like(self.z, np.nan))]:
            with self.subTest(case=str(c[:2])):
                with self.assertRaises(ValueError):
                    solve_modes(z, c, 100.0)

    def test_invalid_frequency_and_basis(self):
        for f, b in [(0.0, 128), (np.nan, 128), (100.0, 2), (100.0, 8)]:
            with self.subTest(f=f, b=b):
                with self.assertRaises(ValueError):
                    solve_modes(self.z, self.c, f, b)

    def test_no_propagating_mode(self):
        with self.assertRaises(ValueError):
            solve_modes(self.z, self.c, 1.0)

    def test_invalid_receiver_geometry(self):
        for r, zs, zr in [([0.0], 50.0, [25.0]), ([1.0], 0.0, [25.0]), ([1.0], 50.0, [200.0])]:
            with self.subTest(r=r, zs=zs, zr=zr):
                with self.assertRaises(ValueError):
                    pressure(self.modes, np.array(r), zs, np.array(zr))

    def test_invalid_metric_fields(self):
        with self.assertRaises(ValueError):
            array_mismatch(np.zeros((2, 3), complex), np.zeros((2, 3), complex))
        with self.assertRaises(ValueError):
            array_mismatch(np.ones((2, 3), complex), np.ones((3, 3), complex))
if __name__ == '__main__':
    unittest.main()
