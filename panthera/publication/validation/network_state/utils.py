"""
Shared utilities for network state validation tests.

Provides:
  - Physical constants (natural units)
  - NetworkNode: node with address (position), queue_depth (momentum), phase, frequency
  - NetworkSimulator: N-body MD simulator (velocity-Verlet) with ideal-gas and LJ modes
  - Statistical helpers: chi-squared, KS, bootstrap CI
"""

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Constants (natural units)
# ---------------------------------------------------------------------------
k_B = 1.0        # Boltzmann constant
MASS = 1.0        # particle mass
H_PLANCK = 1.0    # reduced Planck constant (for uncertainty relation tests)

# ---------------------------------------------------------------------------
# NetworkNode
# ---------------------------------------------------------------------------
class NetworkNode:
    """Lightweight descriptor for a single network node."""
    __slots__ = ("address", "queue_depth", "phase", "frequency")

    def __init__(self, address, queue_depth, phase=0.0, frequency=1.0):
        self.address = np.asarray(address, dtype=np.float64)
        self.queue_depth = np.asarray(queue_depth, dtype=np.float64)
        self.phase = float(phase)
        self.frequency = float(frequency)


# ---------------------------------------------------------------------------
# NetworkSimulator
# ---------------------------------------------------------------------------
class NetworkSimulator:
    """
    Molecular-dynamics simulator that models network nodes as classical
    particles in a d-dimensional box.

    Two interaction modes:
      - 'ideal'  : no inter-particle forces (ideal gas)
      - 'lj'     : Lennard-Jones 12-6 potential with periodic boundary
                    conditions and minimum-image convention

    Integrator: velocity-Verlet
    Thermostat : velocity-rescaling (optional, for equilibration)
    """

    def __init__(self, N, V, dim=3, mode="ideal",
                 epsilon=1.0, sigma=1.0, r_cut_factor=2.5,
                 dt=0.005, seed=42):
        self.N = N
        self.dim = dim
        self.mode = mode
        self.dt = dt
        self.epsilon = epsilon
        self.sigma = sigma
        self.mass = MASS

        # Box side length from volume
        self.L = V ** (1.0 / dim)
        self.V = V

        # LJ cutoff
        self.r_cut = r_cut_factor * sigma
        self.r_cut2 = self.r_cut ** 2
        # LJ shift so potential is continuous at cutoff
        sr6 = (sigma / self.r_cut) ** 6
        self.u_shift = 4.0 * epsilon * (sr6**2 - sr6)

        self.rng = np.random.default_rng(seed)

        # State arrays
        self.positions = None   # (N, dim)
        self.velocities = None  # (N, dim)
        self.forces = None      # (N, dim)

        # Accumulators for pressure (wall momentum transfer for ideal gas)
        self._wall_momentum = 0.0
        self._wall_time = 0.0

        # Accumulators for virial pressure (LJ)
        self._virial_sum = 0.0
        self._virial_steps = 0

        # Energy bookkeeping
        self.potential_energy = 0.0
        self.kinetic_energy = 0.0

        # Stress tensor accumulator (for transport coefficients)
        self._stress_xy_history = []
        self._energy_current_history = []

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def init_positions_uniform(self):
        """Place nodes uniformly at random inside the box [0, L)^dim."""
        self.positions = self.rng.uniform(0, self.L, size=(self.N, self.dim))

    def init_positions_lattice(self, spacing=None):
        """Place nodes on a regular lattice (1D chain or cubic lattice)."""
        if self.dim == 1:
            if spacing is None:
                spacing = self.L / self.N
            self.positions = np.linspace(spacing/2, self.L - spacing/2,
                                         self.N).reshape(-1, 1)
        else:
            n_side = int(np.ceil(self.N ** (1.0 / self.dim)))
            coords = np.linspace(0, self.L, n_side, endpoint=False) + self.L / (2*n_side)
            grids = np.meshgrid(*([coords] * self.dim), indexing='ij')
            pts = np.column_stack([g.ravel() for g in grids])
            self.positions = pts[:self.N].copy()

    def init_velocities_mb(self, T):
        """Sample velocities from Maxwell-Boltzmann at temperature T."""
        sigma_v = np.sqrt(k_B * T / self.mass)
        self.velocities = self.rng.normal(0, sigma_v, size=(self.N, self.dim))
        # Remove net momentum so centre of mass is stationary
        self.velocities -= self.velocities.mean(axis=0)

    def rescale_velocities(self, T_target):
        """Rescale velocities to match target temperature exactly."""
        T_cur = self.temperature()
        if T_cur > 1e-15:
            factor = np.sqrt(T_target / T_cur)
            self.velocities *= factor

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------
    def compute_forces(self):
        """Compute forces and potential energy; return virial contribution."""
        self.forces = np.zeros_like(self.positions)
        self.potential_energy = 0.0
        virial = 0.0

        if self.mode == "ideal":
            return 0.0

        # Lennard-Jones with minimum image convention
        pos = self.positions
        L = self.L
        sigma2 = self.sigma ** 2
        eps4 = 4.0 * self.epsilon

        for i in range(self.N - 1):
            dr = pos[i+1:] - pos[i]  # (N-i-1, dim)
            # Minimum image
            dr -= L * np.round(dr / L)
            r2 = np.sum(dr * dr, axis=1)  # (N-i-1,)

            mask = r2 < self.r_cut2
            if not np.any(mask):
                continue

            r2m = r2[mask]
            drm = dr[mask]

            inv_r2 = sigma2 / r2m
            inv_r6 = inv_r2 ** 3
            inv_r12 = inv_r6 ** 2

            # Potential
            self.potential_energy += np.sum(eps4 * (inv_r12 - inv_r6) - self.u_shift)

            # Force magnitude / r
            f_over_r = eps4 * (12.0 * inv_r12 - 6.0 * inv_r6) / r2m  # (n_pairs,)

            fvec = (f_over_r[:, None]) * drm  # (n_pairs, dim)
            self.forces[i] += fvec.sum(axis=0)
            idx = np.where(mask)[0] + i + 1
            np.subtract.at(self.forces, idx, fvec)

            # Virial: W = sum r_ij * f_ij
            virial += np.sum(f_over_r * r2m)

        return virial

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def step(self, thermostat_T=None):
        """
        One velocity-Verlet step.  If thermostat_T is given, rescale
        velocities after the step (simple velocity-rescaling thermostat).
        """
        dt = self.dt

        if self.forces is None:
            self.compute_forces()

        # Half-kick
        self.velocities += 0.5 * dt * self.forces / self.mass

        # Drift
        self.positions += dt * self.velocities

        # Boundary conditions
        if self.mode == "ideal":
            self._reflect_walls()
        else:
            # Periodic BC for LJ
            self.positions %= self.L

        # New forces
        virial = self.compute_forces()

        # Second half-kick
        self.velocities += 0.5 * dt * self.forces / self.mass

        # Kinetic energy
        self.kinetic_energy = 0.5 * self.mass * np.sum(self.velocities ** 2)

        # Virial pressure accumulator (LJ mode)
        if self.mode == "lj":
            self._virial_sum += virial
            self._virial_steps += 1

        # Stress tensor component (xy)
        if self.mode == "lj":
            sigma_xy = np.sum(self.velocities[:, 0] * self.velocities[:, 1]) * self.mass
            # Add force (configurational) contribution
            # Already included via virial but we need per-component
            # Approximate: use kinetic part only for Green-Kubo
            self._stress_xy_history.append(sigma_xy)

        # Thermostat
        if thermostat_T is not None and thermostat_T > 0:
            self.rescale_velocities(thermostat_T)
            self.kinetic_energy = 0.5 * self.mass * np.sum(self.velocities ** 2)

    def _reflect_walls(self):
        """Elastic reflection off box walls [0, L].  Accumulate wall momentum."""
        for d in range(self.dim):
            # Lower wall
            mask_lo = self.positions[:, d] < 0
            if np.any(mask_lo):
                self.positions[mask_lo, d] = -self.positions[mask_lo, d]
                self._wall_momentum += 2.0 * self.mass * np.sum(
                    np.abs(self.velocities[mask_lo, d]))
                self.velocities[mask_lo, d] = np.abs(self.velocities[mask_lo, d])

            # Upper wall
            mask_hi = self.positions[:, d] > self.L
            if np.any(mask_hi):
                self.positions[mask_hi, d] = 2.0 * self.L - self.positions[mask_hi, d]
                self._wall_momentum += 2.0 * self.mass * np.sum(
                    np.abs(self.velocities[mask_hi, d]))
                self.velocities[mask_hi, d] = -np.abs(self.velocities[mask_hi, d])

    def run(self, n_steps, thermostat_T=None, progress_every=0):
        """Run n_steps of integration."""
        for i in range(n_steps):
            self.step(thermostat_T=thermostat_T)
            if progress_every > 0 and (i + 1) % progress_every == 0:
                print(f"  step {i+1}/{n_steps}")

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    def temperature(self):
        """Kinetic temperature: T = 2 * KE / (dim * N * k_B)."""
        if self.velocities is None:
            return 0.0
        KE = 0.5 * self.mass * np.sum(self.velocities ** 2)
        dof = self.dim * self.N
        if dof == 0:
            return 0.0
        return 2.0 * KE / (dof * k_B)

    def pressure_ideal(self, elapsed_time):
        """
        Pressure from wall momentum transfer (ideal gas, reflecting walls).
        P = dp / (dt * A) summed over all walls.
        Total wall area for a d-dim box of side L: 2*d * L^(d-1).
        """
        if elapsed_time <= 0:
            return 0.0
        area = 2.0 * self.dim * self.L ** (self.dim - 1)
        P = self._wall_momentum / (elapsed_time * area)
        return P

    def pressure_virial(self):
        """
        Virial pressure for periodic-boundary LJ simulation.
        P = (N k_B T + W / dim) / V  where W is average virial per step.
        """
        if self._virial_steps == 0:
            return 0.0
        W_avg = self._virial_sum / self._virial_steps
        T = self.temperature()
        P = (self.N * k_B * T + W_avg / self.dim) / self.V
        return P

    def reset_pressure_accumulators(self):
        self._wall_momentum = 0.0
        self._wall_time = 0.0
        self._virial_sum = 0.0
        self._virial_steps = 0

    def order_parameter(self):
        """
        Orientational order parameter:
        Psi = |(1/N) sum exp(i * phi_j)|
        where phi_j is the angle of position vector from box centre.
        """
        centre = self.L / 2.0
        rel = self.positions - centre
        if self.dim >= 2:
            angles = np.arctan2(rel[:, 1], rel[:, 0])
        else:
            angles = np.sign(rel[:, 0]) * np.pi / 2
        psi = np.abs(np.mean(np.exp(1j * angles)))
        return psi

    def total_energy(self):
        return self.kinetic_energy + self.potential_energy

    def speeds(self):
        return np.sqrt(np.sum(self.velocities ** 2, axis=1))

    def pair_correlation(self, n_bins=100, r_max=None):
        """Compute pair correlation function g(r)."""
        if r_max is None:
            r_max = self.L / 2.0
        dr_bin = r_max / n_bins
        hist = np.zeros(n_bins)
        pos = self.positions
        L = self.L

        for i in range(self.N - 1):
            delta = pos[i+1:] - pos[i]
            if self.mode == "lj":
                delta -= L * np.round(delta / L)
            r = np.sqrt(np.sum(delta**2, axis=1))
            bins = (r / dr_bin).astype(int)
            valid = bins < n_bins
            np.add.at(hist, bins[valid], 1)

        # Normalise
        r_edges = np.arange(n_bins) * dr_bin
        r_mid = r_edges + dr_bin / 2
        rho = self.N / self.V
        if self.dim == 3:
            shell_vol = 4.0 * np.pi * r_mid**2 * dr_bin
        elif self.dim == 2:
            shell_vol = 2.0 * np.pi * r_mid * dr_bin
        else:
            shell_vol = 2.0 * dr_bin
        n_pairs = self.N * (self.N - 1) / 2.0
        ideal_count = n_pairs * shell_vol / self.V
        with np.errstate(divide='ignore', invalid='ignore'):
            g_r = np.where(ideal_count > 0, hist / ideal_count, 0.0)
        return r_mid, g_r

    def mean_squared_displacement(self, ref_positions):
        """MSD relative to reference positions (handles PBC unwrap approximately)."""
        delta = self.positions - ref_positions
        if self.mode == "lj":
            delta -= self.L * np.round(delta / self.L)
        return np.mean(np.sum(delta**2, axis=1))


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def chi_squared_test(observed, expected):
    """
    Chi-squared goodness-of-fit.  Returns (chi2, p_value).
    Bins with expected < 5 are merged from the tails inward.
    """
    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)

    # Merge bins with low expected counts
    while len(exp) > 1 and exp[0] < 5:
        obs = np.concatenate([[obs[0] + obs[1]], obs[2:]])
        exp = np.concatenate([[exp[0] + exp[1]], exp[2:]])
    while len(exp) > 1 and exp[-1] < 5:
        obs = np.concatenate([obs[:-2], [obs[-2] + obs[-1]]])
        exp = np.concatenate([exp[:-2], [exp[-2] + exp[-1]]])

    mask = exp > 0
    chi2 = np.sum((obs[mask] - exp[mask])**2 / exp[mask])
    dof = max(np.sum(mask) - 1, 1)
    p_value = float(sp_stats.chi2.sf(chi2, dof))
    return float(chi2), p_value


def kolmogorov_smirnov(samples, cdf_func):
    """
    KS test of samples against a theoretical CDF.
    Returns (ks_stat, p_value).
    """
    result = sp_stats.kstest(samples, cdf_func)
    return float(result.statistic), float(result.pvalue)


def bootstrap_confidence_interval(data, stat_func=np.mean,
                                   n_boot=1000, ci=0.95, seed=42):
    """
    Bootstrap confidence interval for a statistic.
    Returns (lower, upper, point_estimate).
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    point = float(stat_func(data))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot[i] = stat_func(sample)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot, 100 * alpha))
    hi = float(np.percentile(boot, 100 * (1 - alpha)))
    return lo, hi, point
