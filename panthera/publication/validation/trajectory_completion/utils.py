"""
Shared utilities for trajectory completion (Paper 2) validation tests.

Provides:
  - S-entropy coordinate extraction (S_k, S_t, S_e)
  - Shannon entropy, autocorrelation
  - KDTree wrapper for S-space navigation
  - Program library (48+ programs across 7 categories)
  - Triple convergence checker
  - NetworkNode for security simulations
"""

import numpy as np
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
k_B = 1.0
EPSILON = 1e-15  # Guard against log(0), division by zero

# Eight-scale hierarchy frequencies (Hz equivalents in natural units)
SCALE_HIERARCHY = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------
def shannon_entropy(data):
    """Compute Shannon entropy: -sum p_i log p_i. Works on discrete bins."""
    data = np.asarray(data, dtype=np.float64).ravel()
    if len(data) == 0:
        return 0.0
    # Bin the data
    if np.all(data == data[0]):
        return 0.0
    # For integer-like data, use value counts; otherwise histogram
    if np.allclose(data, np.round(data)):
        vals, counts = np.unique(np.round(data).astype(int), return_counts=True)
    else:
        n_bins = max(int(np.sqrt(len(data))), 2)
        counts, _ = np.histogram(data, bins=n_bins)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + EPSILON)))


def autocorrelation_lag1(sequence):
    """Compute lag-1 autocorrelation of a sequence."""
    seq = np.asarray(sequence, dtype=np.float64).ravel()
    if len(seq) < 2:
        return 0.0
    mean = np.mean(seq)
    var = np.var(seq)
    if var < EPSILON:
        return 0.0
    n = len(seq)
    cov = np.sum((seq[:-1] - mean) * (seq[1:] - mean)) / n
    return float(cov / (var + EPSILON))


# ---------------------------------------------------------------------------
# S-entropy coordinates
# ---------------------------------------------------------------------------
def S_k(inp, out):
    """Knowledge entropy: H(output) / H(input).
    Measures how information transforms through computation."""
    h_in = shannon_entropy(inp)
    h_out = shannon_entropy(out)
    if h_in < EPSILON:
        return h_out / EPSILON if h_out > EPSILON else 1.0
    return float(h_out / (h_in + EPSILON))


def S_t(output_sequence):
    """Temporal entropy: 1 - autocorrelation.
    Measures temporal unpredictability of output."""
    ac = autocorrelation_lag1(output_sequence)
    return float(1.0 - ac)


def S_e(inp, out):
    """Evolution entropy: sigma(delta) / mu(|delta|).
    Measures variability of the input-output mapping."""
    inp = np.asarray(inp, dtype=np.float64).ravel()
    out = np.asarray(out, dtype=np.float64).ravel()
    min_len = min(len(inp), len(out))
    if min_len == 0:
        return 0.0
    delta = out[:min_len] - inp[:min_len]
    mu_abs = np.mean(np.abs(delta))
    if mu_abs < EPSILON:
        return 0.0
    sigma = np.std(delta)
    return float(sigma / (mu_abs + EPSILON))


def extract_s_coordinates(inp, out, output_sequence=None):
    """Extract full S-coordinate triple (S_k, S_t, S_e) from input/output."""
    if output_sequence is None:
        output_sequence = out
    sk = S_k(inp, out)
    st = S_t(output_sequence)
    se = S_e(inp, out)
    return np.array([sk, st, se], dtype=np.float64)


# ---------------------------------------------------------------------------
# KDTree wrapper for S-space navigation
# ---------------------------------------------------------------------------
class SSpaceNavigator:
    """KD-tree based navigator in S-entropy coordinate space."""

    def __init__(self, programs, coordinates):
        """
        programs: list of program names
        coordinates: (N, 3) array of S-coordinates
        """
        self.programs = list(programs)
        self.coordinates = np.asarray(coordinates, dtype=np.float64)
        self.tree = KDTree(self.coordinates)
        self._query_count = 0

    def navigate(self, target_coords, k=1):
        """Find k nearest programs to target S-coordinates.
        Returns (distances, indices, program_names)."""
        target = np.asarray(target_coords, dtype=np.float64)
        dist, idx = self.tree.query(target, k=k)
        if k == 1:
            return float(dist), int(idx), self.programs[int(idx)]
        else:
            names = [self.programs[int(i)] for i in idx]
            return dist, idx, names

    def estimate_comparisons(self, target_coords):
        """Estimate nodes visited during query (approx log2(N) for KD-tree)."""
        n = len(self.programs)
        # KD-tree average comparisons ~ O(log N)
        return max(1, int(np.log2(n + 1) + 0.5))

    @property
    def size(self):
        return len(self.programs)


# ---------------------------------------------------------------------------
# Program library
# ---------------------------------------------------------------------------
def _safe_div(a, b):
    if b == 0:
        return 0.0
    return a / b


def _safe_mod(a, b):
    if b == 0:
        return 0
    return a % b


# Aggregation (10)
def prog_sum(x): return [sum(x)]
def prog_product(x):
    r = 1
    for v in x: r *= v
    return [r]
def prog_max(x): return [max(x)] if x else [0]
def prog_min(x): return [min(x)] if x else [0]
def prog_mean(x): return [sum(x) / max(len(x), 1)]
def prog_length(x): return [len(x)]
def prog_range(x): return [max(x) - min(x)] if len(x) > 0 else [0]
def prog_count_positive(x): return [sum(1 for v in x if v > 0)]
def prog_count_negative(x): return [sum(1 for v in x if v < 0)]
def prog_sum_of_squares(x): return [sum(v * v for v in x)]

# Access (6)
def prog_first(x): return [x[0]] if x else [0]
def prog_last(x): return [x[-1]] if x else [0]
def prog_second(x): return [x[1]] if len(x) > 1 else [0]
def prog_nth_2(x): return [x[2]] if len(x) > 2 else [0]
def prog_second_to_last(x): return [x[-2]] if len(x) > 1 else [0]
def prog_middle(x): return [x[len(x) // 2]] if x else [0]

# Transformation (10)
def prog_double_all(x): return [v * 2 for v in x]
def prog_square_all(x): return [v * v for v in x]
def prog_negate_all(x): return [-v for v in x]
def prog_reverse(x): return list(reversed(x))
def prog_sort_asc(x): return sorted(x)
def prog_sort_desc(x): return sorted(x, reverse=True)
def prog_filter_positive(x): return [v for v in x if v > 0] or [0]
def prog_filter_even(x): return [v for v in x if v % 2 == 0] or [0]
def prog_filter_odd(x): return [v for v in x if v % 2 != 0] or [0]
def prog_unique(x): return list(dict.fromkeys(x))

# Arithmetic (6) — operate on pairs or single values packed in list
def prog_add(x): return [x[0] + x[1]] if len(x) >= 2 else [sum(x)]
def prog_subtract(x): return [x[0] - x[1]] if len(x) >= 2 else x
def prog_multiply(x): return [x[0] * x[1]] if len(x) >= 2 else x
def prog_divide(x): return [_safe_div(x[0], x[1])] if len(x) >= 2 else x
def prog_power(x): return [x[0] ** min(x[1], 10)] if len(x) >= 2 else x
def prog_modulo(x): return [_safe_mod(x[0], x[1])] if len(x) >= 2 else x

# Conditional (4) — operate on scalar or pair
def prog_max_of_two(x): return [max(x[0], x[1])] if len(x) >= 2 else x
def prog_min_of_two(x): return [min(x[0], x[1])] if len(x) >= 2 else x
def prog_abs_single(x): return [abs(x[0])] if x else [0]
def prog_sign(x): return [1 if x[0] > 0 else (-1 if x[0] < 0 else 0)] if x else [0]

# Composition (8)
def prog_sum_then_double(x): return [sum(x) * 2]
def prog_double_then_sum(x): return [sum(v * 2 for v in x)]
def prog_filter_positive_then_sum(x): return [sum(v for v in x if v > 0)]
def prog_sum_of_squares_comp(x): return [sum(v * v for v in x)]
def prog_product_of_positives(x):
    pos = [v for v in x if v > 0]
    r = 1
    for v in pos: r *= v
    return [r] if pos else [0]
def prog_mean_of_sorted(x):
    s = sorted(x)
    return [sum(s) / max(len(s), 1)]
def prog_max_of_doubled(x): return [max(v * 2 for v in x)] if x else [0]
def prog_length_of_filtered(x): return [len([v for v in x if v > 0])]

# Recursive (4)
def prog_factorial(x):
    n = min(abs(int(x[0])), 12) if x else 0
    r = 1
    for i in range(2, n + 1): r *= i
    return [r]
def prog_fibonacci(x):
    n = min(abs(int(x[0])), 20) if x else 0
    a, b = 0, 1
    for _ in range(n): a, b = b, a + b
    return [a]
def prog_sum_recursive(x):
    # same as sum, but conceptually recursive
    return [sum(x)]
def prog_product_recursive(x):
    r = 1
    for v in x: r *= v
    return [r]


# Category definitions
PROGRAM_CATEGORIES = {
    "aggregation": [
        ("sum", prog_sum), ("product", prog_product), ("max", prog_max),
        ("min", prog_min), ("mean", prog_mean), ("length", prog_length),
        ("range", prog_range), ("count_positive", prog_count_positive),
        ("count_negative", prog_count_negative), ("sum_of_squares", prog_sum_of_squares),
    ],
    "access": [
        ("first", prog_first), ("last", prog_last), ("second", prog_second),
        ("nth_2", prog_nth_2), ("second_to_last", prog_second_to_last),
        ("middle", prog_middle),
    ],
    "transformation": [
        ("double_all", prog_double_all), ("square_all", prog_square_all),
        ("negate_all", prog_negate_all), ("reverse", prog_reverse),
        ("sort_asc", prog_sort_asc), ("sort_desc", prog_sort_desc),
        ("filter_positive", prog_filter_positive), ("filter_even", prog_filter_even),
        ("filter_odd", prog_filter_odd), ("unique", prog_unique),
    ],
    "arithmetic": [
        ("add", prog_add), ("subtract", prog_subtract), ("multiply", prog_multiply),
        ("divide", prog_divide), ("power", prog_power), ("modulo", prog_modulo),
    ],
    "conditional": [
        ("max_of_two", prog_max_of_two), ("min_of_two", prog_min_of_two),
        ("abs_single", prog_abs_single), ("sign", prog_sign),
    ],
    "composition": [
        ("sum_then_double", prog_sum_then_double),
        ("double_then_sum", prog_double_then_sum),
        ("filter_positive_then_sum", prog_filter_positive_then_sum),
        ("sum_of_squares_comp", prog_sum_of_squares_comp),
        ("product_of_positives", prog_product_of_positives),
        ("mean_of_sorted", prog_mean_of_sorted),
        ("max_of_doubled", prog_max_of_doubled),
        ("length_of_filtered", prog_length_of_filtered),
    ],
    "recursive": [
        ("factorial", prog_factorial), ("fibonacci", prog_fibonacci),
        ("sum_recursive", prog_sum_recursive), ("product_recursive", prog_product_recursive),
    ],
}


def get_program_library():
    """Return dict: name -> (function, category)."""
    lib = {}
    for cat, progs in PROGRAM_CATEGORIES.items():
        for name, func in progs:
            lib[name] = (func, cat)
    return lib


def get_all_program_names():
    """Return list of all 48 program names."""
    names = []
    for cat, progs in PROGRAM_CATEGORIES.items():
        for name, func in progs:
            names.append(name)
    return names


def get_category_for_program(name):
    """Return category string for a program name."""
    for cat, progs in PROGRAM_CATEGORIES.items():
        for pname, _ in progs:
            if pname == name:
                return cat
    return "unknown"


# ---------------------------------------------------------------------------
# Synthetic program generation for scaling tests
# ---------------------------------------------------------------------------
def generate_synthetic_programs(target_size, rng=None):
    """Generate a program library of the given size by composing/parameterizing base programs.
    Returns dict: name -> (function, category)."""
    if rng is None:
        rng = np.random.default_rng(42)

    lib = get_program_library()
    base_names = list(lib.keys())

    # Start with base programs
    result = dict(lib)

    # Parameterized arithmetic: add_N, multiply_by_N
    param_id = 0
    while len(result) < target_size:
        param_id += 1
        k = param_id
        # add_k
        name = f"add_{k}"
        result[name] = (lambda x, k=k: [v + k for v in x], "arithmetic")
        if len(result) >= target_size:
            break
        # multiply_by_k
        name = f"multiply_by_{k}"
        result[name] = (lambda x, k=k: [v * k for v in x], "arithmetic")
        if len(result) >= target_size:
            break
        # subtract_k
        name = f"subtract_{k}"
        result[name] = (lambda x, k=k: [v - k for v in x], "arithmetic")
        if len(result) >= target_size:
            break
        # Compositions: sum_then_add_k
        name = f"sum_then_add_{k}"
        result[name] = (lambda x, k=k: [sum(x) + k], "composition")
        if len(result) >= target_size:
            break
        # filter_gt_k
        name = f"filter_gt_{k}"
        result[name] = (lambda x, k=k: [v for v in x if v > k] or [0], "conditional")
        if len(result) >= target_size:
            break
        # power_k
        name = f"power_{k}"
        result[name] = (lambda x, k=k: [v ** min(k, 5) for v in x], "transformation")
        if len(result) >= target_size:
            break

    # Trim to exact size
    names = list(result.keys())[:target_size]
    return {n: result[n] for n in names}


def generate_test_inputs(program_name, rng=None, n_examples=3):
    """Generate appropriate test input lists for a program."""
    if rng is None:
        rng = np.random.default_rng(42)

    cat = get_category_for_program(program_name)
    inputs = []
    for i in range(n_examples):
        if cat in ("arithmetic", "conditional") or program_name in ("max_of_two", "min_of_two"):
            # Pair inputs
            inp = list(rng.integers(-10, 10, size=2).astype(int))
        elif cat == "recursive":
            inp = [int(rng.integers(1, 8))]
        else:
            length = rng.integers(3, 8)
            inp = list(rng.integers(-10, 10, size=length).astype(int))
        inputs.append(inp)
    return inputs


# ---------------------------------------------------------------------------
# Triple convergence checker
# ---------------------------------------------------------------------------
def check_triple_convergence(eps_osc, eps_cat, eps_par, delta=0.1):
    """Check if three residue distances converge within threshold delta.
    Returns (converged: bool, max_discrepancy: float)."""
    d1 = abs(eps_osc - eps_cat)
    d2 = abs(eps_cat - eps_par)
    d3 = abs(eps_osc - eps_par)
    max_disc = max(d1, d2, d3)
    return max_disc < delta, float(max_disc)


# ---------------------------------------------------------------------------
# NetworkNode for security simulations
# ---------------------------------------------------------------------------
class NetworkNode:
    """Network node for thermodynamic security simulations."""
    __slots__ = ("frequency", "phase", "is_legitimate", "gpsdo_coupling",
                 "timing_variance", "node_id")

    def __init__(self, node_id, frequency=1.0, phase=0.0,
                 is_legitimate=True, gpsdo_coupling=0.1):
        self.node_id = node_id
        self.frequency = float(frequency)
        self.phase = float(phase)
        self.is_legitimate = bool(is_legitimate)
        self.gpsdo_coupling = float(gpsdo_coupling)
        self.timing_variance = 0.0

    def step(self, dt, rng, injection_rate=0.0):
        """Advance node state by one time step.
        Legitimate: variance decays via GPSDO coupling.
        Attacker: variance grows via injection_rate."""
        if self.is_legitimate:
            # GPSDO coupling: dσ²/dt = -σ²/τ
            tau = 1.0 / (self.gpsdo_coupling + EPSILON)
            self.timing_variance *= np.exp(-dt / tau)
            # Small thermal noise
            self.timing_variance += rng.normal(0, 0.001) ** 2
        else:
            # Attacker injects timing perturbations
            self.timing_variance += injection_rate * dt
            self.timing_variance += rng.normal(0, injection_rate * 0.1) ** 2
        self.timing_variance = max(self.timing_variance, 0.0)
        # Phase evolution
        self.phase += self.frequency * dt * 2 * np.pi
        return self.timing_variance


# ---------------------------------------------------------------------------
# Gear ratio computation
# ---------------------------------------------------------------------------
def compute_gear_ratio(omega_i, omega_j):
    """Compute gear ratio R_ij = omega_i / omega_j."""
    if abs(omega_j) < EPSILON:
        return float('inf')
    return omega_i / omega_j


def compute_all_gear_ratios(frequencies):
    """Compute all pairwise gear ratios. Returns NxN matrix."""
    freqs = np.asarray(frequencies, dtype=np.float64)
    n = len(freqs)
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if abs(freqs[j]) > EPSILON:
                R[i, j] = freqs[i] / freqs[j]
            else:
                R[i, j] = float('inf')
    return R


# ---------------------------------------------------------------------------
# Network temperature and order parameter
# ---------------------------------------------------------------------------
def network_temperature(timing_variances):
    """T = variance of all packet timing variances."""
    return float(np.var(timing_variances))


def network_pressure(frequencies, volume=1.0):
    """P = sum(f_i^2) / (dim * V) — kinetic pressure analogue."""
    freqs = np.asarray(frequencies, dtype=np.float64)
    return float(np.sum(freqs ** 2) / (3.0 * volume))


def order_parameter(phases):
    """Psi = |<exp(i*phi)>| — phase coherence."""
    phases = np.asarray(phases, dtype=np.float64)
    return float(np.abs(np.mean(np.exp(1j * phases))))
