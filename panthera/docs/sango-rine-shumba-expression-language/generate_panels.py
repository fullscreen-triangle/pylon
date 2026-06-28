"""
generate_panels.py — SRN paper figure panels
Four panels, 4 charts each (2x2 grid), white background, at least one 3D per panel.
All charts plot real mathematical data from the paper's theorems.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from matplotlib import cm
import math, os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

VIRIDIS = cm.viridis
PLASMA  = cm.plasma
CIVIDIS = cm.cividis
INFERNO = cm.inferno

# ─── helpers ──────────────────────────────────────────────────────────────────

def shell_capacity(n): return 2 * n * n

def floor_val(K): return 100.0 * (1.0 - 1.0 / K)

def T(n, d): return d * (1 + d) ** (n - 1)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Partition Space Geometry
# ══════════════════════════════════════════════════════════════════════════════

def panel1():
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    fig.patch.set_facecolor('white')

    # ── Chart 1: Shell capacity C(n) = 2n²  (bar)
    ax1 = fig.add_subplot(2, 2, 1)
    ns = np.arange(1, 9)
    caps = [shell_capacity(n) for n in ns]
    colors = VIRIDIS(np.linspace(0.2, 0.9, len(ns)))
    ax1.bar(ns, caps, color=colors, width=0.6, edgecolor='none')
    ax1.set_xlabel('n', fontsize=11)
    ax1.set_ylabel('C(n)', fontsize=11)
    ax1.set_xticks(ns)
    ax1.set_facecolor('white')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax1.set_axisbelow(True)

    # ── Chart 2: 3D scatter of (n, ℓ, m) for n=1..5
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_facecolor('white')
    xs, ys, zs, cs = [], [], [], []
    for n in range(1, 6):
        for l in range(n):
            for m in range(-l, l + 1):
                for _ in ('+', '-'):
                    xs.append(n); ys.append(l); zs.append(m); cs.append(n)
    sc = ax2.scatter(xs, ys, zs, c=cs, cmap=PLASMA, s=14, alpha=0.75, edgecolors='none')
    ax2.set_xlabel('n', fontsize=9, labelpad=3)
    ax2.set_ylabel('l', fontsize=9, labelpad=3)
    ax2.set_zlabel('m', fontsize=9, labelpad=3)
    ax2.tick_params(labelsize=7)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    # ── Chart 3: Cumulative coordinate count
    ax3 = fig.add_subplot(2, 2, 3)
    ns_long = np.arange(1, 11)
    cumulative = [sum(shell_capacity(k) for k in range(1, n + 1)) for n in ns_long]
    ax3.plot(ns_long, cumulative, color='#1a4e8a', linewidth=2.2)
    ax3.fill_between(ns_long, cumulative, alpha=0.12, color='#1a4e8a')
    ax3.set_xlabel('n', fontsize=11)
    ax3.set_ylabel('Σ C(k)', fontsize=11)
    ax3.set_xticks(ns_long)
    ax3.set_facecolor('white')
    ax3.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax3.set_axisbelow(True)

    # ── Chart 4: (n, ℓ) heatmap — number of valid m values = 2ℓ+1
    ax4 = fig.add_subplot(2, 2, 4)
    N_max, L_max = 6, 6
    Z = np.full((L_max, N_max), np.nan)
    for n in range(1, N_max + 1):
        for l in range(min(n, L_max)):
            Z[l, n - 1] = 2 * l + 1
    im = ax4.pcolormesh(np.arange(1, N_max + 1), np.arange(0, L_max),
                        Z, cmap=CIVIDIS, shading='nearest')
    plt.colorbar(im, ax=ax4, shrink=0.8, label='2ℓ+1')
    ax4.set_xlabel('n', fontsize=11)
    ax4.set_ylabel('l', fontsize=11)
    ax4.set_facecolor('white')

    plt.subplots_adjust(hspace=0.38, wspace=0.38)
    out = os.path.join(OUT, 'panel1_partition_geometry.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Entropic Floor & Catalysis
# ══════════════════════════════════════════════════════════════════════════════

def panel2():
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    fig.patch.set_facecolor('white')

    # ── Chart 1: S_flat(K) line + fill
    ax1 = fig.add_subplot(2, 2, 1)
    Ks = np.linspace(2, 200, 400)
    Sf = 100.0 * (1.0 - 1.0 / Ks)
    ax1.plot(Ks, Sf, color='#c0392b', linewidth=2.2)
    ax1.fill_between(Ks, Sf, alpha=0.15, color='#c0392b')
    ax1.axhline(100, linestyle='--', linewidth=0.8, color='#888888', alpha=0.6)
    ax1.set_xlabel('|K|', fontsize=11)
    ax1.set_ylabel('S♭', fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.set_facecolor('white')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax1.set_axisbelow(True)

    # ── Chart 2: 3D surface — dual-receiver floor average
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_facecolor('white')
    k1 = np.linspace(2, 50, 40)
    k2 = np.linspace(2, 50, 40)
    K1, K2 = np.meshgrid(k1, k2)
    Z2 = 50.0 * (2.0 - 1.0 / K1 - 1.0 / K2)
    ax2.plot_surface(K1, K2, Z2, cmap=INFERNO, alpha=0.88,
                     linewidth=0, antialiased=True)
    ax2.set_xlabel('|K₁|', fontsize=9, labelpad=3)
    ax2.set_ylabel('|K₂|', fontsize=9, labelpad=3)
    ax2.set_zlabel('S♭ avg', fontsize=9, labelpad=3)
    ax2.tick_params(labelsize=7)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    # ── Chart 3: Catalytic residual decay (log scale)
    ax3 = fig.add_subplot(2, 2, 3)
    Ns = np.arange(0, 101)
    kappas = [0.05, 0.1, 0.2, 0.4, 0.7]
    palette = plt.cm.Blues(np.linspace(0.35, 0.95, len(kappas)))
    for kappa, col in zip(kappas, palette):
        residual = (1.0 - kappa) ** Ns
        ax3.plot(Ns, residual, color=col, linewidth=1.8)
    ax3.set_yscale('log')
    ax3.set_xlabel('N', fontsize=11)
    ax3.set_ylabel('(1−κ)ᴺ', fontsize=11)
    ax3.set_facecolor('white')
    ax3.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax3.set_axisbelow(True)

    # ── Chart 4: Composite catalytic power
    ax4 = fig.add_subplot(2, 2, 4)
    k1_range = np.linspace(0, 1, 200)
    for k2_fixed, col in zip([0.1, 0.3, 0.7],
                              ['#2ecc71', '#e67e22', '#8e44ad']):
        k_comb = 1.0 - (1.0 - k1_range) * (1.0 - k2_fixed)
        ax4.plot(k1_range, k_comb, color=col, linewidth=2.0)
    ax4.plot([0, 1], [0, 1], '--', color='#aaaaaa', linewidth=0.9)
    ax4.set_xlabel('κ₁', fontsize=11)
    ax4.set_ylabel('κ_comb', fontsize=11)
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.set_facecolor('white')
    ax4.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax4.set_axisbelow(True)

    plt.subplots_adjust(hspace=0.38, wspace=0.38)
    out = os.path.join(OUT, 'panel2_entropic_floor.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — Composition Inflation
# ══════════════════════════════════════════════════════════════════════════════

def panel3():
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    fig.patch.set_facecolor('white')

    # ── Chart 1: T(n, d) log-scale lines for d=1..4
    ax1 = fig.add_subplot(2, 2, 1)
    ns = np.arange(1, 11)
    palette = plt.cm.plasma(np.linspace(0.15, 0.85, 4))
    for d, col in zip([1, 2, 3, 4], palette):
        vals = [T(n, d) for n in ns]
        ax1.plot(ns, vals, color=col, linewidth=2.0, marker='o',
                 markersize=4, label=f'd={d}')
    ax1.set_yscale('log')
    ax1.set_xlabel('n', fontsize=11)
    ax1.set_ylabel('T(n,d)', fontsize=11)
    ax1.legend(fontsize=9, frameon=False)
    ax1.set_xticks(ns)
    ax1.set_facecolor('white')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax1.set_axisbelow(True)

    # ── Chart 2: 3D surface log₁₀(T(n,d))
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_facecolor('white')
    n_arr = np.arange(1, 9)
    d_arr = np.arange(1, 9)
    N2, D2 = np.meshgrid(n_arr, d_arr)
    Z3 = np.log10(np.vectorize(T)(N2, D2).astype(float))
    ax2.plot_surface(N2, D2, Z3, cmap=VIRIDIS, alpha=0.88,
                     linewidth=0, antialiased=True)
    ax2.set_xlabel('n', fontsize=9, labelpad=3)
    ax2.set_ylabel('d', fontsize=9, labelpad=3)
    ax2.set_zlabel('log₁₀T', fontsize=9, labelpad=3)
    ax2.tick_params(labelsize=7)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    # ── Chart 3: Heatmap log₁₀(T(n,d)) for n,d=1..6
    ax3 = fig.add_subplot(2, 2, 3)
    nd = 6
    heat = np.array([[math.log10(T(n, d)) for n in range(1, nd + 1)]
                     for d in range(1, nd + 1)])
    im3 = ax3.pcolormesh(np.arange(1, nd + 1), np.arange(1, nd + 1),
                         heat, cmap=PLASMA, shading='nearest')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='log₁₀T')
    ax3.set_xlabel('n', fontsize=11)
    ax3.set_ylabel('d', fontsize=11)
    ax3.set_facecolor('white')

    # ── Chart 4: Growth ratio T(n+1,d)/T(n,d) = (1+d) — horizontal lines
    ax4 = fig.add_subplot(2, 2, 4)
    ns4 = np.arange(1, 10)
    palette4 = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
    for d, col in zip([1, 2, 3, 4, 5], palette4):
        ratio = float(1 + d)
        ax4.hlines(ratio, ns4[0], ns4[-1], colors=col, linewidth=2.2,
                   label=f'd={d}')
        ax4.scatter(ns4, [ratio] * len(ns4), color=col, s=18, zorder=3)
    ax4.set_xlabel('n', fontsize=11)
    ax4.set_ylabel('T(n+1)/T(n)', fontsize=11)
    ax4.legend(fontsize=9, frameon=False)
    ax4.set_xticks(ns4)
    ax4.set_facecolor('white')
    ax4.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax4.set_axisbelow(True)

    plt.subplots_adjust(hspace=0.38, wspace=0.38)
    out = os.path.join(OUT, 'panel3_composition_inflation.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — Forest Dynamics
# ══════════════════════════════════════════════════════════════════════════════

def panel4():
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    fig.patch.set_facecolor('white')

    # ── Chart 1: Receiver-relative output heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    n_vals = np.arange(1, 6)
    m_vals = np.arange(-2, 3)
    M_fixed = 50
    Z1 = np.array([[n * 100 + m * 10 + M_fixed
                    for n in n_vals] for m in m_vals], dtype=float)
    im1 = ax1.pcolormesh(n_vals, m_vals, Z1, cmap=VIRIDIS, shading='nearest')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='output')
    ax1.set_xlabel('n', fontsize=11)
    ax1.set_ylabel('m', fontsize=11)
    ax1.set_facecolor('white')
    ax1.set_yticks(m_vals)

    # ── Chart 2: 3D surface κ_comb(κ1, κ2)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_facecolor('white')
    k1g = np.linspace(0, 1, 50)
    k2g = np.linspace(0, 1, 50)
    K1g, K2g = np.meshgrid(k1g, k2g)
    Zcomb = 1.0 - (1.0 - K1g) * (1.0 - K2g)
    ax2.plot_surface(K1g, K2g, Zcomb, cmap=PLASMA, alpha=0.88,
                     linewidth=0, antialiased=True)
    ax2.set_xlabel('κ₁', fontsize=9, labelpad=3)
    ax2.set_ylabel('κ₂', fontsize=9, labelpad=3)
    ax2.set_zlabel('κ_comb', fontsize=9, labelpad=3)
    ax2.tick_params(labelsize=7)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    # ── Chart 3: Channel count N_ch = 2n² horizontal bars
    ax3 = fig.add_subplot(2, 2, 3)
    n_dep = np.arange(1, 8)
    n_ch  = [2 * n * n for n in n_dep]
    colors3 = CIVIDIS(np.linspace(0.2, 0.85, len(n_dep)))
    ax3.barh(n_dep, n_ch, color=colors3, edgecolor='none', height=0.6)
    ax3.set_xlabel('N_ch', fontsize=11)
    ax3.set_ylabel('n', fontsize=11)
    ax3.set_yticks(n_dep)
    ax3.set_facecolor('white')
    ax3.xaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax3.set_axisbelow(True)

    # ── Chart 4: T(n, d) encoding capacity for d=4,8,16 — log scale
    ax4 = fig.add_subplot(2, 2, 4)
    ns4 = np.arange(1, 13)
    for d, col, lbl in zip([4, 8, 16],
                            ['#16a085', '#8e44ad', '#c0392b'],
                            ['d=4', 'd=8', 'd=16']):
        vals = [T(n, d) for n in ns4]
        ax4.plot(ns4, vals, color=col, linewidth=2.0, marker='o',
                 markersize=4, label=lbl)
    ax4.set_yscale('log')
    ax4.set_xlabel('n', fontsize=11)
    ax4.set_ylabel('T(n,d)', fontsize=11)
    ax4.legend(fontsize=9, frameon=False)
    ax4.set_xticks(ns4)
    ax4.set_facecolor('white')
    ax4.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax4.set_axisbelow(True)

    plt.subplots_adjust(hspace=0.38, wspace=0.38)
    out = os.path.join(OUT, 'panel4_forest_dynamics.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == '__main__':
    panel1()
    panel2()
    panel3()
    panel4()
    print("All 4 panels generated.")
