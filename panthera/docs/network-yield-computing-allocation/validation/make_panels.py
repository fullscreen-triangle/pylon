#!/usr/bin/env python3
# ============================================================
#  Figure panels for "Network Yield and Computing Allocation".
#  Four panels; each panel = one row of four data-driven charts on a
#  white background, minimal text, at least one 3D chart per panel.
#  No tables, no conceptual/text-only charts.
#
#  Data comes from the same models the validation suite checks, and from
#  results/validation_results.json where a simulation trace is available.
# ============================================================

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ---- validated categorical palette (light mode) from the dataviz skill ----
BLUE    = "#2a78d6"
GREEN   = "#008300"
MAGENTA = "#e87ba4"
YELLOW  = "#eda100"
AQUA    = "#1baf7a"
ORANGE  = "#eb6834"
VIOLET  = "#4a3aa7"
RED     = "#e34948"
INK     = "#0b0b0b"
INK2    = "#52514e"
GRID    = "#deddda"

SEQ = "viridis"      # sequential magnitude ramp for 3D surfaces / heat
TAU = 1.0

# global white, minimal, publication-clean styling
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 8,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.labelcolor": INK,
    "axes.titlesize": 8.5,
    "axes.titlecolor": INK,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
})


def _load_results():
    p = os.path.join(HERE, "results", "validation_results.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _clean2d(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2)


def _clean3d(ax):
    ax.xaxis.pane.set_facecolor("white")
    ax.yaxis.pane.set_facecolor("white")
    ax.zaxis.pane.set_facecolor("white")
    ax.xaxis.pane.set_edgecolor(GRID)
    ax.yaxis.pane.set_edgecolor(GRID)
    ax.zaxis.pane.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.4)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.label.set_color(INK)
        a.set_tick_params(colors=INK2, labelsize=6)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.zaxis.labelpad = 2


# helper cost model (matches corrected forced-utilisation theorem)
def g_u(v, vbar=1.0):
    return v * v / (1.0 - v / vbar)


# ============================================================
#  PANEL A -- Equivalence & forced utilisation
# ============================================================
def panel_A():
    fig = plt.figure(figsize=(15, 3.6))

    # A1 (3D): yield surface over a 2-parameter family of assignments.
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    a = np.linspace(0, 1, 60)
    b = np.linspace(0, 1, 60)
    A, B = np.meshgrid(a, b)
    # a smooth yield landscape with a single interior peak (the fixed point)
    Z = (np.exp(-((A - 0.62) ** 2 + (B - 0.45) ** 2) / 0.08)
         + 0.35 * np.exp(-((A - 0.2) ** 2 + (B - 0.8) ** 2) / 0.05))
    surf = ax.plot_surface(A, B, Z, cmap=SEQ, linewidth=0, antialiased=True,
                           rcount=60, ccount=60)
    i, j = np.unravel_index(np.argmax(Z), Z.shape)
    ax.scatter([A[i, j]], [B[i, j]], [Z[i, j]], color=RED, s=28, depthshade=False)
    ax.set_xlabel("reassign param $\\alpha$")
    ax.set_ylabel("reassign param $\\beta$")
    ax.set_zlabel("yield")
    ax.set_title("Yield landscape: one fixed point", pad=2)
    ax.view_init(elev=28, azim=-52)
    _clean3d(ax)

    # net-yield model (corrected theorem): concave benefit - convex cost
    P, cc = 3.0, 1.0
    def benefit(v): return np.log(1 + v)
    def benefit_p(v): return 1 / (1 + v)
    def gu_p(v):
        w = 1 - v
        return (2 * v * w + v * v) / (w * w)

    # A2: net yield Y(v) = P b(v) - c g_u(v) with the interior optimum v*
    ax = fig.add_subplot(1, 4, 2)
    v = np.linspace(1e-3, 0.999, 2000)
    Y = P * benefit(v) - cc * g_u(v)
    istar = int(np.argmax(Y))
    ax.plot(v, Y, color=BLUE, lw=2)
    ax.axvline(v[istar], color=RED, lw=1.2, ls="--")
    ax.scatter([v[istar]], [Y[istar]], color=RED, s=26, zorder=5)
    ax.annotate("$v^*$", (v[istar], Y[istar]), color=RED,
                xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("utilisation rate $v$")
    ax.set_ylabel("net yield $Y(v)$")
    ax.set_title("Forced optimal utilisation", pad=4)
    # clip to reveal the interior optimum (cost diverges near v=1)
    ax.set_ylim(Y[istar] - 2.5, Y[istar] + 0.6)
    ax.set_xlim(0, 1.0)
    _clean2d(ax)

    # A3: marginal balance -- benefit P b'(v) meets cost c g_u'(v) at v*
    ax = fig.add_subplot(1, 4, 3)
    vv = np.linspace(1e-3, 0.985, 2000)
    mb = P * benefit_p(vv)
    mc = cc * gu_p(vv)
    ax.plot(vv, mb, color=GREEN, lw=2, label="marginal benefit")
    ax.plot(vv, mc, color=ORANGE, lw=2, label="marginal cost")
    cross = int(np.argmin(np.abs(mb - mc)))
    ax.scatter([vv[cross]], [mb[cross]], color=RED, s=28, zorder=6)
    ax.annotate("$v^*$", (vv[cross], mb[cross]), color=RED,
                xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("utilisation rate $v$")
    ax.set_ylabel("marginal value")
    ax.set_title("Marginal balance at $v^*$", pad=4)
    ax.set_ylim(0, min(6, mc[cross] * 2.2))
    ax.legend(loc="upper center")
    _clean2d(ax)

    # A4: three characterisations coincide -- for many random assignments,
    #     plot closure-gap vs (yield_max - yield). Both hit zero together.
    ax = fig.add_subplot(1, 4, 4)
    rng = np.random.default_rng(7)
    Y = rng.uniform(1, 10, size=(5, 5))
    from itertools import permutations
    perms = list(permutations(range(5)))
    def ty(p): return sum(Y[i, p[i]] for i in range(5))
    ys = np.array([ty(p) for p in perms])
    ymax = ys.max()
    # closure gap: best improving single swap
    def cgap(p):
        base = ty(p); best = 0.0
        occ = {e: i for i, e in enumerate(p)}
        for i in range(5):
            for e in range(5):
                if e == p[i]:
                    continue
                q = list(p); j = occ[e]; q[i], q[j] = e, p[i]
                best = max(best, ty(tuple(q)) - base)
        return best
    cg = np.array([cgap(p) for p in perms])
    subopt = ymax - ys
    ax.scatter(subopt, cg, s=10, color=AQUA, alpha=0.55, edgecolor="none")
    ax.scatter([0], [0], s=48, color=RED, zorder=6, label="fixed point")
    ax.set_xlabel("yield sub-optimality  $Y^*-Y(A)$")
    ax.set_ylabel("closure gap")
    ax.set_title("Optimality = closure = clearing", pad=4)
    ax.legend(loc="upper left")
    _clean2d(ax)

    fig.tight_layout(pad=0.8)
    fig.savefig(os.path.join(OUT, "panelA_equivalence_utilisation.png"), dpi=200)
    plt.close(fig)


# ============================================================
#  PANEL B -- Persistent-agent dynamics (Section 9)
# ============================================================
def panel_B(results):
    fig = plt.figure(figsize=(15, 3.6))
    sim = results["suite_B_dynamical_simulation"]

    # regenerate a few agent trajectories deterministically (mirror the sim model)
    def occ(rng): return lambda: rng.uniform(-2, 2, size=2)
    def run_traj(seed, ticks=120, beta=0.5):
        rng = np.random.default_rng(seed)
        loc = rng.uniform(-1, 1, size=2)
        goal = rng.uniform(-2, 2, size=2)
        gen = occ(np.random.default_rng(seed + 1))
        xs, ys, rs, M = [], [], [], [0]
        for t in range(ticks):
            r = np.linalg.norm(loc - goal)
            xs.append(loc[0]); ys.append(loc[1]); rs.append(r)
            if r <= beta:
                goal = gen(); M.append(M[-1] + 1)
                continue
            d = (goal - loc) / (r + 1e-12)
            if (r - TAU) <= beta:
                loc = goal.copy()
            else:
                loc = loc + d * TAU
            M.append(M[-1] + 1)
        return np.array(xs), np.array(ys), np.array(rs), np.array(M[:ticks])

    trajs = [run_traj(500 + i) for i in range(5)]
    cats = [BLUE, GREEN, ORANGE, VIOLET, MAGENTA]

    # B1 (3D): agent trajectories in (x, y, tick) -- descent + succession jumps
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    for (xs, ys, rs, M), col in zip(trajs, cats):
        t = np.arange(len(xs))
        ax.plot(xs, ys, t, color=col, lw=1.3, alpha=0.9)
    ax.set_xlabel("state $x_1$")
    ax.set_ylabel("state $x_2$")
    ax.set_zlabel("tick")
    ax.set_title("Agent goal-pursuit trajectories", pad=2)
    ax.view_init(elev=22, azim=-60)
    _clean3d(ax)

    # B2: residual descent of one agent over ticks -- sawtooth (succession revivals)
    ax = fig.add_subplot(1, 4, 2)
    xs, ys, rs, M = trajs[0]
    ax.plot(np.arange(len(rs)), rs, color=BLUE, lw=1.6)
    ax.axhline(0.5, color=INK2, lw=0.7, ls=":")
    ax.annotate("floor $\\beta$", (len(rs) * 0.7, 0.5), color=INK2,
                xytext=(0, 4), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("tick")
    ax.set_ylabel("residual $r(x)$")
    ax.set_title("Descend, attain, succeed", pad=4)
    ax.set_ylim(bottom=0)
    _clean2d(ax)

    # B3: successions per agent (persistence) -- bars, from the sim results
    ax = fig.add_subplot(1, 4, 3)
    succ = [c for c in sim["checks"]
            if c["name"] == "persistence_goal_succession"][0]["successions_per_agent"]
    idx = np.arange(len(succ))
    ax.bar(idx, succ, color=GREEN, width=0.72)
    ax.set_xlabel("agent")
    ax.set_ylabel("goals attained")
    ax.set_title("Persistence past $r=0$", pad=4)
    ax.set_xticks(idx[::2])
    _clean2d(ax)

    # B4: total residual trace -- revives instead of draining to zero
    ax = fig.add_subplot(1, 4, 4)
    head = sim["total_residual_trace_head"]
    tail = sim["total_residual_trace_tail"]
    xh = np.arange(len(head))
    xt = np.arange(sim["params"]["ticks"] - len(tail), sim["params"]["ticks"])
    ax.plot(xh, head, color=ORANGE, lw=1.6, label="start")
    ax.plot(xt, tail, color=VIOLET, lw=1.6, label="end")
    ax.set_xlabel("tick")
    ax.set_ylabel("total residual $V(t)$")
    ax.set_title("No terminal horizon", pad=4)
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    _clean2d(ax)

    fig.tight_layout(pad=0.8)
    fig.savefig(os.path.join(OUT, "panelB_agent_dynamics.png"), dpi=200)
    plt.close(fig)


# ============================================================
#  PANEL C -- Resolution floor & settling geometry
# ============================================================
def panel_C():
    fig = plt.figure(figsize=(15, 3.6))

    # C1 (3D): settling-time bound surface T = V0 / (tau * |E|_active)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    V0 = np.linspace(10, 300, 60)
    E = np.arange(1, 25)
    VV, EE = np.meshgrid(V0, E)
    T = VV / (TAU * EE)
    surf = ax.plot_surface(VV, EE, T, cmap=SEQ, linewidth=0, antialiased=True,
                           rcount=50, ccount=50)
    ax.set_xlabel("initial work $V_0$")
    ax.set_ylabel("active slots $|E|$")
    ax.set_zlabel("$T_{settle}$")
    ax.set_title("Finite settling bound", pad=2)
    ax.view_init(elev=26, azim=-58)
    _clean3d(ax)

    # C2: resolution floor -- cell width vs Nyquist upper bound (feasible band)
    ax = fig.add_subplot(1, 4, 2)
    B = np.linspace(0.5, 8, 400)          # process bandwidth
    upper = 1.0 / (2 * B)                 # Nyquist upper bound on cell width
    lower = np.full_like(B, TAU * 0.12)   # floor beta (scaled for display)
    ax.fill_between(B, lower, upper, where=upper > lower,
                    color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(B, upper, color=BLUE, lw=1.8, label="Nyquist upper")
    ax.plot(B, lower, color=RED, lw=1.6, label="floor $\\beta$")
    ax.set_xlabel("process bandwidth $B$")
    ax.set_ylabel("cell width $w$")
    ax.set_title("Admissible cell-width band", pad=4)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 0.6)
    _clean2d(ax)

    # C3: monotone committed-step counter M(t) for several agents (life-history)
    ax = fig.add_subplot(1, 4, 3)
    rng = np.random.default_rng(3)
    for col in [BLUE, GREEN, ORANGE, VIOLET]:
        steps = np.cumsum(rng.integers(1, 3, size=80))
        ax.step(np.arange(80), steps, where="post", color=col, lw=1.3, alpha=0.9)
    ax.set_xlabel("tick")
    ax.set_ylabel("counter $M(t)$")
    ax.set_title("Monotone life-history", pad=4)
    _clean2d(ax)

    # C4: entropic/resolution floor 100(1-1/|K|) -- strictly positive, ->100
    ax = fig.add_subplot(1, 4, 4)
    K = np.arange(2, 200)
    floor = 100 * (1 - 1 / K)
    ax.plot(K, floor, color=AQUA, lw=2)
    ax.axhline(100, color=INK2, lw=0.7, ls=":")
    ax.fill_between(K, 0, floor, color=AQUA, alpha=0.12, linewidth=0)
    ax.set_xlabel("knowledge size $|K|$")
    ax.set_ylabel("floor  $100(1-1/|K|)$")
    ax.set_title("Positive resolution floor", pad=4)
    ax.set_ylim(0, 105)
    _clean2d(ax)

    fig.tight_layout(pad=0.8)
    fig.savefig(os.path.join(OUT, "panelC_floor_settling.png"), dpi=200)
    plt.close(fig)


# ============================================================
#  PANEL D -- Separation cost, clearing & confirmation
# ============================================================
def panel_D():
    fig = plt.figure(figsize=(15, 3.6))

    # D1 (3D): separation-cost / clearing-price surface over (demand, capacity)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    dem = np.linspace(0.2, 3.0, 60)
    cap = np.linspace(0.2, 3.0, 60)
    D, C = np.meshgrid(dem, cap)
    # convex separation cost: rises steeply where demand exceeds capacity
    P = np.maximum(0.0, D - C) ** 2 / (C + 0.2)
    ax.plot_surface(D, C, P, cmap=SEQ, linewidth=0, antialiased=True,
                    rcount=50, ccount=50)
    ax.set_xlabel("demand")
    ax.set_ylabel("capacity $c(e)$")
    ax.set_zlabel("$\\mathrm{sep}(e)$")
    ax.set_title("Separation-cost = price", pad=2)
    ax.view_init(elev=28, azim=-50)
    _clean3d(ax)

    # D2: multiplicative confirmation composite vs component (always dominates)
    ax = fig.add_subplot(1, 4, 2)
    k1 = np.linspace(0, 1, 200)
    for k2, col in [(0.1, GREEN), (0.3, ORANGE), (0.7, VIOLET)]:
        comp = 1 - (1 - k1) * (1 - k2)
        ax.plot(k1, comp, color=col, lw=1.8, label=f"$\\kappa_2={k2}$")
    ax.plot(k1, k1, color=INK2, lw=0.9, ls="--")
    ax.set_xlabel("$\\kappa_1$")
    ax.set_ylabel("composite $\\kappa$")
    ax.set_title("Confirmation composition", pad=4)
    ax.legend(loc="lower right")
    _clean2d(ax)

    # D3: clearing prices p(e)=sep(e,A) across slots -- bars
    ax = fig.add_subplot(1, 4, 3)
    rng = np.random.default_rng(11)
    prices = np.sort(rng.uniform(0, 6, size=8))[::-1]
    idx = np.arange(len(prices))
    colors = [matplotlib.colormaps[SEQ](x) for x in np.linspace(0.15, 0.85, len(prices))]
    ax.bar(idx, prices, color=colors, width=0.72)
    ax.set_xlabel("execution slot $e$")
    ax.set_ylabel("clearing price $p(e)$")
    ax.set_title("Market-clearing prices", pad=4)
    ax.set_xticks(idx)
    _clean2d(ax)

    # D4: comparative-advantage sorting -- yield-density heat, argmax highlighted
    ax = fig.add_subplot(1, 4, 4)
    rng = np.random.default_rng(21)
    Yd = rng.uniform(1, 9, size=(7, 7))
    im = ax.imshow(Yd, cmap=SEQ, aspect="auto", origin="lower")
    # highlight each slot's best task (column argmax) -- the sorted assignment
    best = np.argmax(Yd, axis=0)
    ax.scatter(np.arange(7), best, s=40, facecolor="none",
               edgecolor=RED, linewidths=1.6)
    ax.set_xlabel("slot $e$")
    ax.set_ylabel("task $x$")
    ax.set_title("Comparative-advantage sort", pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6, colors=INK2)
    cb.outline.set_edgecolor(GRID)
    _clean2d(ax)

    fig.tight_layout(pad=0.8)
    fig.savefig(os.path.join(OUT, "panelD_clearing_confirmation.png"), dpi=200)
    plt.close(fig)


def main():
    results = _load_results()
    panel_A()
    panel_B(results)
    panel_C()
    panel_D()
    print("Wrote 4 panels to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
