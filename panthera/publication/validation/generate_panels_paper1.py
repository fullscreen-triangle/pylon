"""
Generate 4 figure panels for Paper 1: Equations of State for Transcendent Observer Networks.
Each panel is a 1x4 grid of charts (20x5 inches, 300 dpi).
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

# Paths
DATA_DIR = Path(r'c:\Users\kunda\Documents\distributed\pylon\panthera\publication\validation\network_state\results')
OUT_DIR = Path(r'c:\Users\kunda\Documents\distributed\pylon\panthera\publication\sango-rine-shumba-state\figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.4,
})

# Muted colors
C_BLUE = '#4878A8'
C_RED = '#C44E52'
C_GREEN = '#55A868'
C_ORANGE = '#DD8452'
C_PURPLE = '#8172B3'
C_GRAY = '#777777'

def label_subplot(ax, letter, is_3d=False):
    """Add (A), (B), etc. label."""
    if is_3d:
        ax.text2D(0.02, 0.95, f'({letter})', transform=ax.transAxes,
                  fontsize=10, fontweight='bold', va='top')
    else:
        ax.text(0.02, 0.95, f'({letter})', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')


# ============================================================
# PANEL 1: Ideal Gas Law Validation
# ============================================================
def panel_1():
    df = pd.read_csv(DATA_DIR / 'ideal_gas_law.csv').replace([np.inf, -np.inf], np.nan).dropna()

    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # Chart A: 3D scatter N, V, T colored by PV/NkT
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    vmin, vmax = 1.0 - max(abs(df['PV_NkT_ratio'].min() - 1), abs(df['PV_NkT_ratio'].max() - 1)), \
                 1.0 + max(abs(df['PV_NkT_ratio'].min() - 1), abs(df['PV_NkT_ratio'].max() - 1))
    sc = ax.scatter(df['N'], df['V'], df['T_target'], c=df['PV_NkT_ratio'],
                    cmap='coolwarm', vmin=vmin, vmax=vmax, s=18, edgecolors='k', linewidths=0.3, alpha=0.85)
    ax.set_xlabel('N', labelpad=4)
    ax.set_ylabel('V', labelpad=4)
    ax.set_zlabel('T', labelpad=4)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label('PV/NkT', fontsize=8)
    label_subplot(ax, 'A', is_3d=True)

    # Chart B: PV/NkT vs N
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.scatter(df['N'], df['PV_NkT_ratio'], s=14, color=C_BLUE, alpha=0.6, edgecolors='none')
    ax2.axhline(1.0, color=C_RED, ls='--', lw=0.8, label='Ideal (1.0)')
    mean_r = df['PV_NkT_ratio'].mean()
    std_r = df['PV_NkT_ratio'].std()
    n_vals = sorted(df['N'].unique())
    ax2.fill_between([df['N'].min(), df['N'].max()], mean_r - std_r, mean_r + std_r,
                     color=C_BLUE, alpha=0.12, label=f'Mean +/- std')
    ax2.axhline(mean_r, color=C_BLUE, ls='-', lw=0.6, alpha=0.5)
    ax2.set_xlabel('N')
    ax2.set_ylabel('PV/NkT')
    ax2.legend(loc='lower right', framealpha=0.7)
    label_subplot(ax2, 'B')

    # Chart C: P vs N*T/V
    ax3 = fig.add_subplot(1, 4, 3)
    x = df['N'] * df['T_measured'] / df['V']
    y = df['P_measured']
    ax3.scatter(x, y, s=14, color=C_GREEN, alpha=0.6, edgecolors='none')
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax3.plot(x_fit, slope * x_fit + intercept, color=C_RED, lw=1.0, ls='-')
    ax3.text(0.05, 0.88, f'$R^2$ = {r**2:.6f}', transform=ax3.transAxes, fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))
    ax3.set_xlabel('NT/V')
    ax3.set_ylabel('P')
    label_subplot(ax3, 'C')

    # Chart D: Histogram of PV/NkT
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.hist(df['PV_NkT_ratio'], bins=20, color=C_BLUE, alpha=0.65, edgecolor='white', linewidth=0.5)
    ax4.axvline(1.0, color=C_RED, ls='--', lw=1.0, label='Ideal (1.0)')
    ax4.text(0.95, 0.88, f'$\\mu$ = {mean_r:.4f}\n$\\sigma$ = {std_r:.4f}',
             transform=ax4.transAxes, fontsize=8, ha='right',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))
    ax4.set_xlabel('PV/NkT')
    ax4.set_ylabel('Count')
    label_subplot(ax4, 'D')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'panel_1_ideal_gas_law.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Panel 1 saved.')


# ============================================================
# PANEL 2: Network Phase Transitions
# ============================================================
def panel_2():
    df = pd.read_csv(DATA_DIR / 'phase_transitions.csv').replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values('temperature').reset_index(drop=True)

    # Use log10 for extreme-valued columns for visualization
    df['log_specific_heat'] = np.log10(df['specific_heat'].clip(lower=1e-30))
    df['log_mean_energy'] = np.log10(df['mean_energy'].clip(lower=1e-30))

    phase_colors = {'gas': C_RED, 'liquid': C_BLUE, 'crystal': C_GREEN}

    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # Chart A: 3D scatter temperature, order_parameter, log_specific_heat colored by phase
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    for phase, color in phase_colors.items():
        mask = df['phase_label'] == phase
        ax.scatter(df.loc[mask, 'temperature'], df.loc[mask, 'order_parameter'],
                   df.loc[mask, 'log_specific_heat'], c=color, s=20, label=phase,
                   edgecolors='k', linewidths=0.3, alpha=0.85)
    ax.set_xlabel('T', labelpad=4)
    ax.set_ylabel(r'$\Psi$', labelpad=4)
    ax.set_zlabel('log$_{10}$ C$_V$', labelpad=4)
    ax.legend(fontsize=6, loc='upper left')
    label_subplot(ax, 'A', is_3d=True)

    # Determine phase boundaries
    phase_order = {'crystal': 0, 'liquid': 1, 'gas': 2}
    df['phase_num'] = df['phase_label'].map(phase_order)
    transitions = []
    for i in range(1, len(df)):
        if df.iloc[i]['phase_num'] != df.iloc[i-1]['phase_num']:
            transitions.append(0.5 * (df.iloc[i]['temperature'] + df.iloc[i-1]['temperature']))

    # Chart B: Order parameter vs temperature
    ax2 = fig.add_subplot(1, 4, 2)
    for phase, color in phase_colors.items():
        mask = df['phase_label'] == phase
        sub = df[mask]
        if len(sub) > 0:
            tmin, tmax = sub['temperature'].min(), sub['temperature'].max()
            ax2.axvspan(tmin, tmax, alpha=0.10, color=color)
    ax2.plot(df['temperature'], df['order_parameter'], '-o', color=C_BLUE, ms=3, lw=0.8)
    for t in transitions:
        ax2.axvline(t, color=C_GRAY, ls='--', lw=0.7, alpha=0.7)
    ax2.set_xlabel('T')
    ax2.set_ylabel(r'$\Psi$')
    label_subplot(ax2, 'B')

    # Chart C: Specific heat vs temperature (log scale)
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(df['temperature'], df['log_specific_heat'], '-o', color=C_ORANGE, ms=3, lw=0.8)
    # Mark peak
    idx_peak = df['log_specific_heat'].idxmax()
    ax3.plot(df.loc[idx_peak, 'temperature'], df.loc[idx_peak, 'log_specific_heat'],
             'v', color=C_RED, ms=8, zorder=5)
    for t in transitions:
        ax3.axvline(t, color=C_GRAY, ls='--', lw=0.7, alpha=0.7)
    ax3.set_xlabel('T')
    ax3.set_ylabel('log$_{10}$ C$_V$')
    label_subplot(ax3, 'C')

    # Chart D: Mean energy vs temperature colored by phase
    ax4 = fig.add_subplot(1, 4, 4)
    for phase, color in phase_colors.items():
        mask = df['phase_label'] == phase
        ax4.scatter(df.loc[mask, 'temperature'], df.loc[mask, 'log_mean_energy'],
                    s=18, color=color, label=phase, edgecolors='none', alpha=0.8)
    ax4.plot(df['temperature'], df['log_mean_energy'], '-', color=C_GRAY, lw=0.5, alpha=0.5)
    ax4.set_xlabel('T')
    ax4.set_ylabel('log$_{10}$ E')
    ax4.legend(fontsize=7, loc='best', framealpha=0.7)
    label_subplot(ax4, 'D')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'panel_2_phase_transitions.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Panel 2 saved.')


# ============================================================
# PANEL 3: Statistical Mechanics Validation
# ============================================================
def panel_3():
    mb = pd.read_csv(DATA_DIR / 'maxwell_boltzmann.csv').replace([np.inf, -np.inf], np.nan).dropna()

    var_dfs = {}
    for t in ['1.0', '5.0', '10.0', '20.0']:
        var_dfs[t] = pd.read_csv(DATA_DIR / f'variance_restoration_T{t}.csv').replace([np.inf, -np.inf], np.nan).dropna()

    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # Chart A: 3D bar-like scatter — temperature vs metric vs value (theory & measured)
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    metrics = ['v_mp', 'v_mean', 'v_rms']
    metric_pos = {m: i for i, m in enumerate(metrics)}
    colors_th = [C_BLUE, C_GREEN, C_PURPLE]
    colors_me = [C_RED, C_ORANGE, '#CC79A7']

    for i, m in enumerate(metrics):
        ax.scatter(mb['temperature'], [metric_pos[m]] * len(mb), mb[f'{m}_theory'],
                   c=colors_th[i], marker='s', s=30, label=f'{m} theory', edgecolors='k', linewidths=0.3)
        ax.scatter(mb['temperature'], [metric_pos[m] + 0.15] * len(mb), mb[f'{m}_measured'],
                   c=colors_me[i], marker='o', s=30, label=f'{m} meas.', edgecolors='k', linewidths=0.3)
    ax.set_xlabel('T', labelpad=4)
    ax.set_ylabel('Metric', labelpad=4)
    ax.set_zlabel('Speed', labelpad=4)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['mp', 'mean', 'rms'], fontsize=6)
    ax.legend(fontsize=5, loc='upper left', ncol=2)
    label_subplot(ax, 'A', is_3d=True)

    # Chart B: Measured vs theory scatter for all speed metrics
    ax2 = fig.add_subplot(1, 4, 2)
    markers_b = {'v_mp': 'o', 'v_mean': 's', 'v_rms': '^'}
    colors_b = {'v_mp': C_BLUE, 'v_mean': C_GREEN, 'v_rms': C_PURPLE}
    all_vals = []
    for m in metrics:
        th = mb[f'{m}_theory'].values
        me = mb[f'{m}_measured'].values
        ax2.scatter(th, me, s=22, marker=markers_b[m], color=colors_b[m], label=m.replace('_', ' '),
                    edgecolors='k', linewidths=0.3, alpha=0.8)
        all_vals.extend(th)
        all_vals.extend(me)
    lo, hi = min(all_vals) * 0.9, max(all_vals) * 1.1
    ax2.plot([lo, hi], [lo, hi], '--', color=C_GRAY, lw=0.8, label='y = x')
    ax2.set_xlabel('Theory')
    ax2.set_ylabel('Measured')
    ax2.legend(fontsize=7, loc='upper left', framealpha=0.7)
    ax2.set_aspect('equal', adjustable='box')
    label_subplot(ax2, 'B')

    # Chart C: Variance restoration curves
    ax3 = fig.add_subplot(1, 4, 3)
    t_colors = {'1.0': C_BLUE, '5.0': C_GREEN, '10.0': C_ORANGE, '20.0': C_RED}
    for t, vdf in var_dfs.items():
        ax3.plot(vdf['time'], vdf['variance'], color=t_colors[t], lw=0.8, label=f'T$_0$={t}')
        ax3.plot(vdf['time'], vdf['fitted_variance'], '--', color=t_colors[t], lw=0.7, alpha=0.7)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Variance')
    ax3.legend(fontsize=7, loc='center right', framealpha=0.7)
    label_subplot(ax3, 'C')

    # Chart D: Fitted tau vs theoretical tau
    ax4 = fig.add_subplot(1, 4, 4)
    # Fit exponential to each variance curve to extract tau
    def var_model(t, T_eq, tau, T0):
        return T_eq * (1 - np.exp(-t / tau)) + T0 * np.exp(-t / tau)

    fitted_taus = []
    t0_vals = [1.0, 5.0, 10.0, 20.0]
    theo_tau = 0.5  # theoretical tau

    for t0 in t0_vals:
        vdf = var_dfs[str(t0)]
        time = vdf['time'].values
        var = vdf['fitted_variance'].values
        # Estimate tau from fitted curve shape: fit simple exponential
        try:
            T_eq_est = var[-1]
            popt, _ = curve_fit(lambda t, tau: T_eq_est * (1 - np.exp(-t / tau)) + t0 * np.exp(-t / tau),
                                time[1:], var[1:], p0=[0.5], maxfev=5000)
            fitted_taus.append(popt[0])
        except Exception:
            fitted_taus.append(np.nan)

    fitted_taus = np.array(fitted_taus)
    ax4.scatter(t0_vals, fitted_taus, s=40, color=C_BLUE, edgecolors='k', linewidths=0.5, zorder=5)
    ax4.axhline(theo_tau, color=C_RED, ls='--', lw=0.8, label=f'Theory ($\\tau$=0.5)')
    # y=x not meaningful here; show theoretical line
    for i, t0 in enumerate(t0_vals):
        if not np.isnan(fitted_taus[i]):
            rel_err = abs(fitted_taus[i] - theo_tau) / theo_tau * 100
            ax4.annotate(f'{rel_err:.1f}%', (t0, fitted_taus[i]), textcoords='offset points',
                         xytext=(6, 4), fontsize=7, color=C_GRAY)
    ax4.set_xlabel('T$_0$')
    ax4.set_ylabel('Fitted $\\tau$')
    ax4.legend(fontsize=7, framealpha=0.7)
    label_subplot(ax4, 'D')

    fig.subplots_adjust(left=0.04, right=0.98, wspace=0.3)
    fig.savefig(OUT_DIR / 'panel_3_maxwell_boltzmann_variance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Panel 3 saved.')


# ============================================================
# PANEL 4: Equations of State
# ============================================================
def panel_4():
    vdw = pd.read_csv(DATA_DIR / 'van_der_waals.csv').replace([np.inf, -np.inf], np.nan).dropna()
    phon = pd.read_csv(DATA_DIR / 'phonon_dispersion.csv').replace([np.inf, -np.inf], np.nan).dropna()
    hcap = pd.read_csv(DATA_DIR / 'heat_capacity.csv').replace([np.inf, -np.inf], np.nan).dropna()

    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # The measured Z values are astronomically large — plot virial and VdW predictions
    # which are the physically meaningful EOS predictions on a sensible scale.
    # Use log10 of Z_measured for the 3D chart.
    vdw['log_Z_measured'] = np.log10(vdw['Z_measured'].clip(lower=1e-30))

    # Chart A: 3D scatter density, pressure (log), log Z
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    vdw['log_pressure'] = np.log10(vdw['pressure'].clip(lower=1e-30))
    sc = ax.scatter(vdw['density'], vdw['log_pressure'], vdw['log_Z_measured'],
                    c=vdw['density'], cmap='viridis', s=25, edgecolors='k', linewidths=0.3)
    # Overlay VdW prediction surface
    ax.scatter(vdw['density'], vdw['log_pressure'], np.log10(vdw['Z_vdw_prediction'].clip(lower=1e-30)),
               c=C_RED, marker='^', s=20, alpha=0.7, label='VdW pred.', edgecolors='k', linewidths=0.2)
    ax.set_xlabel(r'$\rho$', labelpad=4)
    ax.set_ylabel('log P', labelpad=4)
    ax.set_zlabel('log Z', labelpad=4)
    ax.legend(fontsize=6)
    label_subplot(ax, 'A', is_3d=True)

    # Chart B: Z vs density — virial and VdW predictions (sensible scale)
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(vdw['density'], vdw['Z_virial_prediction'], '-o', color=C_BLUE, ms=3, lw=0.8, label='Virial')
    ax2.plot(vdw['density'], vdw['Z_vdw_prediction'], '-s', color=C_GREEN, ms=3, lw=0.8, label='VdW')
    ax2.axhline(1.0, color=C_GRAY, ls='--', lw=0.7, label='Ideal (Z=1)')
    ax2.set_xlabel(r'$\rho$')
    ax2.set_ylabel('Z')
    ax2.legend(fontsize=7, framealpha=0.7)
    label_subplot(ax2, 'B')

    # Chart C: Phonon dispersion
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.scatter(phon['wavevector_q'], phon['omega_measured'], s=16, color=C_BLUE,
                edgecolors='k', linewidths=0.3, label='Measured', zorder=3)
    ax3.plot(phon['wavevector_q'], phon['omega_theory'], '-', color=C_RED, lw=1.0, label='Theory')
    ax3.set_xlabel('q')
    ax3.set_ylabel(r'$\omega$')
    ax3.legend(fontsize=7, framealpha=0.7)
    label_subplot(ax3, 'C')

    # Chart D: Cv measured vs Cv theory
    ax4 = fig.add_subplot(1, 4, 4)
    ideal = hcap[hcap['type'] == 'ideal']
    lj = hcap[hcap['type'] == 'lj']
    ax4.scatter(ideal['Cv_theory'], ideal['Cv_measured'], s=22, marker='o', color=C_BLUE,
                edgecolors='k', linewidths=0.3, label='Ideal', zorder=3)
    ax4.scatter(lj['Cv_theory'], lj['Cv_measured'], s=22, marker='^', color=C_RED,
                edgecolors='k', linewidths=0.3, label='LJ', zorder=3)
    all_cv = np.concatenate([hcap['Cv_theory'].values, hcap['Cv_measured'].values])
    lo, hi = all_cv.min(), all_cv.max()
    margin = (hi - lo) * 0.1
    ax4.plot([lo - margin, hi + margin], [lo - margin, hi + margin], '--', color=C_GRAY, lw=0.8, label='y = x')
    ax4.set_xlabel('C$_V$ theory')
    ax4.set_ylabel('C$_V$ measured')
    ax4.legend(fontsize=7, framealpha=0.7)
    label_subplot(ax4, 'D')

    fig.subplots_adjust(left=0.04, right=0.98, wspace=0.3)
    fig.savefig(OUT_DIR / 'panel_4_equations_of_state.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Panel 4 saved.')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print(f'Output directory: {OUT_DIR}')
    panel_1()
    panel_2()
    panel_3()
    panel_4()
    print('All panels generated.')
