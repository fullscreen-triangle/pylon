"""
Generate 4 figure panels (Panels 5-8) for Paper 2:
Backward Trajectory Completion on Gear Ratio Manifolds.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = r"c:\Users\kunda\Documents\distributed\pylon\panthera\publication\validation\trajectory_completion\results"
OUT_DIR = r"c:\Users\kunda\Documents\distributed\pylon\panthera\publication\sango-rine-shumba-trajectory\figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

# Muted professional palette
PALETTE = ['#4878A8', '#E07B54', '#5EA06E', '#C75D7E', '#8B72B0', '#C2985A', '#6CB3C9']


def _label(ax, letter, is_3d=False):
    """Add subplot label (A), (B), etc."""
    if is_3d:
        ax.text2D(0.02, 0.95, f'({letter})', transform=ax.transAxes,
                  fontsize=10, fontweight='bold', va='top')
    else:
        ax.text(0.02, 0.95, f'({letter})', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')


def _to_bool(series):
    """Convert string True/False to actual booleans."""
    if series.dtype == object:
        return series.map({'True': True, 'False': False, True: True, False: False})
    return series.astype(bool)


def _read(name):
    df = pd.read_csv(os.path.join(DATA_DIR, name))
    # Auto-convert boolean-looking columns
    for col in df.columns:
        if df[col].dtype == object and set(df[col].dropna().unique()).issubset({'True', 'False'}):
            df[col] = _to_bool(df[col])
    return df


# ===================================================================
# Panel 5: Backward Navigation Scaling
# ===================================================================
def panel_5():
    bn = _read('backward_navigation.csv')
    ps = _read('program_synthesis.csv')

    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # --- A: 3D scatter of S-space ---
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    categories = ps['category'].unique()
    for i, cat in enumerate(sorted(categories)):
        sub = ps[ps['category'] == cat]
        ax.scatter(sub['S_k'], sub['S_t'], sub['S_e'],
                   c=PALETTE[i % len(PALETTE)], label=cat, s=18, alpha=0.8, edgecolors='none')
    ax.set_xlabel('S_k')
    ax.set_ylabel('S_t')
    ax.set_zlabel('S_e')
    ax.set_title('Program S-space', fontsize=10)
    ax.legend(fontsize=5, loc='upper left', markerscale=0.7, framealpha=0.5)
    _label(ax, 'A', is_3d=True)

    # --- B: Backward comparisons vs M (log-linear) ---
    ax = fig.add_subplot(1, 4, 2)
    ax.errorbar(bn['library_size'], bn['backward_comparisons_mean'],
                yerr=bn['backward_comparisons_std'], fmt='o-', color=PALETTE[0],
                markersize=3, linewidth=1, label='Backward (mean)', capsize=2)
    ax.plot(bn['library_size'], bn['log2_M'], '--', color=PALETTE[2],
            linewidth=1, label=r'$\log_2 M$')
    ax.plot(bn['library_size'], bn['forward_comparisons'], '+-', color=PALETTE[1],
            markersize=4, linewidth=1, label='Forward (M)')
    ax.set_xscale('log')
    ax.set_xlabel('Library size M')
    ax.set_ylabel('Comparisons')
    ax.set_title('Comparisons vs M', fontsize=10)
    ax.legend(framealpha=0.5)
    _label(ax, 'B')

    # --- C: Speedup factor (log-log) ---
    ax = fig.add_subplot(1, 4, 3)
    ax.plot(bn['library_size'], bn['speedup_factor'], 'o-', color=PALETTE[3],
            markersize=3, linewidth=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Library size M')
    ax.set_ylabel(r'Speedup ($M / \log_2 M$)')
    ax.set_title('Speedup factor', fontsize=10)
    _label(ax, 'C')

    # --- D: Per-category accuracy ---
    ax = fig.add_subplot(1, 4, 4)
    cat_acc = ps.groupby('category')['correct'].mean().sort_index()
    cats = list(cat_acc.index)
    vals = list(cat_acc.values)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(cats))]
    ax.bar(range(len(cats)), vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Synthesis accuracy', fontsize=10)
    _label(ax, 'D')

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'panel_5_backward_navigation.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ===================================================================
# Panel 6: Thermodynamic Security
# ===================================================================
def panel_6():
    ts = _read('thermodynamic_security.csv')
    bz = _read('byzantine_tolerance.csv')

    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # --- A: 3D scatter of security detection ---
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    attack_types = sorted(ts['attack_type'].unique())
    for i, at in enumerate(attack_types):
        sub = ts[ts['attack_type'] == at]
        ax.scatter(sub['attacker_fraction'], sub['detection_time'],
                   sub['temperature_change_rate'],
                   c=PALETTE[i % len(PALETTE)], label=at, s=14, alpha=0.7, edgecolors='none')
    ax.set_xlabel('Attacker frac')
    ax.set_ylabel('Det. time')
    ax.set_zlabel('Temp rate')
    ax.set_title('Security detection', fontsize=10)
    ax.legend(fontsize=5, loc='upper left', markerscale=0.7, framealpha=0.5)
    _label(ax, 'A', is_3d=True)

    # --- B: Detection rate vs attacker_fraction by attack_type ---
    ax = fig.add_subplot(1, 4, 2)
    for i, at in enumerate(attack_types):
        sub = ts[ts['attack_type'] == at]
        det_rate = sub.groupby('attacker_fraction')['detected'].mean()
        ax.plot(det_rate.index, det_rate.values, 'o-', color=PALETTE[i % len(PALETTE)],
                label=at, markersize=3, linewidth=1)
    ax.set_xlabel('Attacker fraction')
    ax.set_ylabel('Detection rate')
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('Detection rate', fontsize=10)
    ax.legend(fontsize=6, framealpha=0.5)
    _label(ax, 'B')

    # --- C: Byzantine tolerance ---
    ax = fig.add_subplot(1, 4, 3)
    for i, method in enumerate(sorted(bz['method'].unique())):
        sub = bz[bz['method'] == method]
        rate = sub.groupby('faulty_fraction')['consensus_correct'].mean()
        ax.plot(rate.index, rate.values, 'o-', color=PALETTE[i],
                label=method, markersize=3, linewidth=1)
    ax.axvline(1/3, ls='--', color=PALETTE[1], alpha=0.6, linewidth=0.8, label='PBFT limit (1/3)')
    ax.axvline(0.5, ls='--', color=PALETTE[0], alpha=0.6, linewidth=0.8, label='Thermo limit (1/2)')
    ax.set_xlabel('Faulty fraction')
    ax.set_ylabel('Correct consensus')
    ax.set_title('Byzantine tolerance', fontsize=10)
    ax.legend(fontsize=6, framealpha=0.5)
    _label(ax, 'C')

    # --- D: Detection time vs attacker_fraction ---
    ax = fig.add_subplot(1, 4, 4)
    for i, at in enumerate(attack_types):
        sub = ts[ts['attack_type'] == at]
        ax.scatter(sub['attacker_fraction'], sub['detection_time'],
                   c=PALETTE[i % len(PALETTE)], label=at, s=12, alpha=0.6, edgecolors='none')
    # Overall trend line
    x = ts['attacker_fraction'].values
    y = ts['detection_time'].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 2:
        slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
        xs = np.linspace(x[mask].min(), x[mask].max(), 50)
        ax.plot(xs, slope * xs + intercept, '-', color='#333333', linewidth=1, label='Trend')
    ax.set_xlabel('Attacker fraction')
    ax.set_ylabel('Detection time')
    ax.set_title('Detection time', fontsize=10)
    ax.legend(fontsize=6, framealpha=0.5)
    _label(ax, 'D')

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'panel_6_security_byzantine.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ===================================================================
# Panel 7: Geometric and Topological Structure
# ===================================================================
def panel_7():
    fb = _read('fiber_bundle.csv')
    tb = _read('topological_protection_bands.csv')
    tf = _read('topological_protection_faults.csv')
    gi = _read('gauge_invariance.csv')

    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # --- A: 3D scatter of fiber bundle ---
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    err = fb['transitivity_error'].values
    # Use log-safe color mapping
    c_vals = np.log10(err + 1e-20)
    sc = ax.scatter(fb['R_ik_direct'], fb['R_ik_transitive'], err,
                    c=c_vals, cmap='viridis', s=14, alpha=0.7, edgecolors='none')
    ax.set_xlabel('R direct')
    ax.set_ylabel('R transitive')
    ax.set_zlabel('Trans. error')
    ax.set_title('Fiber bundle', fontsize=10)
    _label(ax, 'A', is_3d=True)

    # --- B: Topological band structure ---
    ax = fig.add_subplot(1, 4, 2)
    q = tb['q_vector'].values
    b1 = tb['omega_band1'].values
    b2 = tb['omega_band2'].values
    ax.plot(q, b1, '-', color=PALETTE[0], linewidth=1, label='Band 1')
    ax.plot(q, b2, '-', color=PALETTE[1], linewidth=1, label='Band 2')
    ax.fill_between(q, b1, b2, color=PALETTE[4], alpha=0.15, label='Gap')
    # Annotate Berry phases
    bp1 = tb['berry_phase_band1'].iloc[0]
    bp2 = tb['berry_phase_band2'].iloc[0]
    ax.text(0.98, 0.15, f'Berry: {bp1:.2f}, {bp2:.2f}',
            transform=ax.transAxes, fontsize=6, ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='grey', alpha=0.7, boxstyle='round,pad=0.2'))
    ax.set_xlabel('q')
    ax.set_ylabel('Frequency')
    ax.set_title('Band structure', fontsize=10)
    ax.legend(fontsize=6, framealpha=0.5)
    _label(ax, 'B')

    # --- C: Fault tolerance ---
    ax = fig.add_subplot(1, 4, 3)
    ax.plot(tf['fault_fraction'], tf['edge_mode_amplitude'], 'o-',
            color=PALETTE[2], markersize=3, linewidth=1)
    # Mark survival boundary
    boundary = tf[~tf['edge_mode_survives']]
    if len(boundary) > 0:
        thresh = boundary['fault_fraction'].min()
        ax.axvline(thresh, ls='--', color=PALETTE[3], linewidth=0.8,
                   label=f'Survival boundary ({thresh:.2f})')
        ax.legend(fontsize=6, framealpha=0.5)
    ax.set_xlabel('Fault fraction')
    ax.set_ylabel('Edge amplitude')
    ax.set_title('Topological protection', fontsize=10)
    _label(ax, 'C')

    # --- D: Gauge invariance ---
    ax = fig.add_subplot(1, 4, 4)
    max_change = gi[['T_relative_change', 'P_ratio_relative_change',
                     'Psi_relative_change', 'R_max_relative_change']].max(axis=1)
    # Replace zeros with a small floor for log scale
    max_change_plot = max_change.replace(0, 1e-16)
    ax.bar(range(len(gi)), max_change_plot, color=PALETTE[5], edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(gi)))
    ax.set_xticklabels([f'{l:.1f}' for l in gi['lambda']], fontsize=7)
    ax.set_xlabel('Scale factor λ')
    ax.set_ylabel('Max rel. change')
    ax.set_yscale('log')
    ax.set_title('Gauge invariance', fontsize=10)
    _label(ax, 'D')

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'panel_7_geometry_topology.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ===================================================================
# Panel 8: Godelian Residue and Entropy Production
# ===================================================================
def panel_8():
    gr = _read('godelian_residue.csv')
    ec = _read('entropy_computation.csv')
    ot = _read('operational_trichotomy.csv')
    ig = _read('information_geometry.csv')

    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # --- A: 3D scatter of Godelian residue ---
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    correct = gr[gr['correct'] == True]
    incorrect = gr[gr['correct'] == False]
    if len(correct) > 0:
        ax.scatter(correct['epsilon_osc'], correct['epsilon_cat'], correct['epsilon_par'],
                   c='#5EA06E', s=14, alpha=0.7, label='Correct', edgecolors='none')
    if len(incorrect) > 0:
        ax.scatter(incorrect['epsilon_osc'], incorrect['epsilon_cat'], incorrect['epsilon_par'],
                   c='#C75D7E', s=14, alpha=0.5, label='Incorrect', edgecolors='none')
    ax.set_xlabel('ε_osc')
    ax.set_ylabel('ε_cat')
    ax.set_zlabel('ε_par')
    ax.set_title('Gödelian residue', fontsize=10)
    ax.legend(fontsize=6, framealpha=0.5)
    _label(ax, 'A', is_3d=True)

    # --- B: Information geometry ---
    ax = fig.add_subplot(1, 4, 2)
    gd = ig['geodesic_distance'].values
    ed = ig['euclidean_distance'].values
    ax.scatter(ed, gd, c=PALETTE[0], s=8, alpha=0.4, edgecolors='none')
    lim = max(ed.max(), gd.max()) * 1.05
    ax.plot([0, lim], [0, lim], '--', color='#888888', linewidth=0.8, label='y = x')
    ax.set_xlabel('Euclidean dist.')
    ax.set_ylabel('Geodesic dist.')
    ax.set_title('Information geometry', fontsize=10)
    ax.legend(fontsize=6, framealpha=0.5)
    _label(ax, 'B')

    # --- C: Entropy vs computation rate ---
    ax = fig.add_subplot(1, 4, 3)
    mask = np.isfinite(ec['entropy_rate']) & np.isfinite(ec['computation_rate'])
    x = ec.loc[mask, 'computation_rate'].values
    y = ec.loc[mask, 'entropy_rate'].values
    c = ec.loc[mask, 'load_fraction'].values
    sc = ax.scatter(x, y, c=c, cmap='plasma', s=14, alpha=0.7, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Load frac', pad=0.01, shrink=0.8)
    # Trend line + R²
    if len(x) > 2:
        slope, intercept, r, _, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, '-', color='#333333', linewidth=1)
        ax.text(0.98, 0.05, f'R²={r**2:.3f}', transform=ax.transAxes,
                fontsize=7, ha='right',
                bbox=dict(facecolor='white', edgecolor='grey', alpha=0.7, boxstyle='round,pad=0.2'))
    ax.set_xlabel('Computation rate')
    ax.set_ylabel('Entropy rate')
    ax.set_title('Entropy–computation', fontsize=10)
    _label(ax, 'C')

    # --- D: Operational trichotomy box plot ---
    ax = fig.add_subplot(1, 4, 4)
    data_find = ot['T_finding_us'].dropna().values
    data_check = ot['T_checking_us'].dropna().values
    data_recog = ot['T_recognizing_us'].dropna().values
    bp = ax.boxplot([data_find, data_check, data_recog],
                    tick_labels=['Finding', 'Checking', 'Recognizing'],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='#333333', linewidth=1))
    for patch, color in zip(bp['boxes'], [PALETTE[0], PALETTE[2], PALETTE[3]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_yscale('log')
    ax.set_ylabel('Time (μs)')
    ax.set_title('Operational trichotomy', fontsize=10)
    _label(ax, 'D')

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'panel_8_residue_entropy.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    panel_5()
    panel_6()
    panel_7()
    panel_8()
    print('\nAll panels generated.')
