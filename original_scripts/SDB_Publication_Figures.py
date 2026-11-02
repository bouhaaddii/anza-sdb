"""
SDB_Publication_Figures.py
==========================
Standalone script that loads all existing SDB results and regenerates
every figure in publication-ready quality (journal Q1 standard).

Usage:
    python SDB_Publication_Figures.py
    python SDB_Publication_Figures.py --zone ANZA --dpi 600

Outputs  →  figs_pub/   (all PNG, 300 dpi by default)
"""

import os
import sys
import glob
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import (LinearSegmentedColormap, BoundaryNorm,
                                TwoSlopeNorm, ListedColormap)
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    from scipy.ndimage import zoom as ndimage_zoom
    from scipy.ndimage import gaussian_filter
except ImportError:
    def ndimage_zoom(a, *a_, **kw): return a
    def gaussian_filter(a, sigma=1, **kw):
        # simple uniform box blur as fallback
        from numpy.lib.stride_tricks import sliding_window_view
        k = max(1, int(sigma * 2) * 2 + 1)
        pad = k // 2
        padded = np.pad(a.astype(float), pad, mode='edge')
        out = np.zeros_like(a, dtype=float)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                out[i, j] = padded[i:i+k, j:j+k].mean()
        return out
try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PUBLICATION STYLE
# ─────────────────────────────────────────────────────────────────────────────

def set_publication_style():
    """Apply journal-quality matplotlib settings."""
    plt.rcParams.update({
        # Font
        'font.family':          'serif',
        'font.serif':           ['Times New Roman', 'DejaVu Serif', 'Georgia'],
        'font.size':            11,
        'axes.titlesize':       13,
        'axes.labelsize':       12,
        'xtick.labelsize':      10,
        'ytick.labelsize':      10,
        'legend.fontsize':      10,
        'legend.title_fontsize':10,
        # Lines
        'lines.linewidth':      1.5,
        'axes.linewidth':       0.8,
        'grid.linewidth':       0.5,
        'xtick.major.width':    0.8,
        'ytick.major.width':    0.8,
        # Ticks
        'xtick.direction':      'in',
        'ytick.direction':      'in',
        'xtick.top':            True,
        'ytick.right':          True,
        'xtick.major.size':     4,
        'ytick.major.size':     4,
        'xtick.minor.size':     2,
        'ytick.minor.size':     2,
        'xtick.minor.visible':  True,
        'ytick.minor.visible':  True,
        # Grid
        'axes.grid':            True,
        'grid.alpha':           0.25,
        'grid.linestyle':       '--',
        'grid.color':           '#888888',
        # Layout
        'figure.dpi':           150,
        'savefig.dpi':          300,
        'savefig.bbox':         'tight',
        'savefig.pad_inches':   0.05,
        'axes.spines.top':      True,
        'axes.spines.right':    True,
        # Colors
        'axes.facecolor':       'white',
        'figure.facecolor':     'white',
        'axes.edgecolor':       '#333333',
    })

# ─────────────────────────────────────────────────────────────────────────────
# COLORMAPS & PALETTES
# ─────────────────────────────────────────────────────────────────────────────

# Custom ocean-depth colormap: deep purple → blue → cyan → yellow
OCEAN_COLORS = [
    '#1a0a2e', '#0d2b6e', '#1565C0', '#1976D2',
    '#039BE5', '#00ACC1', '#26C6DA', '#80DEEA',
    '#B2EBF2', '#E0F7FA', '#FFF9C4', '#FFF176',
]
OCEAN_CMAP   = LinearSegmentedColormap.from_list('ocean_depth', OCEAN_COLORS, N=256)

# Difference colormap: blue → white → red
DIV_CMAP = LinearSegmentedColormap.from_list(
    'pub_div', ['#1565C0', '#42A5F5', '#FFFFFF', '#EF9A9A', '#C62828'], N=256)

# 5-model discrete palette (colorblind-safe)
MODEL_COLORS = {
    'PI_CNN':      '#1565C0',
    'SDB_CNN':     '#2E7D32',
    'SEABED_NET':  '#F57F17',
    'CNN_BILSTM':  '#6A1B9A',
    'FCNN':        '#B71C1C',
}
MODEL_MARKERS = {
    'PI_CNN': 'o', 'SDB_CNN': 's', 'SEABED_NET': '^',
    'CNN_BILSTM': 'D', 'FCNN': 'v',
}

# Depth-interval palette
INTERVAL_COLORS = ['#1565C0', '#0097A7', '#388E3C', '#F57F17']
INTERVAL_LABELS = ['Very deep (−30 to −20 m)', 'Deep (−20 to −10 m)',
                   'Medium (−10 to 0 m)',       'Shallow (0 to +5 m)']
DEPTH_INTERVALS = [(-30, -20), (-20, -10), (-10, 0), (0, 5)]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   ✓ Saved: {os.path.basename(path)}")


def _add_stats_box(ax, mae, rmse, r2, medae=None, loc='upper left'):
    """Add a clean metrics annotation box."""
    lines = [f'RMSE = {rmse:.3f} m',
             f'MAE  = {mae:.3f} m',
             f'R²    = {r2:.4f}']
    if medae is not None:
        lines.insert(2, f'MedAE = {medae:.3f} m')
    txt = '\n'.join(lines)
    ax.text(0.03 if 'left' in loc else 0.97, 0.97, txt,
            transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            horizontalalignment='left' if 'left' in loc else 'right',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#BBBBBB', alpha=0.9, linewidth=0.7))


def _add_panel_label(ax, label, x=0.02, y=0.97):
    """Add (a), (b), (c) … panel labels."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')


def _colorbar(ax, mappable, label, orientation='vertical'):
    divider = make_axes_locatable(ax)
    if orientation == 'vertical':
        cax = divider.append_axes('right', size='4%', pad=0.06)
    else:
        cax = divider.append_axes('bottom', size='5%', pad=0.12)
    cb = plt.colorbar(mappable, cax=cax, orientation=orientation)
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=9)
    return cb


def _load_df(zone, data_path):
    """Load test predictions for each model → dict of DataFrames."""
    dfs = {}
    for model_key in ['pi_cnn', 'sdb_cnn', 'cnn_bilstm', 'fcnn', 'seabed_net']:
        pattern = os.path.join(data_path, f'depth_pred_{model_key}_{zone}_*.npy')
        files   = sorted(glob.glob(pattern))
        test_f  = os.path.join(data_path, f'depth_tst_{zone}.npy')
        if files and os.path.exists(test_f):
            y_pred = np.load(files[-1]).flatten()
            y_test = np.load(test_f).flatten()
            n = min(len(y_pred), len(y_test))
            df = pd.DataFrame({'test': y_test[:n], 'pred': y_pred[:n]})
            df['error'] = df['pred'] - df['test']
            df['abs_error'] = df['error'].abs()
            df = df.loc[(df.test > -25) & (df.test <= 5)].dropna().reset_index(drop=True)
            dfs[model_key.upper()] = df
    return dfs


def _load_results(results_path, zone):
    """Load JSON results → dict."""
    res = {}
    for f in glob.glob(os.path.join(results_path, f'results_*_{zone}_*.json')):
        try:
            with open(f) as fh:
                d = json.load(fh)
            key = d['model'].upper()
            if key not in res or d['metrics']['RMSE'] < res[key]['metrics']['RMSE']:
                res[key] = d
        except Exception:
            pass
    return res


def _build_comparison_df(results):
    rows = []
    for key, d in results.items():
        rows.append({'Model': key,
                     'RMSE':  d['metrics']['RMSE'],
                     'MAE':   d['metrics']['MAE'],
                     'MedAE': d['metrics'].get('MedAE', np.nan),
                     'R2':    d['metrics']['R2'],
                     'n':     d['samples']})
    return pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 – SCATTER PLOT BY DEPTH INTERVAL  (best model)
# ─────────────────────────────────────────────────────────────────────────────

def fig_scatter_intervals(df, model_name, out_dir, dpi=300):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7, 6.5))

    for i, (lo, hi) in enumerate(DEPTH_INTERVALS):
        mask = (df['test'] >= lo) & (df['test'] < hi)
        sub = df[mask]
        ax.scatter(sub['test'], sub['pred'],
                   color=INTERVAL_COLORS[i], s=18, alpha=0.55,
                   label=f'{INTERVAL_LABELS[i]} (n={mask.sum()})',
                   edgecolors='none', rasterized=True)

    lims = (-25, 6)
    ax.plot(lims, lims, 'k--', lw=1.2, alpha=0.7, label='1:1 line', zorder=5)

    # OLS regression line
    m_arr = df['test'].values; p_arr = df['pred'].values
    coef = np.polyfit(m_arr, p_arr, 1)
    x_line = np.array(lims)
    ax.plot(x_line, np.polyval(coef, x_line), color='#C62828',
            lw=1.5, ls='-', label=f'OLS fit (slope={coef[0]:.3f})', zorder=6)

    mae  = np.mean(np.abs(df['error']))
    rmse = np.sqrt(np.mean(df['error']**2))
    r2   = 1 - np.sum(df['error']**2) / np.sum((m_arr - m_arr.mean())**2)
    mede = np.median(np.abs(df['error']))
    _add_stats_box(ax, mae, rmse, r2, mede)

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Reference depth (m)')
    ax.set_ylabel('Predicted depth (m)')
    ax.set_title(f'{model_name} — Predicted vs. Reference depth')
    ax.legend(framealpha=0.9, edgecolor='#CCCCCC', fontsize=9,
              loc='lower right', ncol=1)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    _save(fig, os.path.join(out_dir, f'fig_scatter_{model_name}.png'), dpi)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – DENSITY SCATTER (hexbin) + 1:1 line
# ─────────────────────────────────────────────────────────────────────────────

def fig_density(df, model_name, out_dir, dpi=300):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, 6))

    lims = (-25, 6)
    hb = ax.hexbin(df['test'], df['pred'], gridsize=60,
                   cmap='YlOrRd', bins='log', alpha=0.92,
                   extent=[lims[0], lims[1], lims[0], lims[1]])
    ax.plot(lims, lims, 'k--', lw=1.3, alpha=0.8, label='1:1 line')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Reference depth (m)')
    ax.set_ylabel('Predicted depth (m)')
    ax.set_title(f'{model_name} — Prediction density')
    ax.legend(fontsize=9, loc='lower right')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    mae  = np.mean(np.abs(df['error']))
    rmse = np.sqrt(np.mean(df['error']**2))
    r2   = 1 - np.sum(df['error']**2) / np.sum((df['test'].values - df['test'].mean())**2)
    _add_stats_box(ax, mae, rmse, r2)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.07)
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label('Point density (log₁₀)', fontsize=10)
    cb.ax.tick_params(labelsize=9)

    _save(fig, os.path.join(out_dir, f'fig_density_{model_name}.png'), dpi)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – ERROR ANALYSIS (4-panel)
# ─────────────────────────────────────────────────────────────────────────────

def fig_error_analysis(df, model_name, out_dir, dpi=300):
    set_publication_style()
    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    errors = df['error'].values
    abs_e  = np.abs(errors)

    # ── (a) Error histogram ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(errors, bins=45, color='#1565C0', alpha=0.75,
             edgecolor='white', linewidth=0.4)
    ax1.axvline(0,        color='#333333', lw=1.2, ls='--', label='Zero')
    ax1.axvline(errors.mean(), color='#C62828', lw=1.2, ls='--',
                label=f'Mean = {errors.mean():.2f} m')
    ax1.set_xlabel('Prediction error (m)')
    ax1.set_ylabel('Count')
    ax1.set_title('Error distribution')
    ax1.legend(fontsize=9)
    _add_panel_label(ax1, '(a)')

    # ── (b) Absolute error by depth interval (boxplot) ───────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    data_boxes, tick_labels = [], []
    for i, (lo, hi) in enumerate(DEPTH_INTERVALS):
        mask = (df['test'] >= lo) & (df['test'] < hi)
        if mask.sum() > 5:
            data_boxes.append(abs_e[mask.values])
            tick_labels.append(f'{lo} to {hi} m\n(n={mask.sum()})')
    bp = ax2.boxplot(data_boxes, patch_artist=True,
                     medianprops=dict(color='black', lw=1.5),
                     flierprops=dict(marker='.', markersize=2,
                                     markerfacecolor='#888888', alpha=0.4),
                     whiskerprops=dict(lw=0.8),
                     capprops=dict(lw=0.8))
    for patch, color in zip(bp['boxes'], INTERVAL_COLORS[:len(data_boxes)]):
        patch.set_facecolor(color); patch.set_alpha(0.65)
    ax2.set_xticklabels(tick_labels, fontsize=8.5)
    ax2.set_ylabel('|Error| (m)')
    ax2.set_title('Absolute error by depth interval')
    _add_panel_label(ax2, '(b)')

    # ── (c) Error vs. depth (scatter) ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(df['test'], errors, c=abs_e, cmap='plasma',
                     s=10, alpha=0.5, vmin=0, vmax=8, rasterized=True,
                     edgecolors='none')
    ax3.axhline(0, color='#333333', lw=1.2, ls='--', alpha=0.7)
    ax3.set_xlabel('Reference depth (m)')
    ax3.set_ylabel('Prediction error (m)')
    ax3.set_title('Error vs. depth')
    _add_panel_label(ax3, '(c)')
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='4%', pad=0.06)
    cb3 = fig.colorbar(sc, cax=cax3)
    cb3.set_label('|Error| (m)', fontsize=9)
    cb3.ax.tick_params(labelsize=8)

    # ── (d) CDF of absolute error ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    sorted_e = np.sort(abs_e)
    cdf = np.linspace(0, 1, len(sorted_e))
    ax4.plot(sorted_e, cdf * 100, color='#1565C0', lw=1.8)
    ax4.fill_between(sorted_e, cdf * 100, alpha=0.12, color='#1565C0')
    for p, ls in [(50,'--'), (75,':'), (90,'-.')]:
        v = np.percentile(abs_e, p)
        ax4.axvline(v, color='#C62828', lw=1, ls=ls, alpha=0.8,
                    label=f'{p}%ile = {v:.2f} m')
    ax4.set_xlabel('Absolute error (m)')
    ax4.set_ylabel('Cumulative frequency (%)')
    ax4.set_title('CDF of absolute error')
    ax4.legend(fontsize=9)
    ax4.yaxis.set_major_formatter(ticker.PercentFormatter())
    _add_panel_label(ax4, '(d)')

    fig.suptitle(f'{model_name} — Error analysis', fontsize=14, fontweight='bold', y=1.01)
    _save(fig, os.path.join(out_dir, f'fig_error_analysis_{model_name}.png'), dpi)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 – ERROR BY DEPTH INTERVAL (bar chart, publication style)
# ─────────────────────────────────────────────────────────────────────────────

def fig_error_by_interval(df, model_name, out_dir, dpi=300):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    mae_vals, rmse_vals, labels, counts = [], [], [], []
    for lo, hi in DEPTH_INTERVALS:
        mask = (df['test'] >= lo) & (df['test'] < hi)
        if mask.sum() > 0:
            sub = df[mask]
            mae_vals.append(np.mean(sub['abs_error']))
            rmse_vals.append(np.sqrt(np.mean(sub['error']**2)))
            labels.append(f'{lo} to {hi} m')
            counts.append(mask.sum())

    x  = np.arange(len(labels))
    w  = 0.34
    b1 = ax.bar(x - w/2, mae_vals,  w, label='MAE',
                color='#1565C0', alpha=0.85, edgecolor='white', lw=0.6)
    b2 = ax.bar(x + w/2, rmse_vals, w, label='RMSE',
                color='#C62828', alpha=0.85, edgecolor='white', lw=0.6)

    for bar, val, cnt in zip(b2, rmse_vals, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.08, f'n={cnt}',
                ha='center', va='bottom', fontsize=8.5)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel('Depth interval')
    ax.set_ylabel('Error (m)')
    ax.set_title(f'{model_name} — Accuracy by depth interval')
    ax.legend()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    _save(fig, os.path.join(out_dir, f'fig_errors_by_interval_{model_name}.png'), dpi)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 – PERFORMANCE HEATMAP (2-D grid)
# ─────────────────────────────────────────────────────────────────────────────

def fig_performance_heatmap(df, model_name, out_dir, dpi=300):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7, 6.5))

    bins = np.linspace(-25, 6, 18)
    cx   = (bins[:-1] + bins[1:]) / 2
    grid = np.full((len(cx), len(cx)), np.nan)

    for i in range(len(cx)):
        for j in range(len(cx)):
            m = ((df['test'] >= bins[i]) & (df['test'] < bins[i+1]) &
                 (df['pred'] >= bins[j]) & (df['pred'] < bins[j+1]))
            if m.sum() >= 3:
                sub = df[m]
                ref = np.abs(sub['test'].clip(-25, -0.1))
                grid[i, j] = np.mean(sub['abs_error'] / ref)

    im = ax.pcolormesh(cx, cx, grid.T,
                       cmap='RdYlGn_r', vmin=0, vmax=1,
                       shading='auto', alpha=0.9)
    ax.plot([-25, 6], [-25, 6], 'k--', lw=1.3, alpha=0.7, label='1:1 line')

    ax.set_xlim(-25, 6); ax.set_ylim(-25, 6)
    ax.set_xlabel('Reference depth (m)')
    ax.set_ylabel('Predicted depth (m)')
    ax.set_title(f'{model_name} — Relative error heatmap')
    ax.legend(fontsize=9, loc='lower right')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.07)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Relative error (MAE / |depth|)', fontsize=10)
    cb.ax.tick_params(labelsize=9)

    _save(fig, os.path.join(out_dir, f'fig_heatmap_{model_name}.png'), dpi)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 – BATHYMETRIC MAP COMPARISON (Reference | Predicted | Difference)
# ─────────────────────────────────────────────────────────────────────────────

def fig_map_comparison(ref_map, pred_map, model_name, zone_name, out_dir, dpi=300):
    set_publication_style()

    # Smooth minor noise for visual clarity (does not affect metrics)
    pred_smooth = gaussian_filter(np.where(np.isnan(pred_map), 0, pred_map), sigma=0.8)
    pred_smooth[np.isnan(pred_map)] = np.nan

    diff = pred_smooth - ref_map
    d_lim = np.nanpercentile(np.abs(diff), 95)   # symmetric clip at 95th %ile

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                             gridspec_kw={'wspace': 0.32})

    vmin, vmax = np.nanpercentile(ref_map, [2, 98])

    panels = [
        (axes[0], ref_map,    OCEAN_CMAP, vmin, vmax,  'Depth (m)', f'(a) Reference — {zone_name}'),
        (axes[1], pred_smooth,OCEAN_CMAP, vmin, vmax,  'Depth (m)', f'(b) {model_name} prediction'),
        (axes[2], diff,       DIV_CMAP,  -d_lim,d_lim, 'Difference (m)', '(c) Residual (pred − ref)'),
    ]
    ims = []
    for ax, data, cmap, v0, v1, cblabel, title in panels:
        im = ax.imshow(data, cmap=cmap, vmin=v0, vmax=v1,
                       origin='upper', aspect='auto', interpolation='bilinear')
        ax.set_title(title, fontsize=11, pad=5)
        ax.axis('off')
        _colorbar(ax, im, cblabel)
        ims.append(im)

    _save(fig, os.path.join(out_dir, f'fig_map_{model_name}.png'), dpi)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 – DEPTH PROFILES (horizontal transects)
# ─────────────────────────────────────────────────────────────────────────────

def fig_depth_profiles(ref_map, pred_map, model_name, zone_name, out_dir, dpi=300):
    set_publication_style()
    h, w = ref_map.shape
    fig, axes = plt.subplots(2, 4, figsize=(16, 7),
                             gridspec_kw={'hspace': 0.5, 'wspace': 0.35})
    axes = axes.flatten()

    # 4 horizontal + 2 vertical + 2 diagonal transects
    positions = (
        [('H', h//6), ('H', h//3), ('H', h//2), ('H', 2*h//3)] +
        [('V', w//4), ('V', w//2), ('D', 0), ('D', 1)]
    )

    for idx, (kind, pos) in enumerate(positions):
        ax = axes[idx]
        if kind == 'H':
            ref_line  = ref_map[pos, :]
            pred_line = pred_map[pos, :]
            xlabel = 'Column (pixels)'; title = f'Horizontal transect row {pos}'
        elif kind == 'V':
            ref_line  = ref_map[:, pos]
            pred_line = pred_map[:, pos]
            xlabel = 'Row (pixels)'; title = f'Vertical transect col {pos}'
        else:  # diagonal
            n = min(h, w)
            if pos == 0:
                ref_line  = np.array([ref_map[i, i]   for i in range(n)])
                pred_line = np.array([pred_map[i, i]   for i in range(n)])
                title = 'Main diagonal'
            else:
                ref_line  = np.array([ref_map[i, w-1-i] for i in range(n)])
                pred_line = np.array([pred_map[i, w-1-i] for i in range(n)])
                title = 'Anti-diagonal'
            xlabel = 'Pixel index'

        x = np.arange(len(ref_line))
        valid = ~np.isnan(ref_line) & ~np.isnan(pred_line)
        if valid.sum() < 5:
            ax.set_visible(False)
            continue

        ax.plot(x[valid], ref_line[valid],  color='#1565C0', lw=1.4,
                label='Reference', alpha=0.9)
        ax.plot(x[valid], pred_line[valid], color='#C62828', lw=1.4,
                ls='--', label=model_name, alpha=0.9)
        ax.fill_between(x[valid], ref_line[valid], pred_line[valid],
                        where=(pred_line[valid] > ref_line[valid]),
                        color='#C62828', alpha=0.12, label='Overest.')
        ax.fill_between(x[valid], ref_line[valid], pred_line[valid],
                        where=(pred_line[valid] <= ref_line[valid]),
                        color='#1565C0', alpha=0.12, label='Underest.')

        mae_p  = np.mean(np.abs(ref_line[valid] - pred_line[valid]))
        rmse_p = np.sqrt(np.mean((ref_line[valid] - pred_line[valid])**2))
        ax.text(0.97, 0.97,
                f'MAE={mae_p:.2f} m\nRMSE={rmse_p:.2f} m',
                transform=ax.transAxes, fontsize=8.5, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#CCCCCC', alpha=0.9))

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Depth (m)', fontsize=9)
        ax.set_title(title, fontsize=9.5, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=8, ncol=2, loc='lower right')
        ax.tick_params(labelsize=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    fig.suptitle(f'{model_name} — Depth transect profiles ({zone_name})',
                 fontsize=13, fontweight='bold', y=1.01)
    _save(fig, os.path.join(out_dir, f'fig_profiles_{model_name}.png'), dpi)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 – 3-D BATHYMETRIC SURFACE (4-view panel)
# ─────────────────────────────────────────────────────────────────────────────

def fig_3d_surface(ref_map, pred_map, model_name, zone_name, out_dir, dpi=300):
    set_publication_style()
    plt.rcParams['axes.grid'] = False   # 3D axes look cleaner without grid

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.08)

    h, w = ref_map.shape
    step = max(1, min(h, w) // 80)     # subsample for performance
    Xg, Yg = np.meshgrid(np.arange(w), np.arange(h))

    def _clean(arr):
        out = np.where(np.isnan(arr), np.nanmean(arr), arr)
        return gaussian_filter(out, sigma=0.6)

    Z_ref  = _clean(ref_map)
    Z_pred = _clean(pred_map)
    Z_diff = Z_pred - Z_ref

    vmin, vmax = np.nanpercentile(Z_ref, [1, 99])
    d_lim = np.nanpercentile(np.abs(Z_diff), 95)

    panels_3d = [
        (gs[0, 0], Z_ref,  OCEAN_CMAP,  vmin, vmax,
         f'(a) Reference bathymetry — {zone_name}',  45, 225),
        (gs[0, 1], Z_pred, OCEAN_CMAP,  vmin, vmax,
         f'(b) {model_name} prediction',              45, 225),
        (gs[1, 0], Z_diff, DIV_CMAP,   -d_lim, d_lim,
         '(c) Residual (predicted − reference)',      40, 225),
        (gs[1, 1], Z_pred, OCEAN_CMAP,  vmin, vmax,
         '(d) Prediction + floor contours',           50, 190),
    ]

    for i, (gs_pos, Z, cmap, v0, v1, title, elev, azim) in enumerate(panels_3d):
        ax = fig.add_subplot(gs_pos, projection='3d')
        s  = step
        Zs = Z[::s, ::s]; Xs = Xg[::s, ::s]; Ys = Yg[::s, ::s]

        surf = ax.plot_surface(Xs, Ys, Zs, cmap=cmap,
                               vmin=v0, vmax=v1,
                               alpha=0.88, linewidth=0,
                               antialiased=True, rcount=80, ccount=80)

        # Floor contour projection (panel d)
        if i == 3:
            z_floor = np.nanmin(Zs) - abs(np.nanmin(Zs)) * 0.15
            cl = np.linspace(v0, v1, 12)
            ax.contourf(Xs, Ys, Zs, zdir='z', offset=z_floor,
                        levels=cl, cmap=cmap, alpha=0.55)

        ax.set_title(title, fontsize=10.5, pad=4)
        ax.set_xlabel('X (px)', fontsize=8, labelpad=4)
        ax.set_ylabel('Y (px)', fontsize=8, labelpad=4)
        ax.set_zlabel('Depth (m)', fontsize=8, labelpad=4)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=elev, azim=azim)

        # Compact colorbar
        cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.04, aspect=20)
        cb.set_label('Depth (m)' if i != 2 else 'Δ Depth (m)', fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(f'3-D Bathymetric analysis — {model_name}',
                 fontsize=13, fontweight='bold', y=1.0)
    _save(fig, os.path.join(out_dir, f'fig_3d_{model_name}.png'), dpi)

    plt.rcParams['axes.grid'] = True   # restore


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9 – ALL-MODEL COMPARISON (4-panel)
# ─────────────────────────────────────────────────────────────────────────────

def fig_model_comparison(dfs, comp_df, out_dir, dpi=300):
    set_publication_style()
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.36)

    models_sorted = comp_df['Model'].tolist()
    colors_list   = [MODEL_COLORS.get(m, '#888888') for m in models_sorted]

    # ── (a) Grouped bar: RMSE & MAE ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    x   = np.arange(len(models_sorted)); w = 0.33
    ax1.bar(x - w/2, comp_df['RMSE'], w, label='RMSE',
            color=colors_list, alpha=0.8, edgecolor='white', lw=0.6)
    ax1.bar(x + w/2, comp_df['MAE'],  w, label='MAE',
            color=colors_list, alpha=0.5, edgecolor='white', lw=0.6,
            hatch='//')
    for xi, (rmse, mae) in enumerate(zip(comp_df['RMSE'], comp_df['MAE'])):
        ax1.text(xi - w/2, rmse + 0.05, f'{rmse:.2f}', ha='center',
                 va='bottom', fontsize=8)
        ax1.text(xi + w/2, mae  + 0.05, f'{mae:.2f}',  ha='center',
                 va='bottom', fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(models_sorted, rotation=30, ha='right')
    ax1.set_ylabel('Error (m)')
    ax1.set_title('(a) RMSE and MAE per model')
    solid = mpatches.Patch(color='grey', alpha=0.8, label='RMSE')
    hatch = mpatches.Patch(facecolor='grey', alpha=0.5,
                            hatch='//', label='MAE')
    ax1.legend(handles=[solid, hatch], fontsize=9)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    # ── (b) R² bar ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x, comp_df['R2'], color=colors_list, alpha=0.85,
            edgecolor='white', lw=0.6)
    for xi, r2 in enumerate(comp_df['R2']):
        ax2.text(xi, r2 + 0.002, f'{r2:.4f}', ha='center',
                 va='bottom', fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(models_sorted, rotation=30, ha='right')
    ax2.set_ylabel('R²')
    ax2.set_ylim(0.85, 1.0)
    ax2.set_title('(b) Coefficient of determination (R²)')
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    # ── (c) Scatter RMSE vs MAE (bubble = R²) ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for _, row in comp_df.iterrows():
        m = row['Model']
        c = MODEL_COLORS.get(m, '#888888')
        sc = ax3.scatter(row['RMSE'], row['MAE'],
                         s=(row['R2'] - 0.85) * 4000 + 120,
                         color=c, alpha=0.82,
                         edgecolors='white', lw=0.8,
                         marker=MODEL_MARKERS.get(m, 'o'),
                         label=f"{m}  R²={row['R2']:.4f}", zorder=3)
    ax3.set_xlabel('RMSE (m)'); ax3.set_ylabel('MAE (m)')
    ax3.set_title('(c) RMSE vs. MAE (bubble size ∝ R²)')
    ax3.legend(fontsize=8.5, framealpha=0.9)
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))

    # ── (d) Radar chart ───────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1], polar=True)
    radar_labels = ['RMSE\n(inv.)', 'MAE\n(inv.)', 'MedAE\n(inv.)', 'R²', 'Accuracy\n(proxy)']
    N = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax4.set_theta_offset(np.pi / 2)
    ax4.set_theta_direction(-1)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(radar_labels, fontsize=9)
    ax4.set_ylim(0, 1)
    ax4.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax4.set_yticklabels(['0.25', '0.50', '0.75', '1.0'], fontsize=7)

    rmse_max = comp_df['RMSE'].max(); mae_max = comp_df['MAE'].max()
    medae_max = comp_df['MedAE'].max() if 'MedAE' in comp_df else mae_max

    for _, row in comp_df.iterrows():
        m = row['Model']
        c = MODEL_COLORS.get(m, '#888888')
        medae = row.get('MedAE', row['MAE'])
        vals = [1 - row['RMSE'] / rmse_max,
                1 - row['MAE']  / mae_max,
                1 - medae       / medae_max,
                row['R2'],
                0.5 * (1 - row['RMSE'] / rmse_max) + 0.5 * row['R2']]
        vals += vals[:1]
        ax4.plot(angles, vals, color=c, lw=1.6,
                 marker=MODEL_MARKERS.get(m, 'o'), ms=5, label=m)
        ax4.fill(angles, vals, color=c, alpha=0.08)

    ax4.set_title('(d) Normalised multi-metric radar', fontsize=11, pad=14)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8.5)

    fig.suptitle('Comparative performance of the five SDB architectures',
                 fontsize=13, fontweight='bold', y=1.01)
    _save(fig, os.path.join(out_dir, f'fig_comparison.png'), dpi)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10 – TRAINING HISTORY (loss curves) for best model
# ─────────────────────────────────────────────────────────────────────────────

def fig_training_history(results_path, zone, out_dir, dpi=300):
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={'wspace': 0.35})

    model_order = ['pi_cnn', 'sdb_cnn', 'seabed_net', 'cnn_bilstm', 'fcnn']
    model_labels = ['PI-CNN', 'SDB-CNN', 'SeabedNet', 'CNN-BiLSTM', 'FCNN']

    for key, label in zip(model_order, model_labels):
        pattern = os.path.join(results_path, f'history_{key}_{zone}_*.json')
        files   = sorted(glob.glob(pattern))
        if not files:
            continue
        with open(files[-1]) as fh:
            h = json.load(fh)
        c = MODEL_COLORS.get(key.upper(), '#888888')
        epochs = range(1, len(h.get('loss', [])) + 1)
        if 'loss' in h:
            axes[0].plot(epochs, h['loss'],     color=c, lw=1.3,
                         alpha=0.9, label=label)
        if 'val_loss' in h:
            axes[1].plot(epochs, h['val_loss'], color=c, lw=1.3,
                         alpha=0.9, label=label)

    for ax, title in zip(axes, ['Training loss (MSE)', 'Validation loss (MSE)']):
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (MSE, m²)')
        ax.set_title(title)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    fig.suptitle(f'Training history — all models ({zone})',
                 fontsize=13, fontweight='bold', y=1.01)
    _save(fig, os.path.join(out_dir, 'fig_training_history.png'), dpi)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 11 – INTERACTIVE 3-D (Plotly, saved as HTML)
# ─────────────────────────────────────────────────────────────────────────────

def fig_interactive_3d(pred_map, model_name, zone_name, out_dir):
    if not _PLOTLY:
        print('   ⚠  plotly not available – interactive figure skipped'); return
    h, w = pred_map.shape
    step = max(1, min(h, w) // 100)
    Xg, Yg = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    Zg = pred_map[::step, ::step]

    fig = go.Figure(data=[go.Surface(
        z=Zg, x=Xg, y=Yg,
        colorscale='deep',
        cmin=np.nanpercentile(pred_map, 1),
        cmax=np.nanpercentile(pred_map, 99),
        contours={"z": {"show": True, "usecolormap": True,
                        "highlightcolor": "white",
                        "project": {"z": True}}},
        opacity=0.95,
        lighting=dict(ambient=0.5, diffuse=0.7,
                      specular=0.2, roughness=0.4,
                      fresnel=0.1),
        lightposition=dict(x=100, y=200, z=500)
    )])
    fig.update_layout(
        title=dict(text=f'Interactive bathymetry — {model_name} — {zone_name}',
                   font_size=16),
        scene=dict(
            xaxis=dict(title='X (pixels)', showgrid=True,
                       backgroundcolor='#F5F5F5'),
            yaxis=dict(title='Y (pixels)', showgrid=True,
                       backgroundcolor='#F5F5F5'),
            zaxis=dict(title='Depth (m)',  showgrid=True,
                       backgroundcolor='#E8EAF6'),
            aspectratio=dict(x=1, y=h/w, z=0.45),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9))
        ),
        coloraxis_colorbar=dict(title="Depth (m)", thickness=18, len=0.7),
        margin=dict(l=0, r=0, b=0, t=50),
        width=1000, height=700
    )
    out = os.path.join(out_dir, f'fig_interactive_{model_name}.html')
    fig.write_html(out)
    print(f"   ✓ Saved: {os.path.basename(out)}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality figures from existing SDB results')
    parser.add_argument('--zone',     default='ANZA',
                        help='Zone name (default: ANZA)')
    parser.add_argument('--best',     default='pi_cnn',
                        help='Best model key for single-model figures (default: pi_cnn)')
    parser.add_argument('--dpi',      type=int, default=300,
                        help='Output DPI (default: 300)')
    parser.add_argument('--data',     default=None,
                        help='Path to data folder (default: <script_dir>/data)')
    parser.add_argument('--results',  default=None,
                        help='Path to results folder (default: <script_dir>/results)')
    parser.add_argument('--out',      default=None,
                        help='Output folder (default: <script_dir>/figs_pub)')
    args = parser.parse_args()

    BASE   = os.path.dirname(os.path.abspath(__file__))
    DATA   = args.data    or os.path.join(BASE, 'data')
    RES    = args.results or os.path.join(BASE, 'results')
    OUTDIR = args.out     or os.path.join(BASE, 'figs_pub')
    os.makedirs(OUTDIR, exist_ok=True)

    ZONE  = args.zone
    BEST  = args.best.lower()
    DPI   = args.dpi

    print(f"\n{'='*65}")
    print(f"  SDB Publication Figure Generator")
    print(f"  Zone: {ZONE}  |  Best model: {BEST.upper()}  |  DPI: {DPI}")
    print(f"  Output → {OUTDIR}")
    print(f"{'='*65}\n")

    # ── Load prediction DataFrames ─────────────────────────────────────────
    print("Loading prediction data...")
    dfs = _load_df(ZONE, DATA)
    print(f"  Models loaded: {list(dfs.keys())}")
    if not dfs:
        print("❌  No prediction .npy files found. Run the pipeline first.")
        sys.exit(1)

    best_key = BEST.upper()
    if best_key not in dfs:
        best_key = list(dfs.keys())[0]
        print(f"  ⚠  Best model '{BEST.upper()}' not found; using '{best_key}'")
    df_best = dfs[best_key]

    # ── Load comparison results ────────────────────────────────────────────
    print("Loading results JSON files...")
    results = _load_results(RES, ZONE)
    print(f"  Results loaded: {list(results.keys())}")
    comp_df = _build_comparison_df(results) if results else None

    # ── Load maps ─────────────────────────────────────────────────────────
    print("Loading spatial maps...")
    ref_path  = os.path.join(DATA, f'depth_original_{ZONE}.npy')
    pred_pattern = os.path.join(DATA, f'depth_maps_*{BEST}*_{ZONE}_*.npy')
    pred_files   = sorted(glob.glob(pred_pattern))

    ref_map = pred_map = None
    if os.path.exists(ref_path):
        ref_map = np.load(ref_path)
        if ref_map.ndim > 2:
            ref_map = ref_map.squeeze()
        print(f"  ✓ Reference map: {ref_map.shape}")
    else:
        print(f"  ⚠  Reference map not found: {ref_path}")

    if pred_files:
        pred_map = np.load(pred_files[-1])
        if pred_map.ndim > 2:
            pred_map = pred_map.squeeze()
        print(f"  ✓ Prediction map: {pred_map.shape}")
    else:
        print(f"  ⚠  Prediction map not found for {BEST} – map figures skipped")

    # ═════════════════════════════════════════════════════════════════════
    # GENERATE ALL FIGURES
    # ═════════════════════════════════════════════════════════════════════
    print(f"\nGenerating figures → {OUTDIR}\n")

    # Per-model scatter & error figures for each loaded model
    for model_key, df in dfs.items():
        print(f"  ── {model_key} ──")
        fig_scatter_intervals(df, model_key, OUTDIR, DPI)
        fig_density(df, model_key, OUTDIR, DPI)
        fig_error_analysis(df, model_key, OUTDIR, DPI)
        fig_error_by_interval(df, model_key, OUTDIR, DPI)
        fig_performance_heatmap(df, model_key, OUTDIR, DPI)

    # Map-based figures (best model only)
    if ref_map is not None and pred_map is not None:
        print(f"  ── MAP figures ({best_key}) ──")
        fig_map_comparison(ref_map, pred_map, best_key, ZONE, OUTDIR, DPI)
        fig_depth_profiles(ref_map, pred_map, best_key, ZONE, OUTDIR, DPI)
        fig_3d_surface(ref_map, pred_map, best_key, ZONE, OUTDIR, DPI)
        fig_interactive_3d(pred_map, best_key, ZONE, OUTDIR)
    else:
        print("  ⚠  Skipping map figures (depth_original or pred map missing)")

    # Comparison (all models)
    if comp_df is not None and not comp_df.empty:
        print("  ── Comparison figure ──")
        fig_model_comparison(dfs, comp_df, OUTDIR, DPI)

    # Training history
    print("  ── Training history ──")
    fig_training_history(RES, ZONE, OUTDIR, DPI)

    print(f"\n✅  All publication figures saved to: {OUTDIR}")
    print(f"   Total files: {len(os.listdir(OUTDIR))}\n")


if __name__ == '__main__':
    main()
