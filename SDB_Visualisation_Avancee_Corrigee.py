"""Corrected evaluation and visualization of the Anza experiment.

Derived from SDB_Visualisation_Avancee_EN.py; original file is preserved.
All seven architecture/loss/label functions below are copied verbatim.
Default: evaluate explicit saved spatial checkpoints, without training.
"""
from pathlib import Path
import argparse, ast, hashlib, json, os, sys
from types import SimpleNamespace
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_squared_error
from scipy.spatial import cKDTree
import tensorflow as tf

MODELS=['sdb_cnn','pi_cnn','cnn_bilstm','fcnn','seabed_net']
LABELS=dict(zip(MODELS,['CNN','PI-CNN','CNN-BiLSTM','FCNN','SeaBed-Net']))
ROLES=['train','val','test']
ROLE_LABELS={'train':'Training','val':'Validation','test':'Test'}
_DEPTH_INTERVALS=[(-np.inf,-20),(-20,-10),(-10,0),(0,5),(5,15),(15,np.inf)]
_INTERVAL_LABELS=['z < -20','-20 <= z < -10','-10 <= z < 0','0 <= z < 5','5 <= z < 15','z >= 15']
_INTERVAL_COLORS=['#1565C0','#0097A7','#388E3C','#F57F17','#6A1B9A','#795548']
FIG_PATH='.'
FIGURE_RECORDS=[]


def build_sdb_cnn(input_size=(9, 9, 6), dropout_rate=0.3):
    """Original SDB-CNN model."""
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(shape=input_size)
    
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_pi_cnn(input_size=(9, 9, 6), dropout_rate=0.3, lambda_phys=0.1):
    """Physics-Informed CNN with Beer-Lambert regularization - CORRECTED VERSION."""
    from tensorflow.keras import layers, models
    
    def beer_lambert_depth(rgb_patch, k=0.1):
        """Physical depth estimation."""
        blue_band = rgb_patch[..., 1]
        green_band = rgb_patch[..., 2]
        
        safe_green = tf.where(green_band > 0.001, green_band, 0.001)
        ratio = blue_band / safe_green
        
        depth_phys = -tf.math.log(ratio) / k
        depth_phys = tf.clip_by_value(depth_phys, -30, 5)
        
        return depth_phys
    
    inputs = layers.Input(shape=input_size)
    
    # CNN branch
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # CNN features
    x = layers.GlobalAveragePooling2D()(x)
    
    # Physical information
    center_patches = layers.Lambda(
        lambda x: tf.reduce_mean(x[:, 3:6, 3:6, :], axis=[1, 2]),
        output_shape=(6,)
    )(inputs)
    
    def compute_physics(center_values):
        blue_band = center_values[:, 1]
        green_band = center_values[:, 2]
        
        safe_green = tf.where(green_band > 0.001, green_band, 0.001)
        ratio = blue_band / safe_green
        
        depth_phys = -tf.math.log(ratio) / 0.1
        depth_phys = tf.clip_by_value(depth_phys, -30, 5)
        
        return tf.expand_dims(depth_phys, axis=-1)
    
    physics_depth = layers.Lambda(
        compute_physics,
        output_shape=(1,)
    )(center_patches)
    
    # Fusion CNN + physics
    x = layers.Concatenate()([x, physics_depth])
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    class PhysicalLoss(tf.keras.losses.Loss):
        def __init__(self, lambda_phys=0.1, name="physical_loss"):
            super().__init__(name=name)
            self.lambda_phys = lambda_phys
            self.mse = tf.keras.losses.MeanSquaredError()
            
        def call(self, y_true, y_pred):
            mse_loss = self.mse(y_true, y_pred)
            return mse_loss
            
        def get_config(self):
            return {"lambda_phys": self.lambda_phys}
    
    return model, PhysicalLoss(lambda_phys=lambda_phys)

def build_cnn_bilstm(input_size=(9, 9, 6), dropout_rate=0.3, lstm_units=64):
    """Hybrid CNN-BiLSTM architecture."""
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(shape=input_size)
    
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Features for BiLSTM
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, -1))(x)
    
    # BiLSTM layer
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False),
        merge_mode='concat'
    )(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_fcnn(input_size=(9, 9, 6), dropout_rate=0.3):
    """Fully Convolutional Network for dense output - CORRECTED VERSION."""
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(shape=input_size)
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 9x9 → 4x4
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 4x4 → 2x2
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder to upsample to 9x9
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), 
                               padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), 
                               padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.ZeroPadding2D(((0, 1), (0, 1)))(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Conv2D(1, (1, 1), activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_seabed_net(input_size=(9, 9, 6), dropout_rate=0.3, num_classes=4):
    """Multi-task network: bathymetry + classification."""
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(shape=input_size)
    
    # Common backbone
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Common features
    common_features = layers.GlobalAveragePooling2D()(x)
    
    # Bathymetry head
    depth_branch = layers.Dense(64, activation='relu')(common_features)
    depth_branch = layers.Dropout(dropout_rate)(depth_branch)
    depth_branch = layers.Dense(32, activation='relu')(depth_branch)
    depth_output = layers.Dense(1, name='depth')(depth_branch)
    
    # Seabed classification head
    seabed_branch = layers.Dense(64, activation='relu')(common_features)
    seabed_branch = layers.Dropout(dropout_rate)(seabed_branch)
    seabed_branch = layers.Dense(32, activation='relu')(seabed_branch)
    seabed_output = layers.Dense(num_classes, activation='softmax', name='seabed_class')(seabed_branch)
    
    model = models.Model(inputs=inputs, outputs=[depth_output, seabed_output])
    return model

def prepare_seabed_labels(depth_data, num_classes=4):
    """Prepares classification labels from depth."""
    if isinstance(depth_data, np.ndarray):
        if depth_data.ndim > 1:
            depth_values = depth_data.reshape(-1)
        else:
            depth_values = depth_data
    else:
        depth_values = np.array(depth_data).reshape(-1)
    
    seabed_labels = np.zeros((len(depth_values), num_classes), dtype=np.float32)
    
    for i, depth in enumerate(depth_values):
        if depth < 2:
            seabed_labels[i, 0] = 1.0
        elif depth < 5:
            seabed_labels[i, 1] = 1.0
        elif depth < 15:
            seabed_labels[i, 2] = 1.0
        else:
            seabed_labels[i, 3] = 1.0
    
    return seabed_labels

def create_model(model_type, input_size, **kwargs):
    """Factory to create requested model."""
    print(f"\n🏗️  Creating model: {model_type}")
    
    if model_type == 'sdb_cnn':
        model = build_sdb_cnn(input_size=input_size, dropout_rate=kwargs.get('dropout_rate', 0.3))
        loss = tf.keras.losses.MeanSquaredError()
        metrics = ['mae', 'mse']
        
    elif model_type == 'pi_cnn':
        model, custom_loss = build_pi_cnn(
            input_size=input_size,
            dropout_rate=kwargs.get('dropout_rate', 0.3),
            lambda_phys=kwargs.get('lambda_phys', 0.1)
        )
        loss = custom_loss
        metrics = ['mae', 'mse']
        
    elif model_type == 'cnn_bilstm':
        model = build_cnn_bilstm(
            input_size=input_size,
            dropout_rate=kwargs.get('dropout_rate', 0.3),
            lstm_units=kwargs.get('lstm_units', 64)
        )
        loss = tf.keras.losses.MeanSquaredError()
        metrics = ['mae', 'mse']
        
    elif model_type == 'fcnn':
        model = build_fcnn(
            input_size=input_size,
            dropout_rate=kwargs.get('dropout_rate', 0.3)
        )
        loss = tf.keras.losses.MeanSquaredError()
        metrics = ['mae', 'mse']
        
    elif model_type == 'seabed_net':
        model = build_seabed_net(
            input_size=input_size,
            dropout_rate=kwargs.get('dropout_rate', 0.3),
            num_classes=kwargs.get('num_classes', 4)
        )
        loss = {
            'depth': tf.keras.losses.MeanSquaredError(),
            'seabed_class': tf.keras.losses.CategoricalCrossentropy()
        }
        loss_weights = {
            'depth': 0.7,
            'seabed_class': 0.3
        }
        metrics = {
            'depth': ['mae', 'mse'],
            'seabed_class': ['accuracy']
        }
        
        return model, loss, loss_weights, metrics
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, loss, None, metrics

def _pub_style():
    """Apply publication-ready rcParams."""
    plt.rcParams.update({
        'font.family':         'serif',
        'font.serif':          ['Times New Roman', 'DejaVu Serif'],
        'font.size':           11,
        'axes.titlesize':      13,
        'axes.labelsize':      12,
        'xtick.labelsize':     10,
        'ytick.labelsize':     10,
        'legend.fontsize':     10,
        'lines.linewidth':     1.5,
        'axes.linewidth':      0.8,
        'grid.linewidth':      0.5,
        'xtick.direction':     'in',
        'ytick.direction':     'in',
        'xtick.top':           True,
        'ytick.right':         True,
        'xtick.major.size':    4,
        'ytick.major.size':    4,
        'xtick.minor.size':    2.5,
        'ytick.minor.size':    2.5,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'axes.grid':           True,
        'grid.alpha':          0.22,
        'grid.linestyle':      '--',
        'grid.color':          '#888888',
        'axes.facecolor':      'white',
        'figure.facecolor':    'white',
        'axes.edgecolor':      '#333333',
        'savefig.dpi':         300,
        'savefig.bbox':        'tight',
        'savefig.pad_inches':  0.05,
    })

def _cbar(ax, mappable, label):
    """Attach a neat colorbar using make_axes_locatable."""
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='4%', pad=0.06)
    cb = plt.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=9)
    return cb

def _stats_box(ax, mae, rmse, r2, medae=None, loc='upper left'):
    """Add a clean metric annotation box."""
    lines = [f'RMSE  = {rmse:.3f} nominal m', f'MAE   = {mae:.3f} nominal m']
    if medae is not None:
        lines.append(f'MedAE = {medae:.3f} nominal m')
    lines.append(f'R²     = {r2:.4f}')
    x0 = 0.03 if 'left' in loc else 0.97
    ha = 'left' if 'left' in loc else 'right'
    ax.text(x0, 0.97, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, va='top', ha=ha, fontfamily='serif',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#BBBBBB', alpha=0.92, linewidth=0.7))

def _panel(ax, label):
    """Add (a), (b) … panel letter."""
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

def create_advanced_visualizations(df_eval, args, zone_name, model_name):
    """Creates publication-quality visualizations for interpretation."""
    print(f"\n🎨 Creating advanced visualizations (publication style)...")
    _pub_style()
    checked_metrics(df_eval['test'], df_eval['pred'])
    df_eval = df_eval.copy()

    # ── 1. SCATTER PLOT WITH DEPTH INTERVALS ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for i, (lo, hi) in enumerate(_DEPTH_INTERVALS):
        mask = (df_eval['test'] >= lo) & (df_eval['test'] < hi)
        sub  = df_eval[mask]
        ax.scatter(sub['test'], sub['pred'],
                   color=_INTERVAL_COLORS[i], s=18, alpha=0.55,
                   label=f'{_INTERVAL_LABELS[i]} (n={mask.sum()})',
                   edgecolors='none', rasterized=True)

    lims = paired_limits(df_eval['test'], df_eval['pred'])
    ax.plot(lims, lims, 'k--', lw=1.2, alpha=0.7, label='1:1 line', zorder=5)
    coef = np.polyfit(df_eval['test'], df_eval['pred'], 1)
    ax.plot(np.array(lims), np.polyval(coef, np.array(lims)),
            color='#C62828', lw=1.5, label=f'OLS fit (slope={coef[0]:.3f})')
    _mae_v  = mean_absolute_error(df_eval['test'], df_eval['pred'])
    _rmse_v = np.sqrt(mean_squared_error(df_eval['test'], df_eval['pred']))
    _r2_v   = r2_score(df_eval['test'], df_eval['pred'])
    _med_v  = median_absolute_error(df_eval['test'], df_eval['pred'])
    _stats_box(ax, _mae_v, _rmse_v, _r2_v, _med_v)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Reference z (nominal m)'); ax.set_ylabel('Prediction (nominal m)')
    ax.set_title('')
    ax.legend(framealpha=0.9, edgecolor='#CCCCCC', fontsize=9, loc='lower right')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.tight_layout()
    scatter_file = os.path.join(FIG_PATH, f'scatter_intervals_{model_name}.png')
    save_figure(scatter_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Scatter plot with intervals: {scatter_file}")

    # ── 2. ERROR BAR CHART BY INTERVAL ───────────────────────────────────────
    mae_vals, rmse_vals, bar_labels, counts = [], [], [], []
    for lo, hi in _DEPTH_INTERVALS:
        mask = (df_eval['test'] >= lo) & (df_eval['test'] < hi)
        if mask.sum() > 0:
            sub = df_eval[mask]
            mae_vals.append(mean_absolute_error(sub['test'], sub['pred']))
            rmse_vals.append(np.sqrt(mean_squared_error(sub['test'], sub['pred'])))
            bar_labels.append(_INTERVAL_LABELS[_DEPTH_INTERVALS.index((lo, hi))])
            counts.append(mask.sum())

    x  = np.arange(len(bar_labels)); w = 0.34
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, mae_vals,  w, label='MAE',  color='#1565C0', alpha=0.85,
                edgecolor='white', linewidth=0.6)
    b2 = ax.bar(x + w/2, rmse_vals, w, label='RMSE', color='#C62828', alpha=0.85,
                edgecolor='white', linewidth=0.6)
    for bar, cnt in zip(b2, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.06,
                f'n={cnt}', ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(bar_labels)
    ax.set_xlabel('Signed target interval (nominal m)'); ax.set_ylabel('Error (nominal m)')
    ax.set_title('')
    ax.legend(); ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.tight_layout()
    bar_file = os.path.join(FIG_PATH, f'errors_by_interval_{model_name}.png')
    save_figure(bar_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Bar chart by interval: {bar_file}")

    # ── 3. DENSITY MATRIX (hexbin) ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 6))
    x_vals = df_eval['test'].values; y_vals = df_eval['pred'].values
    ext = [x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]
    hb = ax.hexbin(x_vals, y_vals, gridsize=55, cmap='YlOrRd', bins='log',
                   alpha=0.92, extent=ext)
    ax.plot([x_vals.min(), x_vals.max()],
            [x_vals.min(), x_vals.max()], 'k--', lw=1.3, label='1:1 line')
    _stats_box(ax, _mae_v, _rmse_v, _r2_v)
    ax.set_xlabel('Reference z (nominal m)'); ax.set_ylabel('Prediction (nominal m)')
    ax.set_title('')
    ax.legend(fontsize=9, loc='lower right')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='4%', pad=0.06)
    cb = plt.colorbar(hb, cax=cax)
    cb.set_label('Paired records per hexagon (log scale)', fontsize=10); cb.ax.tick_params(labelsize=9)
    plt.tight_layout()
    density_file = os.path.join(FIG_PATH, f'density_matrix_{model_name}.png')
    save_figure(density_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Density matrix: {density_file}")

    # ── 4. ERROR ANALYSIS (4-panel) ──────────────────────────────────────────
    errors   = (df_eval['pred'] - df_eval['test']).values
    abs_e    = np.abs(errors)
    # No ratio to z: it is singular near zero and invalidated by signed clipping.

    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    # (a) Error histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(errors, bins=45, color='#1565C0', alpha=0.75,
             edgecolor='white', linewidth=0.4)
    ax1.axvline(0, color='#333333', lw=1.2, ls='--', label='Zero bias')
    ax1.axvline(errors.mean(), color='#C62828', lw=1.2, ls='--',
                label=f'Mean = {errors.mean():.2f} m')
    ax1.set_xlabel('Prediction error (nominal m)'); ax1.set_ylabel('Count')
    ax1.set_title('Error distribution'); ax1.legend(fontsize=9)
    _panel(ax1, '(a)')

    # (b) Absolute error boxplot by depth interval
    ax2  = fig.add_subplot(gs[0, 1])
    bxdata, bxlabels, bxcolors = [], [], []
    for i, (lo, hi) in enumerate(_DEPTH_INTERVALS):
        mask = (df_eval['test'] >= lo) & (df_eval['test'] < hi)
        if mask.sum() > 0:
            bxdata.append(abs_e[mask.values]); bxcolors.append(_INTERVAL_COLORS[i])
            bxlabels.append(_INTERVAL_LABELS[i] + f'\n(n={mask.sum()})')
    bp = ax2.boxplot(bxdata, patch_artist=True,
                     medianprops=dict(color='black', lw=1.5),
                     flierprops=dict(marker='.', ms=2, markerfacecolor='#888', alpha=0.4),
                     whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for patch, c in zip(bp['boxes'], bxcolors):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    ax2.set_xticklabels(bxlabels, fontsize=8, rotation=15)
    ax2.set_ylabel('|Error| (nominal m)')
    ax2.set_title('Absolute error by signed target interval')
    _panel(ax2, '(b)')

    # (c) Error vs. depth
    ax3 = fig.add_subplot(gs[1, 0])
    sc3 = ax3.scatter(df_eval['test'], errors, c=abs_e, cmap='plasma',
                      s=10, alpha=0.5, vmin=0, vmax=max(float(abs_e.max()), 1e-6),
                      edgecolors='none', rasterized=True)
    ax3.axhline(0, color='#333333', lw=1.2, ls='--', alpha=0.7)
    ax3.set_xlabel('Reference z (nominal m)'); ax3.set_ylabel('Prediction error (nominal m)')
    ax3.set_title('Error vs. signed target')
    _panel(ax3, '(c)')
    _cbar(ax3, sc3, '|Error| (nominal m)')

    # (d) CDF of absolute error
    ax4 = fig.add_subplot(gs[1, 1])
    sorted_e = np.sort(abs_e)
    cdf_vals = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
    ax4.step(np.r_[0, sorted_e], np.r_[0, cdf_vals * 100], where='post', color='#1565C0', lw=1.8)
    ax4.set_ylim(0, 100)
    for p, ls in [(50,'--'), (75,':'), (90,'-.')]:
        v = np.percentile(abs_e, p)
        ax4.axvline(v, color='#C62828', lw=1, ls=ls, alpha=0.8,
                    label=f'{p}%ile = {v:.2f} m')
    ax4.set_xlabel('Absolute error (nominal m)'); ax4.set_ylabel('Cumulative frequency (%)')
    ax4.set_title('CDF of absolute error'); ax4.legend(fontsize=9)
    ax4.yaxis.set_major_formatter(mticker.PercentFormatter())
    _panel(ax4, '(d)')

    # Overall title belongs in the figure caption.
    error_analysis_file = os.path.join(FIG_PATH, f'error_analysis_{model_name}.png')
    save_figure(error_analysis_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Error analysis: {error_analysis_file}")

    # ── 5. PERFORMANCE HEATMAP ───────────────────────────────────────────────
    bins = np.linspace(*lims, 18)
    cx    = (bins[:-1] + bins[1:]) / 2
    grid  = np.full((len(cx), len(cx)), np.nan)
    for i in range(len(cx)):
        for j in range(len(cx)):
            m = ((df_eval['test'] >= bins[i]) & (df_eval['test'] < bins[i+1]) &
                 (df_eval['pred'] >= bins[j]) & (df_eval['pred'] < bins[j+1]))
            if m.sum() >= 3:
                sub = df_eval[m]
                grid[i, j] = np.mean(np.abs(sub['pred'] - sub['test']))

    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.pcolormesh(bins, bins, grid.T, cmap='magma', vmin=0,
                       shading='flat', alpha=0.9)
    ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], 'k--', lw=1.3, label='1:1 line')
    ax.set_xlabel('Reference z (nominal m)'); ax.set_ylabel('Prediction (nominal m)')
    ax.set_title('Cells with at least 3 paired records')
    ax.legend(fontsize=9, loc='lower right')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='4%', pad=0.06)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label('Mean absolute error (nominal m)', fontsize=10)
    cb.ax.tick_params(labelsize=9)
    plt.tight_layout()
    heatmap_file = os.path.join(FIG_PATH, f'performance_heatmap_{model_name}.png')
    save_figure(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Performance heatmap: {heatmap_file}")

    return True

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def checked_metrics(reference, prediction):
    y=np.asarray(reference,dtype='float64').reshape(-1)
    p=np.asarray(prediction,dtype='float64').reshape(-1)
    if y.size==0 or y.shape!=p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Expected nonempty, finite, paired arrays; no silent exclusions.')
    e=p-y;den=float(np.sum((y-y.mean())**2))
    return {'n':len(y),'rmse':float(np.sqrt(np.mean(e**2))),
            'mae':float(np.mean(abs(e))),'medae':float(np.median(abs(e))),
            'bias':float(e.mean()),'r2':float(1-np.sum(e**2)/den) if den>0 else None,
            'p95_abs_error':float(np.quantile(abs(e),.95))}

def paired_limits(y,p):
    lo=min(np.min(y),np.min(p));hi=max(np.max(y),np.max(p))
    pad=max(float(hi-lo)*.04,1.)
    return float(lo-pad),float(hi+pad)

def validate_split(coords, split, n):
    if set(split)!=set(ROLES):raise ValueError('Expected train, val and test indices.')
    for role,ids in split.items():
        if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or len(ids)==0:
            raise ValueError('Each partition must have a nonempty vector of integer patch IDs.')
        if np.min(ids)<0 or np.max(ids)>=n or len(np.unique(ids))!=len(ids):
            raise ValueError('Invalid or duplicate patch IDs in '+role)
    report={}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if np.intersect1d(split[a],split[b]).size:raise ValueError('Shared patch IDs.')
        distance=cKDTree(coords[split[a]]).query(coords[split[b]],p=np.inf)[0]
        shared=int((distance<9).sum())
        if shared:raise ValueError('Partitions share input pixels: '+a+'/'+b)
        report[a+'_'+b]={'min_top_left_chebyshev_distance':float(distance.min()),'shared_pixel_patches':shared}
    report['counts']={k:len(v) for k,v in split.items()}
    report['excluded']=int(n-sum(len(v) for v in split.values()))
    return report

def verify_scientific_functions(original):
    def nodes(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef)}
    before=nodes(original);after=nodes(__file__)
    names=['build_sdb_cnn','build_pi_cnn','build_cnn_bilstm','build_fcnn','build_seabed_net','prepare_seabed_labels','create_model']
    result={name:before[name]==after[name] for name in names}
    if not all(result.values()):raise ValueError('Scientific function changed: '+str(result))
    return result

def save_figure(path, **kwargs):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig=plt.gcf()
    # Folder names are metadata only. Disable grids/minor ticks for readable print.
    for ax in fig.axes:ax.grid(False);ax.minorticks_off()
    for text in fig.findobj(matplotlib.text.Text):
        if any(word in text.get_text().split() for word in ['EN','Original','original']):
            raise ValueError('A folder/source label entered a figure.')
    for ext in ['png','pdf','svg']:
        fig.savefig(path.with_suffix('.'+ext),dpi=350,bbox_inches='tight',facecolor='white')
    FIGURE_RECORDS.append({'stem':str(path.with_suffix('')),'png_sha256':sha(path.with_suffix('.png'))})

def interval_metrics(frame):
    result=[]
    for label,(lo,hi) in zip(_INTERVAL_LABELS,_DEPTH_INTERVALS):
        sub=frame[(frame.reference>=lo)&(frame.reference<hi)]
        if len(sub):result.append({'interval':label,**checked_metrics(sub.reference,sub.prediction)})
    if sum(r['n'] for r in result)!=len(frame):raise AssertionError('Unbinned data.')
    return result

def depth_prediction(model, X, name, batch_size):
    p=model.predict(X,batch_size=batch_size,verbose=0)
    classes=None
    if name=='seabed_net':p,classes=p
    if name=='fcnn':p=p[:,4,4,0]
    return np.asarray(p).reshape(-1),classes

def training_curves(histories, out):
    """Actual recorded objectives; selected epoch uses validation loss alone."""
    _pub_style();fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,name in zip(axes.flat,MODELS):
        h=histories[name];epochs=np.arange(1,len(h['loss'])+1)
        ax.plot(epochs,h['loss'],label='Training',color='#1565C0')
        ax.plot(epochs,h['val_loss'],label='Validation',color='#D55E00')
        best=int(np.argmin(h['val_loss'])+1)
        ax.axvline(best,ls=':',color='#555555',label=f'Selected epoch {best}')
        ax.set(xlabel='Epoch',ylabel='Weighted objective' if name=='seabed_net' else 'MSE objective (nominal m²)',title=chr(40)+chr(97+MODELS.index(name))+chr(41)+chr(32)+LABELS[name])
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,.9,'One common spatial partition\nCheckpoint selection: validation loss\n\nSeaBed-Net combines two losses.\nFCNN supervises a repeated centre label.\n\nTraining curves include dropout; paired\nmetrics use inference at the selected epoch.',va='top',fontsize=11,transform=axes.flat[-1].transAxes)
    fig.tight_layout();save_figure(Path(out)/'training_validation_objectives.png');plt.close(fig)

def role_comparison(rows,out):
    _pub_style();df=pd.DataFrame(rows);fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,key,title in zip(axes.flat,['rmse','mae','bias','r2'],['RMSE (nominal m)','MAE (nominal m)','Bias (nominal m)','R²']):
        for j,role in enumerate(ROLES):
            vals=df[df.role==role].set_index('model').reindex(MODELS)[key]
            ax.bar(np.arange(5)+(j-1)*.24,vals,width=.24,label=ROLE_LABELS[role],color=['#4477AA','#EEAA33','#228833'][j])
        ax.axhline(0,color='#555555',lw=.7)
        ax.set(xticks=np.arange(5),xticklabels=list(LABELS.values()),ylabel=title)
        ax.tick_params(axis='x',rotation=18);ax.legend(fontsize=9)
    fig.tight_layout();save_figure(Path(out)/'comparison_training_validation_test.png');plt.close(fig)

def paired_panels(frames,role,out):
    """A compact view of primary-script interval scatters for all five models."""
    _pub_style();fig,axes=plt.subplots(2,3,figsize=(12,8))
    lims=paired_limits(np.concatenate([frames[n,role].reference for n in MODELS]),np.concatenate([frames[n,role].prediction for n in MODELS]))
    for ax,name in zip(axes.flat,MODELS):
        df=frames[name,role];m=checked_metrics(df.reference,df.prediction)
        for color,(lo,hi),label in zip(_INTERVAL_COLORS,_DEPTH_INTERVALS,_INTERVAL_LABELS):
            sub=df[(df.reference>=lo)&(df.reference<hi)]
            ax.scatter(sub.reference,sub.prediction,s=7,alpha=.5,color=color,edgecolors='none',rasterized=True)
        ax.plot(lims,lims,'k--',lw=1)
        ax.set(xlim=lims,ylim=lims,xlabel='Reference z (nominal m)',ylabel='Prediction (nominal m)',title=chr(40)+chr(97+MODELS.index(name))+chr(41)+chr(32)+LABELS[name])
        ax.text(.04,.95,f"n = {m['n']:,}\nRMSE = {m['rmse']:.3f}\nMAE = {m['mae']:.3f}\nR² = {m['r2']:.3f}",transform=ax.transAxes,va='top',fontsize=9,bbox={'facecolor':'white','alpha':.85,'edgecolor':'#cccccc'})
    ax=axes.flat[-1];ax.axis('off')
    for color,label in zip(_INTERVAL_COLORS,_INTERVAL_LABELS):ax.scatter([],[],color=color,label=label)
    ax.legend(loc='upper left',title=ROLE_LABELS[role]+' signed-target intervals',fontsize=10)
    ax.text(0,.3,'Internal spatial evaluation\nSame partition for all five models\nAll paired records retained\nUnits and vertical datum unverified',va='top',fontsize=11,transform=ax.transAxes)
    fig.tight_layout();save_figure(Path(out)/f'paired_{role}_all_models.png');plt.close(fig)

def split_figure(coords,split,out):
    _pub_style();fig,ax=plt.subplots(figsize=(8,5))
    used=np.concatenate(list(split.values()));excluded=np.setdiff1d(np.arange(len(coords)),used)
    for ids,color,label in [(excluded,'#bbbbbb',f'Excluded (n={len(excluded):,})')]+[(split[r],c,f'{ROLE_LABELS[r]} (n={len(split[r]):,})') for r,c in zip(ROLES,['#4477AA','#EEAA33','#228833'])]:
        ax.scatter(coords[ids,1]+4,coords[ids,0]+4,s=3,color=color,label=label,rasterized=True)
    ax.set(xlabel='Prepared-grid column',ylabel='Prepared-grid row');ax.invert_yaxis();ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',markerscale=2)
    fig.tight_layout();save_figure(Path(out)/'spatial_partition.png');plt.close(fig)

def fit_new_runs(args, X, y, split, pack_path, split_path):
    """Optional new training in a NEW directory; no historical split overwrites."""
    dest=args.new_run.resolve()
    if dest.exists():raise FileExistsError('Choose a new --new-run directory to preserve existing runs.')
    dest.mkdir(parents=True)
    for name in MODELS:
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(args.seed)
        tf.config.experimental.enable_op_determinism()
        model,loss,loss_weights,metric=create_model(name,input_size=(9,9,6))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss=loss,loss_weights=loss_weights,metrics=metric)
        def target(ids):
            v=y[ids,None]
            if name=='seabed_net':return {'depth':v,'seabed_class':prepare_seabed_labels(v)}
            if name=='fcnn':return np.broadcast_to(v[:,None,None,:],(len(v),9,9,1)).copy()
            return v
        run=dest/name;run.mkdir();checkpoint=run/'best.weights.h5'
        callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint),save_weights_only=True,save_best_only=True,monitor='val_loss'),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=50,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=10,min_lr=1e-7)]
        config={'model':name,'protocol':'spatial','seed':args.seed,'epochs_max':args.epochs,'batch_size':args.batch_size,'learning_rate':.0001,'dropout':.3,'patience':50,'input_sha256':sha(pack_path),'split_sha256':sha(split_path),'counts':{k:len(v) for k,v in split.items()},'tensorflow':tf.__version__,'numpy':np.__version__,'selection':'minimum validation loss only'}
        save_json(run/'config.json',config)
        h=model.fit(X[split['train']],target(split['train']),validation_data=(X[split['val']],target(split['val'])),epochs=args.epochs,batch_size=args.batch_size,callbacks=callbacks,verbose=2)
        save_json(run/'history.json',{k:[float(v) for v in values] for k,values in h.history.items()})
    return dest

def main():
    global FIG_PATH
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=root/'data/anza_study',help='Prepared work directory containing prepared/, experiments/ and reproduced/.')
    parser.add_argument('--out',type=Path,default=root/'SDB_Resultats_Corriges')
    parser.add_argument('--experiment',type=Path,help='Explicit completed run directory containing one subfolder per model.')
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--train',action='store_true',help='Explicitly train new weights; requires a fresh --new-run directory.')
    parser.add_argument('--new-run',type=Path)
    parser.add_argument('--seed',type=int,default=20260908)
    parser.add_argument('--epochs',type=int,default=300)
    parser.add_argument('--skip-historical',action='store_true')
    args=parser.parse_args()
    if args.train and args.new_run is None:parser.error('--train requires --new-run.')
    if args.new_run and not args.train:parser.error('--new-run requires --train.')
    if args.batch_size<1 or args.epochs<1:parser.error('Batch size and epochs must be positive.')
    data=args.data_root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if out==data or data in out.parents:raise ValueError('Use an output directory outside the source data.')
    originals=[root/'SDB_Visualisation_Avancee_EN.py',root/'original_scripts/SDB_Visualisation_Avancee_EN.py']
    original=next((p for p in originals if p.is_file()),None)
    if original is None:raise FileNotFoundError('The unchanged original script must accompany this corrected version.')
    scientific=verify_scientific_functions(original)
    pack_path=data/'prepared/patches.npz';split_path=data/'prepared/spatial_split.npz'
    with np.load(pack_path) as pack:X,y,coords=pack['X'],pack['y'].reshape(-1),pack['coords']
    with np.load(split_path) as pack:split={k:pack[k] for k in pack.files}
    if X.shape!=(len(y),9,9,6) or coords.shape!=(len(y),2):raise ValueError('Expected 9 x 9 x 6 patches with matching target and coordinates.')
    if not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Nonfinite prepared data.')
    overlap=validate_split(coords,split,len(y));save_json(out/'split_audit.json',overlap)
    experiment=args.experiment.resolve() if args.experiment else data/'experiments/spatial_seed20260908'
    if args.train:experiment=fit_new_runs(args,X,y,split,pack_path,split_path)
    source_hashes={'patches':sha(pack_path),'split':sha(split_path),'original_script':sha(original),'corrected_script':sha(__file__)}
    rows=[];frames={};histories={};runs=[]
    for name in MODELS:
        run=experiment/name;config=json.loads((run/'config.json').read_text())
        if config['input_sha256']!=source_hashes['patches'] or config['split_sha256']!=source_hashes['split']:
            raise ValueError('Checkpoint input/split provenance mismatch: '+name)
        if config['model']!=name or config['counts']!={k:len(v) for k,v in split.items()}:
            raise ValueError('Checkpoint model or counts mismatch: '+name)
        weights=run/'best.weights.h5'
        tf.keras.backend.clear_session();model,*_=create_model(name,input_size=(9,9,6));model.load_weights(weights)
        histories[name]=json.loads((run/'history.json').read_text())
        prediction,classes=depth_prediction(model,X,name,args.batch_size)
        checked_metrics(y,prediction)
        record={'model':name,'weights_sha256':sha(weights),'checkpoint':str(weights),'best_epoch':int(np.argmin(histories[name]['val_loss'])+1),'epochs_run':len(histories[name]['loss'])}
        for role in ROLES:
            ids=split[role]
            df=pd.DataFrame({'patch_id':ids,'row':coords[ids,0]+4,'col':coords[ids,1]+4,'reference':y[ids],'prediction':prediction[ids]})
            df['error']=df.prediction-df.reference;frames[name,role]=df
            path=out/'pairs'/f'{name}_{role}.csv';path.parent.mkdir(exist_ok=True);df.to_csv(path,index=False)
            rec={'model':name,'role':role,'best_epoch':record['best_epoch'],**checked_metrics(df.reference,df.prediction)};rows.append(rec)
            save_json(out/'metrics'/f'{name}_{role}.json',rec)
            pd.DataFrame(interval_metrics(df)).to_csv(out/'metrics'/f'{name}_{role}_intervals.csv',index=False)
            FIG_PATH=str(out/'figures'/role/name);Path(FIG_PATH).mkdir(parents=True,exist_ok=True)
            create_advanced_visualizations(df.rename(columns={'reference':'test','prediction':'pred'}),SimpleNamespace(model_type=name),'Anza',name)
        saved=run/'test_pairs.csv'
        if saved.exists():
            old=pd.read_csv(saved);new=frames[name,'test']
            if not np.array_equal(old.patch_id,new.patch_id) or not np.allclose(old.reference,new.reference,atol=1e-6,rtol=0):raise ValueError('Archived test pairing mismatch.')
            difference=float(np.max(abs(old.prediction-new.prediction)))
            if difference>1e-4:raise ValueError('Archived test predictions are not reproduced: '+str(difference))
            record['archived_test_max_abs_difference']=difference
        if classes is not None:np.savez_compressed(out/'pairs'/f'{name}_classes.npz',**{role:classes[split[role]] for role in ROLES})
        runs.append(record);print('Evaluated',name,'training / validation / test',flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics_all_partitions.csv',index=False)
    overview=out/'figures/overview';overview.mkdir(parents=True,exist_ok=True)
    training_curves(histories,overview);role_comparison(rows,overview);split_figure(coords,split,overview)
    for role in ROLES:paired_panels(frames,role,overview)
    if not args.skip_historical:
        ids=np.load(data/'prepared/legacy_last_split.npz')['test']
        for name in MODELS:
            p=np.load(data/f'reproduced/{name}_all_predictions.npy')[ids]
            df=pd.DataFrame({'test':y[ids],'pred':p});checked_metrics(df.test,df.pred)
            FIG_PATH=str(out/'figures/historical'/name);Path(FIG_PATH).mkdir(parents=True,exist_ok=True)
            create_advanced_visualizations(df,SimpleNamespace(model_type=name),'Anza',name)
    report={'source_hashes':source_hashes,'scientific_functions_unchanged':scientific,'split_audit':overlap,'runs':runs,'tensorflow':tf.__version__,'numpy':np.__version__,'figures':FIGURE_RECORDS,'scope':'Signed target in nominal metres; unverified vertical datum. Training and validation diagnostics are not external validation. Same fixed spatial partition for every model. Historical plots are reproduction only. No test-selected run ranking or confidence claims.'}
    save_json(out/'reproduction_manifest.json',report)
    print('Completed:',out,flush=True)

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
