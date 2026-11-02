"""
SDB_CNN_Complete_Models_Pipeline_ADVANCED_VISUALIZATION.py
Complete pipeline with 5 model architectures for satellite bathymetry
Version with advanced visualizations and 3D interpretations
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import time
import warnings
import glob
from scipy.ndimage import zoom
from scipy import interpolate
from scipy.ndimage import distance_transform_edt
import json
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# ============================================================================
# PUBLICATION STYLE CONSTANTS
# ============================================================================

# Custom ocean-depth colormap: dark purple → blue → cyan → pale yellow
_OCEAN_CMAP = LinearSegmentedColormap.from_list('ocean_depth', [
    '#1a0a2e', '#0d2b6e', '#1565C0', '#039BE5',
    '#26C6DA', '#80DEEA', '#E0F7FA', '#FFF9C4', '#FFF176'], N=256)

# Diverging colormap for difference maps
_DIV_CMAP = LinearSegmentedColormap.from_list('pub_div', [
    '#1565C0', '#42A5F5', '#FFFFFF', '#EF9A9A', '#C62828'], N=256)

# Colorblind-safe palette per model
_MODEL_COLORS = {
    'PI_CNN':     '#1565C0', 'pi_cnn':     '#1565C0',
    'SDB_CNN':    '#2E7D32', 'sdb_cnn':    '#2E7D32',
    'SEABED_NET': '#F57F17', 'seabed_net': '#F57F17',
    'CNN_BILSTM': '#6A1B9A', 'cnn_bilstm': '#6A1B9A',
    'FCNN':       '#B71C1C', 'fcnn':       '#B71C1C',
}
_MODEL_MARKERS = {
    'PI_CNN': 'o', 'pi_cnn': 'o',
    'SDB_CNN': 's', 'sdb_cnn': 's',
    'SEABED_NET': '^', 'seabed_net': '^',
    'CNN_BILSTM': 'D', 'cnn_bilstm': 'D',
    'FCNN': 'v', 'fcnn': 'v',
}

# Depth-interval colours and labels
_INTERVAL_COLORS = ['#1565C0', '#0097A7', '#388E3C', '#F57F17']
_INTERVAL_LABELS = [
    'Very deep (−30 to −20 m)', 'Deep (−20 to −10 m)',
    'Medium (−10 to 0 m)',       'Shallow (0 to +5 m)',
]
_DEPTH_INTERVALS = [(-30, -20), (-20, -10), (-10, 0), (0, 5)]


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
    lines = [f'RMSE  = {rmse:.3f} m', f'MAE   = {mae:.3f} m']
    if medae is not None:
        lines.append(f'MedAE = {medae:.3f} m')
    lines.append(f'R²     = {r2:.4f}')
    x0 = 0.03 if 'left' in loc else 0.97
    ha = 'left' if 'left' in loc else 'right'
    ax.text(x0, 0.97, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, va='top', ha=ha, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#BBBBBB', alpha=0.92, linewidth=0.7))


def _panel(ax, label):
    """Add (a), (b) … panel letter."""
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
DATA_PATH = os.path.join(ROOT_DIR, 'data/')
CKPT_PATH = os.path.join(ROOT_DIR, 'ckpts/')
FIG_PATH = os.path.join(ROOT_DIR, 'figs/')
RESULTS_PATH = os.path.join(ROOT_DIR, 'results/')
METRICS_PATH = os.path.join(ROOT_DIR, 'metrics/')
VIS3D_PATH = os.path.join(ROOT_DIR, 'visualizations_3d/')

# Default parameters
WINDOW = 9
STRIDE = 3
CHANNELS = 6
BATCH_SIZE = 512
LEARNING_RATE = 0.0001
EPOCHS = 300
DROPOUT_RATE = 0.3
VALIDATION_SPLIT = 0.2
PATIENCE = 50

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories():
    """Creates all necessary directories."""
    for path in [DATA_PATH, CKPT_PATH, FIG_PATH, RESULTS_PATH, METRICS_PATH, VIS3D_PATH]:
        os.makedirs(path, exist_ok=True)
    print(f"📁 Directories configured")

def extract_zone_name(filename):
    """Extracts zone name from a file."""
    name = os.path.basename(filename)
    if '_rgb.tif' in name:
        return name.replace('_rgb.tif', '')
    elif '_crt.tif' in name:
        return name.replace('_crt.tif', '')
    elif '.tif' in name:
        return name.replace('.tif', '')
    return name

def collect_npy_data_anza(data_path, rgb_file, depth_file, window=9, stride=3):
    """Specific version for ANZA."""
    print(f"\n🔍 Data collection:")
    print(f"   • RGB: {rgb_file}")
    print(f"   • Bathymetry: {depth_file}")
    
    try:
        rgb_path = os.path.join(data_path, rgb_file)
        depth_path = os.path.join(data_path, depth_file)
        
        with rasterio.open(rgb_path) as src_rgb:
            rgb_data = src_rgb.read()
            rgb_transform = src_rgb.transform
            rgb_crs = src_rgb.crs
            
        with rasterio.open(depth_path) as src_depth:
            depth_data = src_depth.read()
            depth_transform = src_depth.transform
            depth_crs = src_depth.crs
        
        # Resizing if necessary
        if rgb_data.shape[1:] != depth_data.shape[1:]:
            target_height, target_width = depth_data.shape[1], depth_data.shape[2]
            current_height, current_width = rgb_data.shape[1], rgb_data.shape[2]
            
            rgb_resized = []
            for i in range(rgb_data.shape[0]):
                channel_data = rgb_data[i:i+1, :, :]
                zoom_factor = (1, target_height/current_height, target_width/current_width)
                channel_resized = zoom(channel_data, zoom_factor, order=1)
                rgb_resized.append(channel_resized[0])
            
            rgb_data = np.array(rgb_resized)
        
        # Normalization
        if rgb_data.max() > 1.0:
            rgb_data = rgb_data / 10000.0
        rgb_data = np.clip(rgb_data, 0, 1)
        
        # Prepare depth data
        depth_data = depth_data.astype(np.float32)
        
        # Extract patches
        height, width = rgb_data.shape[1], rgb_data.shape[2]
        patches_rgb = []
        patches_depth = []
        
        for i in range(0, height - window + 1, stride):
            for j in range(0, width - window + 1, stride):
                patch_rgb = rgb_data[:, i:i+window, j:j+window]
                patch_depth = depth_data[:, i:i+window, j:j+window]
                
                if not np.isnan(patch_depth).any() and np.any(patch_depth != 0):
                    patches_rgb.append(patch_rgb)
                    center_depth = patch_depth[:, window//2, window//2]
                    patches_depth.append(center_depth)
        
        patches_rgb = np.array(patches_rgb)
        patches_depth = np.array(patches_depth)
        
        # Transpose for CNN
        patches_rgb_cnn = np.transpose(patches_rgb, (0, 2, 3, 1))
        
        return patches_rgb_cnn, patches_depth, rgb_data.shape[1:], depth_data[0]
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def merged_tiles_corrected(predictions, original_height, original_width, window_size=9, stride=3):
    """Merges patch predictions into a complete image."""
    print(f"\n🧩 Merging patches...")
    
    merged = np.full((original_height, original_width), np.nan, dtype=np.float32)
    count = np.zeros((original_height, original_width), dtype=np.int32)
    
    num_patches_height = (original_height - window_size) // stride + 1
    num_patches_width = (original_width - window_size) // stride + 1
    expected_patches = num_patches_height * num_patches_width
    
    if len(predictions) != expected_patches:
        min_patches = min(len(predictions), expected_patches)
        predictions = predictions[:min_patches]
    
    idx = 0
    half_window = window_size // 2
    
    for i in range(0, original_height - window_size + 1, stride):
        for j in range(0, original_width - window_size + 1, stride):
            if idx < len(predictions):
                center_i = i + half_window
                center_j = j + half_window
                
                if (center_i < original_height and center_j < original_width):
                    merged[center_i, center_j] = predictions[idx][0]
                    count[center_i, center_j] += 1
                
                idx += 1
    
    # Interpolation
    known_mask = ~np.isnan(merged)
    if np.sum(known_mask) > 0:
        y_coords, x_coords = np.where(known_mask)
        known_values = merged[known_mask]
        
        y_grid, x_grid = np.mgrid[0:original_height, 0:original_width]
        
        try:
            filled = interpolate.griddata(
                (y_coords, x_coords), 
                known_values, 
                (y_grid, x_grid), 
                method='linear',
                fill_value=np.nan
            )
            
            if np.isnan(filled).any():
                filled_final = filled.copy()
                nan_mask = np.isnan(filled)
                if nan_mask.any():
                    distances, indices = distance_transform_edt(
                        nan_mask, 
                        return_distances=True, 
                        return_indices=True
                    )
                    y_indices, x_indices = indices
                    filled_final[nan_mask] = filled[y_indices[nan_mask], x_indices[nan_mask]]
                
                merged = filled_final
            else:
                merged = filled
                
        except Exception as e:
            print(f"   ⚠  Interpolation error: {e}")
            mean_val = np.nanmean(merged)
            merged = np.where(np.isnan(merged), mean_val, merged)
    else:
        print(f"   ⚠  No valid data")
        merged = np.zeros((original_height, original_width))
    
    return merged

def write_tif_with_profile(input_path, reference_file, output_file, data):
    """Writes data to GeoTIFF file."""
    reference_path = os.path.join(input_path, reference_file)
    
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        profile.update({
            'dtype': 'float32',
            'count': 1,
            'nodata': -9999.0,
            'compress': 'lzw'
        })
    
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(data.astype(np.float32), 1)
    
    print(f"   ✓ File saved: {output_file}")
    return output_file

# ============================================================================
# CORRECTED ARCHITECTURE MODELS
# ============================================================================

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

# ============================================================================
# ADVANCED VISUALIZATION FUNCTIONS
# ============================================================================

def create_advanced_visualizations(df_eval, args, zone_name, model_name):
    """Creates publication-quality visualizations for interpretation."""
    print(f"\n🎨 Creating advanced visualizations (publication style)...")
    _pub_style()

    # ── 1. SCATTER PLOT WITH DEPTH INTERVALS ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for i, (lo, hi) in enumerate(_DEPTH_INTERVALS):
        mask = (df_eval['test'] >= lo) & (df_eval['test'] < hi)
        sub  = df_eval[mask]
        ax.scatter(sub['test'], sub['pred'],
                   color=_INTERVAL_COLORS[i], s=18, alpha=0.55,
                   label=f'{_INTERVAL_LABELS[i]} (n={mask.sum()})',
                   edgecolors='none', rasterized=True)

    lims = (df_eval['test'].min() - 1, df_eval['test'].max() + 1)
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
    ax.set_xlabel('Reference depth (m)'); ax.set_ylabel('Predicted depth (m)')
    ax.set_title(f'{args.model_type.upper()} — Predicted vs. Reference depth')
    ax.legend(framealpha=0.9, edgecolor='#CCCCCC', fontsize=9, loc='lower right')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.tight_layout()
    scatter_file = os.path.join(FIG_PATH, f'scatter_intervals_{model_name}.png')
    plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
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
            bar_labels.append(f'{lo} to {hi} m')
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
    ax.set_xlabel('Depth interval'); ax.set_ylabel('Error (m)')
    ax.set_title(f'{args.model_type.upper()} — Accuracy by depth interval')
    ax.legend(); ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.tight_layout()
    bar_file = os.path.join(FIG_PATH, f'errors_by_interval_{model_name}.png')
    plt.savefig(bar_file, dpi=300, bbox_inches='tight')
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
    ax.set_xlabel('Reference depth (m)'); ax.set_ylabel('Predicted depth (m)')
    ax.set_title(f'{args.model_type.upper()} — Prediction density')
    ax.legend(fontsize=9, loc='lower right')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='4%', pad=0.06)
    cb = plt.colorbar(hb, cax=cax)
    cb.set_label('Point density (log₁₀)', fontsize=10); cb.ax.tick_params(labelsize=9)
    plt.tight_layout()
    density_file = os.path.join(FIG_PATH, f'density_matrix_{model_name}.png')
    plt.savefig(density_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Density matrix: {density_file}")

    # ── 4. ERROR ANALYSIS (4-panel) ──────────────────────────────────────────
    errors   = (df_eval['pred'] - df_eval['test']).values
    abs_e    = np.abs(errors)
    df_eval['relative_error'] = (abs_e / np.abs(df_eval['test'].clip(-30, -0.1))).clip(0, 5)

    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    # (a) Error histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(errors, bins=45, color='#1565C0', alpha=0.75,
             edgecolor='white', linewidth=0.4)
    ax1.axvline(0, color='#333333', lw=1.2, ls='--', label='Zero bias')
    ax1.axvline(errors.mean(), color='#C62828', lw=1.2, ls='--',
                label=f'Mean = {errors.mean():.2f} m')
    ax1.set_xlabel('Prediction error (m)'); ax1.set_ylabel('Count')
    ax1.set_title('Error distribution'); ax1.legend(fontsize=9)
    _panel(ax1, '(a)')

    # (b) Absolute error boxplot by depth interval
    ax2  = fig.add_subplot(gs[0, 1])
    bxdata, bxlabels = [], []
    for i, (lo, hi) in enumerate(_DEPTH_INTERVALS):
        mask = (df_eval['test'] >= lo) & (df_eval['test'] < hi)
        if mask.sum() > 5:
            bxdata.append(abs_e[mask.values])
            bxlabels.append(f'{lo} to {hi} m\n(n={mask.sum()})')
    bp = ax2.boxplot(bxdata, patch_artist=True,
                     medianprops=dict(color='black', lw=1.5),
                     flierprops=dict(marker='.', ms=2, markerfacecolor='#888', alpha=0.4),
                     whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for patch, c in zip(bp['boxes'], _INTERVAL_COLORS[:len(bxdata)]):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    ax2.set_xticklabels(bxlabels, fontsize=8.5)
    ax2.set_ylabel('|Error| (m)')
    ax2.set_title('Absolute error by depth interval')
    _panel(ax2, '(b)')

    # (c) Error vs. depth
    ax3 = fig.add_subplot(gs[1, 0])
    sc3 = ax3.scatter(df_eval['test'], errors, c=abs_e, cmap='plasma',
                      s=10, alpha=0.5, vmin=0, vmax=8,
                      edgecolors='none', rasterized=True)
    ax3.axhline(0, color='#333333', lw=1.2, ls='--', alpha=0.7)
    ax3.set_xlabel('Reference depth (m)'); ax3.set_ylabel('Prediction error (m)')
    ax3.set_title('Error vs. depth')
    _panel(ax3, '(c)')
    _cbar(ax3, sc3, '|Error| (m)')

    # (d) CDF of absolute error
    ax4 = fig.add_subplot(gs[1, 1])
    sorted_e = np.sort(abs_e)
    cdf_vals = np.linspace(0, 1, len(sorted_e))
    ax4.plot(sorted_e, cdf_vals * 100, color='#1565C0', lw=1.8)
    ax4.fill_between(sorted_e, cdf_vals * 100, alpha=0.12, color='#1565C0')
    for p, ls in [(50,'--'), (75,':'), (90,'-.')]:
        v = np.percentile(abs_e, p)
        ax4.axvline(v, color='#C62828', lw=1, ls=ls, alpha=0.8,
                    label=f'{p}%ile = {v:.2f} m')
    ax4.set_xlabel('Absolute error (m)'); ax4.set_ylabel('Cumulative frequency (%)')
    ax4.set_title('CDF of absolute error'); ax4.legend(fontsize=9)
    ax4.yaxis.set_major_formatter(mticker.PercentFormatter())
    _panel(ax4, '(d)')

    fig.suptitle(f'{args.model_type.upper()} — Error analysis',
                 fontsize=14, fontweight='bold', y=0.98)
    error_analysis_file = os.path.join(FIG_PATH, f'error_analysis_{model_name}.png')
    plt.savefig(error_analysis_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Error analysis: {error_analysis_file}")

    # ── 5. PERFORMANCE HEATMAP ───────────────────────────────────────────────
    bins  = np.linspace(df_eval['test'].min(), df_eval['test'].max(), 18)
    cx    = (bins[:-1] + bins[1:]) / 2
    grid  = np.full((len(cx), len(cx)), np.nan)
    for i in range(len(cx)):
        for j in range(len(cx)):
            m = ((df_eval['test'] >= bins[i]) & (df_eval['test'] < bins[i+1]) &
                 (df_eval['pred'] >= bins[j]) & (df_eval['pred'] < bins[j+1]))
            if m.sum() >= 3:
                sub = df_eval[m]
                ref_abs = np.abs(sub['test'].clip(-30, -0.1))
                grid[i, j] = np.mean(np.abs(sub['pred'] - sub['test']) / ref_abs)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.pcolormesh(cx, cx, grid.T, cmap='RdYlGn_r', vmin=0, vmax=1,
                       shading='auto', alpha=0.9)
    ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], 'k--', lw=1.3, label='1:1 line')
    ax.set_xlabel('Reference depth (m)'); ax.set_ylabel('Predicted depth (m)')
    ax.set_title(f'{args.model_type.upper()} — Relative error heatmap')
    ax.legend(fontsize=9, loc='lower right')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='4%', pad=0.06)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label('Relative error (MAE / |depth|)', fontsize=10)
    cb.ax.tick_params(labelsize=9)
    plt.tight_layout()
    heatmap_file = os.path.join(FIG_PATH, f'performance_heatmap_{model_name}.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Performance heatmap: {heatmap_file}")

    return True

def create_3d_visualization(map_data, model_name, zone_name):
    """Creates publication-quality 4-panel 3D visualization."""
    print(f"   🎨 Creating 3D visualization...")
    _pub_style()
    plt.rcParams['axes.grid'] = False
    try:
        height, width = map_data.shape
        step = max(1, min(height, width) // 80)
        Xg, Yg = np.meshgrid(np.arange(width), np.arange(height))
        Z_clean = np.where(np.isnan(map_data), np.nanmean(map_data), map_data)
        vmin, vmax = np.nanpercentile(Z_clean, [1, 99])
        Z_filt  = Z_clean[~np.isnan(map_data)]

        fig = plt.figure(figsize=(16, 12))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.08)

        views = [(gs[0,0], 45, 225, '(a) 3D surface'),
                 (gs[0,1], 25, 260, '(b) Side view'),
                 (gs[1,0], 70, 225, '(c) Top-down view'),
                 (gs[1,1], 50, 190, '(d) Surface + contour floor')]

        s = step
        for i, (gs_pos, elev, azim, title) in enumerate(views):
            ax = fig.add_subplot(gs_pos, projection='3d')
            surf = ax.plot_surface(Xg[::s,::s], Yg[::s,::s], Z_clean[::s,::s],
                                   cmap=_OCEAN_CMAP, vmin=vmin, vmax=vmax,
                                   alpha=0.88, linewidth=0, antialiased=True,
                                   rcount=80, ccount=80)
            if i == 3:
                z_floor = np.nanmin(Z_clean) - abs(np.nanmin(Z_clean)) * 0.12
                cl = np.linspace(vmin, vmax, 10)
                ax.contourf(Xg[::s,::s], Yg[::s,::s], Z_clean[::s,::s],
                            zdir='z', offset=z_floor, levels=cl,
                            cmap=_OCEAN_CMAP, alpha=0.55)
            ax.set_title(title, fontsize=11, pad=4)
            ax.set_xlabel('X (px)', fontsize=8, labelpad=4)
            ax.set_ylabel('Y (px)', fontsize=8, labelpad=4)
            ax.set_zlabel('Depth (m)', fontsize=8, labelpad=4)
            ax.tick_params(labelsize=7)
            ax.view_init(elev=elev, azim=azim)
            cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.04, aspect=20)
            cb.set_label('Depth (m)', fontsize=8)
            cb.ax.tick_params(labelsize=7)

        fig.suptitle(f'3-D Bathymetric visualization — {model_name} — {zone_name}',
                     fontsize=13, fontweight='bold', y=1.0)
        output_file = os.path.join(VIS3D_PATH, f'3d_visualization_{model_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ 3D visualization saved: {output_file}")
        create_interactive_3d_plot(map_data, model_name, zone_name)
        return True
    except Exception as e:
        print(f"   ⚠  Error 3D: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        plt.rcParams['axes.grid'] = True

def create_interactive_3d_plot(map_data, model_name, zone_name):
    """Creates interactive 3D visualization with Plotly."""
    try:
        height, width = map_data.shape
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        Z = map_data
        
        # Create figure
        fig = go.Figure()
        
        # Main surface
        fig.add_trace(go.Surface(
            z=Z,
            x=X,
            y=Y,
            colorscale='turbo',
            cmin=-30,
            cmax=5,
            name='Surface',
            contours={
                "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project": {"z": True}}
            },
            opacity=0.9
        ))
        
        # Layout
        fig.update_layout(
            title=f'Interactive 3D Visualization - {model_name} - Zone: {zone_name}',
            scene=dict(
                xaxis_title='X (pixels)',
                yaxis_title='Y (pixels)',
                zaxis_title='Depth [m]',
                aspectratio=dict(x=1, y=height/width, z=0.5),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1))
            ),
            coloraxis_colorbar=dict(
                title="Depth [m]",
                thickness=20,
                len=0.75,
                yanchor="middle",
                y=0.5
            ),
            width=1200,
            height=800
        )
        
        # Save
        output_file = os.path.join(VIS3D_PATH, f'interactive_3d_{model_name}.html')
        fig.write_html(output_file)
        print(f"   ✓ Interactive 3D visualization saved: {output_file}")
        
    except Exception as e:
        print(f"   ⚠  Plotly error: {e}")

def create_depth_profile_visualization(ref_map, pred_map, model_name, zone_name):
    """Creates publication-quality depth transect profiles."""
    print(f"   📈 Creating depth profiles...")
    _pub_style()
    try:
        h, w = ref_map.shape
        profiles = []
        for row in [h//6, h//3, h//2, 2*h//3]:
            profiles.append(('H', row, ref_map[row,:], pred_map[row,:],
                             f'Horizontal transect (row {row})'))
        for col in [w//4, w//2, 3*w//4]:
            profiles.append(('V', col, ref_map[:,col], pred_map[:,col],
                             f'Vertical transect (col {col})'))
        n = min(h, w)
        profiles.append(('D', 0,
                         np.array([ref_map[i,i] for i in range(n)]),
                         np.array([pred_map[i,i] for i in range(n)]),
                         'Main diagonal'))

        fig, axes = plt.subplots(2, 4, figsize=(16, 7),
                                 gridspec_kw=dict(hspace=0.50, wspace=0.35))
        axes = axes.flatten()

        for idx, (kind, pos, ref_d, pred_d, title) in enumerate(profiles[:8]):
            ax = axes[idx]
            valid = ~np.isnan(ref_d) & ~np.isnan(pred_d)
            if valid.sum() < 5:
                ax.set_visible(False); continue
            x = np.arange(len(ref_d))
            ax.plot(x[valid], ref_d[valid],  color='#1565C0', lw=1.4,
                    label='Reference', alpha=0.9)
            ax.plot(x[valid], pred_d[valid], color='#C62828', lw=1.4,
                    ls='--', label=model_name, alpha=0.9)
            ax.fill_between(x[valid], ref_d[valid], pred_d[valid],
                            where=(pred_d[valid] > ref_d[valid]),
                            color='#C62828', alpha=0.12)
            ax.fill_between(x[valid], ref_d[valid], pred_d[valid],
                            where=(pred_d[valid] <= ref_d[valid]),
                            color='#1565C0', alpha=0.12)
            _mae_p  = mean_absolute_error(ref_d[valid], pred_d[valid])
            _rmse_p = np.sqrt(mean_squared_error(ref_d[valid], pred_d[valid]))
            ax.text(0.97, 0.97, f'MAE={_mae_p:.2f} m\nRMSE={_rmse_p:.2f} m',
                    transform=ax.transAxes, fontsize=8.5, va='top', ha='right',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#CCCCCC', alpha=0.9))
            ax.set_xlabel('Pixel index', fontsize=9)
            ax.set_ylabel('Depth (m)', fontsize=9)
            ax.set_title(title, fontsize=9.5, fontweight='bold')
            ax.tick_params(labelsize=8)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            if idx == 0:
                ax.legend(fontsize=8, ncol=1, loc='lower right')
            # auto Y limits
            all_v = np.concatenate([ref_d[valid], pred_d[valid]])
            ax.set_ylim(all_v.min()-1.5, all_v.max()+1.5)

        fig.suptitle(f'Depth transect profiles — {model_name} — {zone_name}',
                     fontsize=13, fontweight='bold', y=0.98)
        output_file = os.path.join(FIG_PATH, f'depth_profiles_{model_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Depth profiles saved: {output_file}")
        return True
    except Exception as e:
        print(f"   ⚠  Error profiles: {e}")
        return False

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

# ============================================================================
# PHASE 1: DATA COLLECTION
# ============================================================================

def collect_data(args):
    """Phase 1: Data collection and preparation."""
    print("\n" + "="*80)
    print("PHASE 1: DATA COLLECTION")
    print("="*80)
    
    setup_directories()
    
    if not args.rgb_file or not args.depth_file:
        print("❌ Please specify --rgb and --depth")
        return False
    
    rgb_path = os.path.join(DATA_PATH, args.rgb_file)
    depth_path = os.path.join(DATA_PATH, args.depth_file)
    
    if not os.path.exists(rgb_path):
        print(f"❌ RGB file not found: {rgb_path}")
        return False
    
    if not os.path.exists(depth_path):
        print(f"❌ Depth file not found: {depth_path}")
        return False
    
    zone_name = extract_zone_name(args.rgb_file)
    print(f"📍 Study area: {zone_name}")
    
    patches_rgb, patches_depth, original_shape, depth_original = collect_npy_data_anza(
        DATA_PATH, args.rgb_file, args.depth_file, args.window, args.stride
    )
    
    if patches_rgb is None:
        return False
    
    # Save
    np.save(os.path.join(DATA_PATH, f'rgbnss_{zone_name}.npy'), patches_rgb)
    np.save(os.path.join(DATA_PATH, f'depth_{zone_name}.npy'), patches_depth)
    np.save(os.path.join(DATA_PATH, f'original_shape_{zone_name}.npy'), np.array(original_shape))
    np.save(os.path.join(DATA_PATH, f'depth_original_{zone_name}.npy'), depth_original)
    
    print(f"\n✅ Data saved:")
    print(f"   • rgbnss_{zone_name}.npy: {patches_rgb.shape}")
    print(f"   • depth_{zone_name}.npy: {patches_depth.shape}")
    
    return True

# ============================================================================
# PHASE 2: TRAINING
# ============================================================================

def train_model(args):
    """Phase 2: Model training."""
    print("\n" + "="*80)
    print(f"PHASE 2: TRAINING - Model: {args.model_type.upper()}")
    print("="*80)
    
    setup_directories()
    
    if not args.rgb_file:
        print("❌ Please specify --rgb")
        return False
    
    zone_name = extract_zone_name(args.rgb_file)
    
    rgb_file = f'rgbnss_{zone_name}.npy'
    depth_file = f'depth_{zone_name}.npy'
    
    rgb_path = os.path.join(DATA_PATH, rgb_file)
    depth_path = os.path.join(DATA_PATH, depth_file)
    
    if not os.path.exists(rgb_path):
        print(f"❌ RGB file not found: {rgb_path}")
        return False
    
    if not os.path.exists(depth_path):
        print(f"❌ Depth file not found: {depth_path}")
        return False
    
    print(f"\n📊 Loading data...")
    rgb_all = np.load(rgb_path)
    depth_all = np.load(depth_path)
    
    print(f"   ✓ Data loaded: {rgb_all.shape}")
    
    indices = np.random.permutation(len(rgb_all))
    rgb_all = rgb_all[indices]
    depth_all = depth_all[indices]
    
    split_idx = int(len(rgb_all) * 0.8)
    split_idx2 = int(len(rgb_all) * 0.9)
    
    rgb_trn, rgb_val, rgb_tst = np.split(rgb_all, [split_idx, split_idx2])
    depth_trn, depth_val, depth_tst = np.split(depth_all, [split_idx, split_idx2])
    
    np.save(os.path.join(DATA_PATH, f'rgbnss_trn_{zone_name}.npy'), rgb_trn)
    np.save(os.path.join(DATA_PATH, f'rgbnss_val_{zone_name}.npy'), rgb_val)
    np.save(os.path.join(DATA_PATH, f'rgbnss_tst_{zone_name}.npy'), rgb_tst)
    np.save(os.path.join(DATA_PATH, f'depth_trn_{zone_name}.npy'), depth_trn)
    np.save(os.path.join(DATA_PATH, f'depth_val_{zone_name}.npy'), depth_val)
    np.save(os.path.join(DATA_PATH, f'depth_tst_{zone_name}.npy'), depth_tst)
    
    print(f"   • Train: {len(rgb_trn)}")
    print(f"   • Validation: {len(rgb_val)}")
    print(f"   • Test: {len(rgb_tst)}")
    
    if args.model_type == 'seabed_net':
        seabed_trn = prepare_seabed_labels(depth_trn)
        seabed_val = prepare_seabed_labels(depth_val)
        seabed_tst = prepare_seabed_labels(depth_tst)
        
        np.save(os.path.join(DATA_PATH, f'seabed_trn_{zone_name}.npy'), seabed_trn)
        np.save(os.path.join(DATA_PATH, f'seabed_val_{zone_name}.npy'), seabed_val)
        np.save(os.path.join(DATA_PATH, f'seabed_tst_{zone_name}.npy'), seabed_tst)
        
        print(f"   • Seabed labels generated: {seabed_trn.shape}")
    
    model_config = {
        'input_size': (args.window, args.window, args.channels),
        'dropout_rate': args.dropout,
        'lambda_phys': 0.1,
        'lstm_units': 64,
        'num_classes': 4
    }
    
    try:
        model, loss, loss_weights, metrics = create_model(args.model_type, **model_config)
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model_type}_{zone_name}_{timestamp}"
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=1,
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(CKPT_PATH, f'{model_name}.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(CKPT_PATH, 'logs', model_name),
        histogram_freq=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [model_checkpoint, tensorboard, early_stopping, reduce_lr]
    
    print(f"\n⚙️  Compiling model...")
    try:
        if args.model_type == 'seabed_net':
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss=loss,
                loss_weights=loss_weights,
                metrics=metrics
            )
            
            train_data = rgb_trn
            train_labels = {'depth': depth_trn, 'seabed_class': seabed_trn}
            val_data = rgb_val
            val_labels = {'depth': depth_val, 'seabed_class': seabed_val}
        elif args.model_type == 'fcnn':
            print(f"   • Adapting labels for FCNN...")
            
            depth_trn_fcnn = np.repeat(depth_trn[:, np.newaxis, np.newaxis, :], 9, axis=1)
            depth_trn_fcnn = np.repeat(depth_trn_fcnn, 9, axis=2)
            
            depth_val_fcnn = np.repeat(depth_val[:, np.newaxis, np.newaxis, :], 9, axis=1)
            depth_val_fcnn = np.repeat(depth_val_fcnn, 9, axis=2)
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss=loss,
                metrics=metrics
            )
            
            train_data = rgb_trn
            train_labels = depth_trn_fcnn
            val_data = rgb_val
            val_labels = depth_val_fcnn
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss=loss,
                metrics=metrics
            )
            
            train_data = rgb_trn
            train_labels = depth_trn
            val_data = rgb_val
            val_labels = depth_val
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    model.summary()
    
    print(f"\n🚀 Starting training...")
    start_time = time.time()
    
    try:
        if args.model_type == 'seabed_net':
            history = model.fit(
                x=train_data,
                y=train_labels,
                batch_size=args.batch_size,
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=1,
                validation_data=(val_data, val_labels)
            )
        else:
            history = model.fit(
                x=train_data,
                y=train_labels,
                batch_size=args.batch_size,
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=1,
                validation_data=(val_data, val_labels)
            )
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    elapsed_time = time.time() - start_time
    print(f'\n✅ Training completed. Time: {elapsed_time:.2f} seconds')
    
    history_path = os.path.join(RESULTS_PATH, f'history_{model_name}.json')
    try:
        with open(history_path, 'w') as f:
            history_dict = {}
            for key, value in history.history.items():
                history_dict[key] = [float(v) for v in value]
            json.dump(history_dict, f, indent=2)
        print(f"   ✓ History saved: {history_path}")
    except Exception as e:
        print(f"   ⚠  Error saving history: {e}")
    
    return True

# ============================================================================
# PHASE 3: TESTING WITH VISUALIZATIONS
# ============================================================================

def test_model(args):
    """Phase 3: Model testing with advanced visualizations."""
    print("\n" + "="*80)
    print(f"PHASE 3: TESTING - Model: {args.model_type.upper()}")
    print("="*80)
    
    setup_directories()
    
    if not args.rgb_file:
        print("❌ Please specify --rgb")
        return False
    
    zone_name = extract_zone_name(args.rgb_file)
    
    rgb_test_file = f'rgbnss_tst_{zone_name}.npy'
    depth_test_file = f'depth_tst_{zone_name}.npy'
    
    rgb_path = os.path.join(DATA_PATH, rgb_test_file)
    depth_path = os.path.join(DATA_PATH, depth_test_file)
    
    if not os.path.exists(rgb_path):
        print(f"❌ Test RGB file not found: {rgb_path}")
        return False
    
    if not os.path.exists(depth_path):
        print(f"❌ Test Depth file not found: {depth_path}")
        return False
    
    print(f"\n📊 Loading test data...")
    rgb_tst = np.load(rgb_path)
    depth_tst = np.load(depth_path)
    
    print(f"   ✓ Test images: {rgb_tst.shape}")
    print(f"   ✓ Test depths: {depth_tst.shape}")
    
    print(f"\n🏗️  Searching for model...")
    model_pattern = os.path.join(CKPT_PATH, f"{args.model_type}_{zone_name}_*.h5")
    model_files = glob.glob(model_pattern)

    if not model_files:
        all_files = glob.glob(os.path.join(CKPT_PATH, "*.h5"))
        model_files = [f for f in all_files if args.model_type in f and zone_name in f]
        
        if not model_files:
            print(f"❌ No model found for {args.model_type}")
            return False

    latest_model = sorted(model_files, key=os.path.getmtime)[-1]
    model_name = os.path.basename(latest_model).replace('.h5', '')
    
    print(f"   ✓ Models found: {len(model_files)}")
    print(f"   ✓ Using model: {model_name}")
    
    model_config = {
        'input_size': (args.window, args.window, args.channels),
        'dropout_rate': args.dropout
    }
    
    try:
        model, _, _, _ = create_model(args.model_type, **model_config)
        model.load_weights(latest_model)
        print(f"   ✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n🔮 Making predictions...")
    
    try:
        if args.model_type == 'seabed_net':
            depth_pred, seabed_pred = model.predict(rgb_tst, verbose=1, batch_size=args.batch_size)
            print(f"   ✓ Depth predictions: {depth_pred.shape}")
            print(f"   ✓ Seabed predictions: {seabed_pred.shape}")
            
            pred_depth_file = f'depth_pred_{model_name}.npy'
            pred_seabed_file = f'seabed_pred_{model_name}.npy'
            
            np.save(os.path.join(DATA_PATH, pred_depth_file), depth_pred)
            np.save(os.path.join(DATA_PATH, pred_seabed_file), seabed_pred)
            
            print(f"   ✓ Predictions saved: {pred_depth_file}, {pred_seabed_file}")
            
            y_test = depth_tst
            y_pred = depth_pred
            
        elif args.model_type == 'fcnn':
            pred_maps = model.predict(rgb_tst, verbose=1, batch_size=args.batch_size)
            print(f"   ✓ FCNN predictions: {pred_maps.shape}")
            
            depth_pred = pred_maps[:, 4, 4, 0].reshape(-1, 1)
            print(f"   ✓ Central values extracted: {depth_pred.shape}")
            
            pred_file = f'depth_pred_{model_name}.npy'
            pred_maps_file = f'depth_maps_{model_name}.npy'
            
            np.save(os.path.join(DATA_PATH, pred_file), depth_pred)
            np.save(os.path.join(DATA_PATH, pred_maps_file), pred_maps)
            
            print(f"   ✓ Predictions saved: {pred_file}")
            print(f"   ✓ Complete maps saved: {pred_maps_file}")
            
            y_test = depth_tst
            y_pred = depth_pred
            
        else:
            depth_pred = model.predict(rgb_tst, verbose=1, batch_size=args.batch_size)
            print(f"   ✓ Predictions: {depth_pred.shape}")
            
            pred_file = f'depth_pred_{model_name}.npy'
            np.save(os.path.join(DATA_PATH, pred_file), depth_pred)
            
            print(f"   ✓ Predictions saved: {pred_file}")
            
            y_test = depth_tst
            y_pred = depth_pred
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False
    
    # Evaluation with advanced visualizations
    try:
        # Prepare data for evaluation
        df_eval = pd.DataFrame({'test': y_test.flatten(), 'pred': y_pred.flatten()})
        df_eval = df_eval.dropna()
        
        # Calculate metrics
        mae = mean_absolute_error(df_eval['test'], df_eval['pred'])
        rmse = np.sqrt(mean_squared_error(df_eval['test'], df_eval['pred']))
        r2 = r2_score(df_eval['test'], df_eval['pred'])
        medae = median_absolute_error(df_eval['test'], df_eval['pred'])
        
        print(f"\n📊 MODEL EVALUATION:")
        print(f"   • MAE:  {mae:.4f} m")
        print(f"   • RMSE: {rmse:.4f} m")
        print(f"   • MedAE: {medae:.4f} m")
        print(f"   • R²:   {r2:.4f}")
        
        # Create advanced visualizations
        df_eval['error'] = df_eval['pred'] - df_eval['test']
        df_eval = df_eval.loc[(df_eval.test < 22.0) & (df_eval.test >= -22.0)]
        
        create_advanced_visualizations(df_eval, args, zone_name, model_name)
        
        # Save results
        results = {
            'model': args.model_type,
            'zone': zone_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics': {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'MedAE': float(medae),
                'R2': float(r2)
            },
            'samples': len(df_eval)
        }
        
        results_file = os.path.join(RESULTS_PATH, f'results_{model_name}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   ✓ Results saved: {results_file}")
        
        metrics_file = os.path.join(METRICS_PATH, f'metrics_{args.model_type}_{zone_name}.csv')
        metrics_df = pd.DataFrame([{
            'model': args.model_type,
            'zone': zone_name,
            'mae': mae,
            'rmse': rmse,
            'medae': medae,
            'r2': r2,
            'samples': len(df_eval),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        
        if os.path.exists(metrics_file):
            existing_df = pd.read_csv(metrics_file)
            metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
        
        metrics_df.to_csv(metrics_file, index=False)
        print(f"   ✓ Metrics saved: {metrics_file}")
        
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

# ============================================================================
# PHASE 4: MAP GENERATION WITH 3D VISUALIZATIONS
# ============================================================================

def generate_map(args):
    """Phase 4: Bathymetric map generation with 3D visualizations."""
    print("\n" + "="*80)
    print(f"PHASE 4: MAP GENERATION - Model: {args.model_type.upper()}")
    print("="*80)
    
    setup_directories()
    
    if not args.rgb_file or not args.depth_file:
        print("❌ Please specify --rgb and --depth")
        return False
    
    zone_name = extract_zone_name(args.rgb_file)
    
    rgb_full_file = f'rgbnss_{zone_name}.npy'
    original_shape_file = f'original_shape_{zone_name}.npy'
    depth_ref_file = args.depth_file
    
    rgb_path = os.path.join(DATA_PATH, rgb_full_file)
    shape_path = os.path.join(DATA_PATH, original_shape_file)
    depth_ref_path = os.path.join(DATA_PATH, depth_ref_file)
    
    if not os.path.exists(rgb_path):
        print(f"❌ Complete RGB file not found: {rgb_full_file}")
        return False
    
    if not os.path.exists(shape_path):
        print(f"❌ Original shape file not found")
        return False
    
    if not os.path.exists(depth_ref_path):
        print(f"❌ Reference file not found: {depth_ref_file}")
        return False
    
    print(f"\n📊 Loading data...")
    rgb_full = np.load(rgb_path)
    original_shape = np.load(shape_path)
    
    print(f"   ✓ Complete images: {rgb_full.shape}")
    print(f"   ✓ Original dimensions: {original_shape}")
    
    print(f"\n🏗️  Searching for model...")
    model_pattern = os.path.join(CKPT_PATH, f"{args.model_type}_{zone_name}_*.h5")
    model_files = glob.glob(model_pattern)

    if not model_files:
        all_files = glob.glob(os.path.join(CKPT_PATH, "*.h5"))
        model_files = [f for f in all_files if args.model_type in f and zone_name in f]
        
        if not model_files:
            print(f"❌ No model found for {args.model_type}")
            return False

    latest_model = sorted(model_files, key=os.path.getmtime)[-1]
    model_name = os.path.basename(latest_model).replace('.h5', '')
    
    print(f"   ✓ Using model: {model_name}")
    
    model_config = {
        'input_size': (args.window, args.window, args.channels),
        'dropout_rate': args.dropout
    }
    
    try:
        model, _, _, _ = create_model(args.model_type, **model_config)
        model.load_weights(latest_model)
        print(f"   ✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    print(f"\n🔮 Predictions on complete image...")
    
    try:
        if args.model_type == 'fcnn':
            pred_maps = model.predict(rgb_full, verbose=1, batch_size=args.batch_size)
            print(f"   ✓ FCNN predictions: {pred_maps.shape}")
            depth_pred = pred_maps[:, 4, 4, 0].reshape(-1, 1)
            
        elif args.model_type == 'seabed_net':
            depth_pred, seabed_pred = model.predict(rgb_full, verbose=1, batch_size=args.batch_size)
            print(f"   ✓ Depth predictions: {depth_pred.shape}")
            print(f"   ✓ Seabed predictions: {seabed_pred.shape}")
            
            seabed_classes = np.argmax(seabed_pred, axis=1)
            seabed_map = merged_tiles_corrected(
                seabed_classes.reshape(-1, 1),
                original_shape[0],
                original_shape[1],
                args.window,
                args.stride
            )
            
            seabed_output = os.path.join(DATA_PATH, f'seabed_map_{model_name}.tif')
            write_tif_with_profile(DATA_PATH, depth_ref_file, seabed_output, seabed_map)
            print(f"   ✓ Class map saved: seabed_map_{model_name}.tif")
            
        else:
            depth_pred = model.predict(rgb_full, verbose=1, batch_size=args.batch_size)
        
        print(f"   ✓ Depth predictions: {depth_pred.shape}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False
    
    merged_map = merged_tiles_corrected(
        depth_pred,
        original_shape[0],
        original_shape[1],
        args.window,
        args.stride
    )
    
    print(f"\n💾 Saving map...")
    output_file = os.path.join(DATA_PATH, f'sdb_{model_name}.tif')
    
    try:
        write_tif_with_profile(
            DATA_PATH,
            depth_ref_file,
            output_file,
            merged_map
        )
        
        print(f"\n✅ Map generated successfully: sdb_{model_name}.tif")
    except Exception as e:
        print(f"❌ Error saving map: {e}")
        return False
    
    # Load reference for visualizations
    try:
        with rasterio.open(depth_ref_path) as src:
            ref_data = src.read(1)
        
        # 1. Publication-quality 2D bathymetric map comparison
        _pub_style()
        diff     = merged_map - ref_data
        vmin_map = np.nanpercentile(ref_data, 2)
        vmax_map = np.nanpercentile(ref_data, 98)
        d_lim    = np.nanpercentile(np.abs(diff[~np.isnan(diff)]), 95)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                                 gridspec_kw={'wspace': 0.32})
        panels_map = [
            (axes[0], ref_data,    _OCEAN_CMAP, vmin_map, vmax_map,
             'Depth (m)',       f'(a) Reference — {zone_name}'),
            (axes[1], merged_map,  _OCEAN_CMAP, vmin_map, vmax_map,
             'Depth (m)',       f'(b) {args.model_type.upper()} prediction'),
            (axes[2], diff,        _DIV_CMAP,   -d_lim,   d_lim,
             'Residual (m)',    '(c) Residual (pred − ref)'),
        ]
        for ax, data, cmap, v0, v1, cblabel, title in panels_map:
            im = ax.imshow(data, cmap=cmap, vmin=v0, vmax=v1,
                           origin='upper', aspect='auto', interpolation='bilinear')
            ax.set_title(title, fontsize=11, pad=5)
            ax.axis('off')
            _cbar(ax, im, cblabel)

        fig_file = os.path.join(FIG_PATH, f'map_comparison_{model_name}.png')
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ 2D visualization saved: map_comparison_{model_name}.png")
        
        # 2. Create 3D visualization of predicted map
        create_3d_visualization(merged_map, model_name, zone_name)
        
        # 3. Create depth profiles
        create_depth_profile_visualization(ref_data, merged_map, model_name, zone_name)
        
        # 4. Comparative 3D visualization
        create_comparative_3d_visualization(ref_data, merged_map, model_name, zone_name)
        
    except Exception as e:
        print(f"   ⚠  Error in visualizations: {e}")
    
    return True

def create_comparative_3d_visualization(ref_map, pred_map, model_name, zone_name):
    """Creates publication-quality comparative 3D: reference | prediction | residual | overlay."""
    _pub_style()
    plt.rcParams['axes.grid'] = False
    try:
        h, w = ref_map.shape
        step  = max(1, min(h, w) // 80)
        Xg, Yg = np.meshgrid(np.arange(w), np.arange(h))
        Z_ref  = np.where(np.isnan(ref_map),  np.nanmean(ref_map),  ref_map)
        Z_pred = np.where(np.isnan(pred_map), np.nanmean(pred_map), pred_map)
        Z_diff = Z_pred - Z_ref
        vmin, vmax  = np.nanpercentile(Z_ref, [1, 99])
        d_lim = np.nanpercentile(np.abs(Z_diff), 95)
        s = step

        fig = plt.figure(figsize=(20, 14))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.08)

        panels = [
            (gs[0,0], Z_ref,  _OCEAN_CMAP,  vmin,  vmax,  'Depth (m)',
             '(a) Reference bathymetry', 45, 225),
            (gs[0,1], Z_pred, _OCEAN_CMAP,  vmin,  vmax,  'Depth (m)',
             f'(b) {model_name} prediction', 45, 225),
            (gs[1,0], Z_diff, _DIV_CMAP,   -d_lim, d_lim, 'Δ Depth (m)',
             '(c) Residual (pred − ref)', 40, 225),
            (gs[1,1], Z_pred, _OCEAN_CMAP,  vmin,  vmax,  'Depth (m)',
             '(d) Prediction + floor contours', 50, 190),
        ]
        for i, (gs_pos, Z, cmap, v0, v1, cblabel, title, elev, azim) in enumerate(panels):
            ax = fig.add_subplot(gs_pos, projection='3d')
            surf = ax.plot_surface(Xg[::s,::s], Yg[::s,::s], Z[::s,::s],
                                   cmap=cmap, vmin=v0, vmax=v1,
                                   alpha=0.88, linewidth=0, antialiased=True,
                                   rcount=80, ccount=80)
            if i == 3:
                z_fl = np.nanmin(Z) - abs(np.nanmin(Z)) * 0.12
                ax.contourf(Xg[::s,::s], Yg[::s,::s], Z[::s,::s],
                            zdir='z', offset=z_fl,
                            levels=np.linspace(v0, v1, 10),
                            cmap=cmap, alpha=0.55)
            ax.set_title(title, fontsize=11, pad=4)
            ax.set_xlabel('X (px)', fontsize=8, labelpad=4)
            ax.set_ylabel('Y (px)', fontsize=8, labelpad=4)
            ax.set_zlabel('Depth (m)', fontsize=8, labelpad=4)
            ax.tick_params(labelsize=7)
            ax.view_init(elev=elev, azim=azim)
            cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.04, aspect=20)
            cb.set_label(cblabel, fontsize=8); cb.ax.tick_params(labelsize=7)

        fig.suptitle(f'Comparative 3-D Analysis — {model_name} — {zone_name}',
                     fontsize=13, fontweight='bold', y=1.0)
        output_file = os.path.join(VIS3D_PATH, f'comparative_3d_{model_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Comparative 3D saved: {output_file}")
        return True
    except Exception as e:
        print(f"   ⚠  Error comparative 3D: {e}")
        return False
    finally:
        plt.rcParams['axes.grid'] = True

def generate_complete_dashboard(df_eval, args, zone_name):
    """Generates a publication-quality 9-panel analysis dashboard."""
    print(f"\n📊 Generating analysis dashboard...")
    _pub_style()

    # Recompute summary metrics from df_eval
    rmse  = np.sqrt(mean_squared_error(df_eval['test'], df_eval['pred']))
    mae   = mean_absolute_error(df_eval['test'], df_eval['pred'])
    medAE = median_absolute_error(df_eval['test'], df_eval['pred'])
    r2    = r2_score(df_eval['test'], df_eval['pred'])

    try:
        fig = plt.figure(figsize=(20, 16))
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.40)
        errors = (df_eval['pred'] - df_eval['test']).values
        abs_e  = np.abs(errors)

        # (a) Scatter
        ax1 = fig.add_subplot(gs[0, 0])
        sc1 = ax1.scatter(df_eval['test'], df_eval['pred'],
                          c=df_eval['test'], cmap=_OCEAN_CMAP,
                          alpha=0.55, s=15, edgecolors='none', rasterized=True)
        lims = (df_eval['test'].min()-1, df_eval['test'].max()+1)
        ax1.plot(lims, lims, 'k--', lw=1.2, alpha=0.7)
        ax1.set_xlabel('Reference depth (m)'); ax1.set_ylabel('Predicted depth (m)')
        ax1.set_title('Predicted vs. Reference')
        _cbar(ax1, sc1, 'Depth (m)'); _panel(ax1, '(a)')
        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

        # (b) Error histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(errors, bins=45, color='#1565C0', alpha=0.75, edgecolor='white', lw=0.4)
        ax2.axvline(0, color='#333', lw=1.2, ls='--')
        ax2.axvline(errors.mean(), color='#C62828', lw=1.2, ls='--',
                    label=f'Mean={errors.mean():.2f} m')
        ax2.set_xlabel('Error (m)'); ax2.set_ylabel('Count')
        ax2.set_title('Error distribution'); ax2.legend(fontsize=9)
        _panel(ax2, '(b)')

        # (c) Error vs depth
        ax3 = fig.add_subplot(gs[0, 2])
        sc3 = ax3.scatter(df_eval['test'], errors, c=abs_e, cmap='plasma',
                          s=10, alpha=0.5, vmin=0, vmax=8,
                          edgecolors='none', rasterized=True)
        ax3.axhline(0, color='#333', lw=1.2, ls='--')
        ax3.set_xlabel('Reference depth (m)'); ax3.set_ylabel('Error (m)')
        ax3.set_title('Error vs. Depth')
        _cbar(ax3, sc3, '|Error| (m)'); _panel(ax3, '(c)')

        # (d) Boxplot by depth interval
        ax4 = fig.add_subplot(gs[1, 0])
        intervals = [(-30,-20),(-20,-10),(-10,0),(0,5)]
        bx_data, bx_lab = [], []
        for lo, hi in intervals:
            mask = (df_eval['test'] >= lo) & (df_eval['test'] < hi)
            if mask.any():
                bx_data.append(errors[mask.values])
                bx_lab.append(f'{lo} to\n{hi} m')
        bp = ax4.boxplot(bx_data, patch_artist=True, labels=bx_lab,
                         medianprops=dict(color='black', lw=1.5),
                         flierprops=dict(marker='.', ms=2, markerfacecolor='#888', alpha=0.4))
        for patch, c in zip(bp['boxes'], _INTERVAL_COLORS[:len(bx_data)]):
            patch.set_facecolor(c); patch.set_alpha(0.65)
        ax4.axhline(0, color='#333', lw=1, ls='--')
        ax4.set_ylabel('Error (m)'); ax4.set_title('Error by depth interval')
        ax4.tick_params(axis='x', labelsize=8)
        _panel(ax4, '(d)')

        # (e) CDF
        ax5 = fig.add_subplot(gs[1, 1])
        se = np.sort(abs_e); cdf5 = np.linspace(0,1,len(se))
        ax5.plot(se, cdf5*100, color='#1565C0', lw=1.8)
        ax5.fill_between(se, cdf5*100, alpha=0.12, color='#1565C0')
        for p, ls in [(50,'--'),(75,':'),(90,'-.')]:
            v = np.percentile(abs_e, p)
            ax5.axvline(v, color='#C62828', lw=1, ls=ls, alpha=0.8,
                        label=f'{p}%={v:.2f} m')
        ax5.set_xlabel('|Error| (m)'); ax5.set_ylabel('Cumulative (%)')
        ax5.set_title('CDF of absolute error'); ax5.legend(fontsize=9)
        ax5.yaxis.set_major_formatter(mticker.PercentFormatter())
        _panel(ax5, '(e)')

        # (f) Density hexbin
        ax6 = fig.add_subplot(gs[1, 2])
        hb6 = ax6.hexbin(df_eval['test'], df_eval['pred'],
                         gridsize=40, cmap='YlOrRd', bins='log', alpha=0.9)
        ax6.plot(lims, lims, 'k--', lw=1.2, alpha=0.7)
        ax6.set_xlabel('Reference depth (m)'); ax6.set_ylabel('Predicted depth (m)')
        ax6.set_title('Prediction density')
        div6 = make_axes_locatable(ax6)
        cax6 = div6.append_axes('right', size='4%', pad=0.06)
        cb6  = plt.colorbar(hb6, cax=cax6); cb6.set_label('Density (log)', fontsize=9)
        cb6.ax.tick_params(labelsize=8)
        _panel(ax6, '(f)')

        # (g) Metrics text block
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.axis('off')
        info = ("  MODEL: " + args.model_type.upper() + "\n"
                "  ZONE:  " + zone_name + "\n\n"
                f"  RMSE  = {rmse:.4f} m\n"
                f"  MAE   = {mae:.4f} m\n"
                f"  MedAE = {medAE:.4f} m\n"
                f"  R2     = {r2:.4f}\n\n"
                f"  Bias  = {errors.mean():.4f} m\n"
                f"  Std   = {errors.std():.4f} m\n"
                f"  n     = {len(df_eval)}")
        ax7.text(0.05, 0.95, info, transform=ax7.transAxes, fontsize=10,
                 va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5',
                           edgecolor='#BBBBBB', alpha=0.95))
        _panel(ax7, '(g)')

        # (h) MAE/RMSE bar by interval
        ax8 = fig.add_subplot(gs[2, 1])
        m_vals, r_vals, h_lab = [], [], []
        for lo, hi in intervals:
            mask = (df_eval['test'] >= lo) & (df_eval['test'] < hi)
            if mask.any():
                sub = df_eval[mask]
                m_vals.append(mean_absolute_error(sub['test'], sub['pred']))
                r_vals.append(np.sqrt(mean_squared_error(sub['test'], sub['pred'])))
                h_lab.append(f'{lo} to {hi} m')
        xb = np.arange(len(m_vals)); wb = 0.34
        ax8.bar(xb-wb/2, m_vals, wb, label='MAE',  color='#1565C0', alpha=0.85, edgecolor='white')
        ax8.bar(xb+wb/2, r_vals, wb, label='RMSE', color='#C62828', alpha=0.85, edgecolor='white')
        ax8.set_xticks(xb); ax8.set_xticklabels(h_lab, fontsize=8)
        ax8.set_ylabel('Error (m)'); ax8.set_title('Performance by depth interval')
        ax8.legend(fontsize=9); ax8.yaxis.set_minor_locator(AutoMinorLocator(2))
        _panel(ax8, '(h)')

        # (i) Radar/spider chart
        ax9 = fig.add_subplot(gs[2, 2], polar=True)
        radar_labels = ['Accuracy', 'Precision', 'Bias\n(inv.)', 'Variance\n(inv.)', 'Stability']
        N = len(radar_labels)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        norm_mae   = max(0, 1 - mae / 10)
        norm_rmse  = max(0, 1 - rmse / 15)
        norm_bias  = max(0, 1 - abs(errors.mean()) / 5)
        norm_var   = max(0, 1 - errors.var() / 25)
        values = [norm_mae, norm_rmse, norm_bias, norm_var, r2]
        values += values[:1]
        ax9.set_theta_offset(np.pi / 2); ax9.set_theta_direction(-1)
        ax9.set_xticks(angles[:-1]); ax9.set_xticklabels(radar_labels, fontsize=9)
        ax9.set_ylim(0, 1); ax9.set_yticks([0.25,0.50,0.75,1.0])
        ax9.set_yticklabels(['0.25','0.50','0.75','1.0'], fontsize=7)
        ax9.plot(angles, values, color='#1565C0', lw=2, marker='o', ms=5)
        ax9.fill(angles, values, color='#1565C0', alpha=0.18)
        ax9.set_title('Performance profile', fontsize=11, pad=14)
        _panel(ax9, '(i)')   # not ideal on polar but harmless

        fig.suptitle(f'ANALYSIS DASHBOARD — {args.model_type.upper()} — {zone_name}',
                     fontsize=15, fontweight='bold', y=0.98)
        dashboard_file = os.path.join(FIG_PATH, f'dashboard_{args.model_type}_{zone_name}.png')
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Dashboard saved: {dashboard_file}")
        return True
    except Exception as e:
        print(f"   ⚠  Error dashboard: {e}")
        import traceback; traceback.print_exc()
        return False

def compare_all_models(args):
    """Phase 6: Comparison of all trained models."""
    print("\n" + "="*80)
    print("PHASE 6: COMPARISON OF ALL MODELS")
    print("="*80)
    
    setup_directories()
    
    if not args.rgb_file:
        print("❌ Please specify --rgb")
        return False
    
    zone_name = extract_zone_name(args.rgb_file)
    
    results_files = glob.glob(os.path.join(RESULTS_PATH, f'results_*.json'))
    
    if not results_files:
        print("❌ No results found")
        return False
    
    all_results = []
    model_types = ['sdb_cnn', 'pi_cnn', 'cnn_bilstm', 'fcnn', 'seabed_net']
    
    for model_type in model_types:
        model_results = []
        for file in results_files:
            if f'_{model_type}_{zone_name}_' in file:
                try:
                    with open(file, 'r') as f:
                        result = json.load(f)
                        model_results.append(result)
                except:
                    continue
        
        if model_results:
            best_result = min(model_results, key=lambda x: x['metrics']['RMSE'])
            all_results.append(best_result)
    
    if not all_results:
        print("❌ No valid results found")
        return False
    
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Model': result['model'].upper(),
            'RMSE (m)': result['metrics']['RMSE'],
            'MAE (m)': result['metrics']['MAE'],
            'R²': result['metrics']['R2'],
            'Samples': result['samples']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    comp_file = os.path.join(RESULTS_PATH, f'model_comparison_{zone_name}.csv')
    df_comparison.to_csv(comp_file, index=False)
    
    metrics_summary_file = os.path.join(METRICS_PATH, f'all_models_summary_{zone_name}.csv')
    df_comparison.to_csv(metrics_summary_file, index=False)
    
    print(f"\n📊 MODEL COMPARISON:")
    print(df_comparison.to_string(index=False))
    print(f"\n✓ Comparison saved: {comp_file}")
    print(f"✓ Metrics summary saved: {metrics_summary_file}")
    
    # Create advanced comparison visualization
    create_advanced_comparison_visualization(df_comparison, zone_name)
    
    return True

def create_advanced_comparison_visualization(df_comparison, zone_name):
    """Creates a publication-quality 4-panel multi-model comparison figure."""
    _pub_style()
    try:
        models_sorted = df_comparison.sort_values('RMSE (m)')['Model'].tolist()
        colors_list   = [_MODEL_COLORS.get(m, '#888888') for m in models_sorted]
        df_s = df_comparison.sort_values('RMSE (m)').reset_index(drop=True)

        fig = plt.figure(figsize=(14, 11))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.36)

        # (a) Grouped bar: RMSE & MAE
        ax1 = fig.add_subplot(gs[0, 0])
        x  = np.arange(len(df_s)); w = 0.33
        ax1.bar(x-w/2, df_s['RMSE (m)'], w, label='RMSE',
                color=colors_list, alpha=0.85, edgecolor='white', lw=0.6)
        ax1.bar(x+w/2, df_s['MAE (m)'],  w, label='MAE',
                color=colors_list, alpha=0.50, edgecolor='white', lw=0.6, hatch='//')
        for xi,(rmse,mae) in enumerate(zip(df_s['RMSE (m)'],df_s['MAE (m)'])):
            ax1.text(xi-w/2, rmse+0.05, f'{rmse:.2f}', ha='center', va='bottom', fontsize=7.5)
            ax1.text(xi+w/2, mae +0.05, f'{mae:.2f}',  ha='center', va='bottom', fontsize=7.5)
        ax1.set_xticks(x); ax1.set_xticklabels(df_s['Model'], rotation=30, ha='right')
        ax1.set_ylabel('Error (m)'); ax1.set_title('(a) RMSE and MAE per model')
        solid = mpatches.Patch(color='grey', alpha=0.85, label='RMSE')
        hatch = mpatches.Patch(facecolor='grey', alpha=0.50, hatch='//', label='MAE')
        ax1.legend(handles=[solid,hatch], fontsize=9)
        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

        # (b) R² bar
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(x, df_s['R²'], color=colors_list, alpha=0.85, edgecolor='white', lw=0.6)
        for xi,r2 in enumerate(df_s['R²']):
            ax2.text(xi, r2+0.001, f'{r2:.4f}', ha='center', va='bottom', fontsize=7.5)
        ax2.set_xticks(x); ax2.set_xticklabels(df_s['Model'], rotation=30, ha='right')
        ax2.set_ylabel('R²'); ax2.set_ylim(0.85, 1.0)
        ax2.set_title('(b) Coefficient of determination (R²)')
        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

        # (c) Scatter RMSE vs MAE (bubble = R²)
        ax3 = fig.add_subplot(gs[1, 0])
        for _, row in df_s.iterrows():
            m = row['Model']; c = _MODEL_COLORS.get(m, '#888')
            ax3.scatter(row['RMSE (m)'], row['MAE (m)'],
                        s=(row['R²']-0.85)*4000+120,
                        color=c, alpha=0.82, edgecolors='white', lw=0.8,
                        marker=_MODEL_MARKERS.get(m,'o'),
                        label=f"{m}  R²={row['R²']:.4f}", zorder=3)
        ax3.set_xlabel('RMSE (m)'); ax3.set_ylabel('MAE (m)')
        ax3.set_title('(c) RMSE vs MAE (bubble ∝ R²)')
        ax3.legend(fontsize=8.5, framealpha=0.9)
        ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax3.yaxis.set_minor_locator(AutoMinorLocator(2))

        # (d) Normalised radar chart
        ax4 = fig.add_subplot(gs[1, 1], polar=True)
        radar_lbls = ['RMSE\n(inv.)', 'MAE\n(inv.)', 'R²', 'Stability']
        N = len(radar_lbls)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
        ax4.set_theta_offset(np.pi/2); ax4.set_theta_direction(-1)
        ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(radar_lbls, fontsize=9)
        ax4.set_ylim(0, 1); ax4.set_yticks([0.25,0.5,0.75,1.0])
        ax4.set_yticklabels(['0.25','0.50','0.75','1.0'], fontsize=7)
        rmse_max = df_s['RMSE (m)'].max(); mae_max = df_s['MAE (m)'].max()
        for _, row in df_s.iterrows():
            m = row['Model']; c = _MODEL_COLORS.get(m, '#888')
            vals = [1-row['RMSE (m)']/rmse_max,
                    1-row['MAE (m)']/mae_max,
                    row['R²'],
                    0.5*(1-row['RMSE (m)']/rmse_max)+0.5*row['R²']]
            vals += vals[:1]
            ax4.plot(angles, vals, color=c, lw=1.6,
                     marker=_MODEL_MARKERS.get(m,'o'), ms=5, label=m)
            ax4.fill(angles, vals, color=c, alpha=0.08)
        ax4.set_title('(d) Normalised radar', fontsize=11, pad=14)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.35,1.15), fontsize=8.5)

        fig.suptitle(f'COMPARATIVE ANALYSIS — SDB MODELS — Zone: {zone_name}',
                     fontsize=13, fontweight='bold', y=0.98)
        comp_fig = os.path.join(FIG_PATH, f'advanced_comparison_{zone_name}.png')
        plt.savefig(comp_fig, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Advanced comparative visualization saved: {comp_fig}")

        best = df_s.iloc[0]
        print(f"\n🎯 BEST MODEL: {best['Model']}  RMSE={best['RMSE (m)']:.3f} m  R²={best['R²']:.4f}")
        return True
    except Exception as e:
        print(f"⚠  Error comparison viz: {e}")
        import traceback; traceback.print_exc()
        return False

def run_complete_pipeline(args):
    """Runs all phases for all models in a single command."""
    print("\n" + "="*80)
    print("🚀 COMPLETE PIPELINE EXECUTION FOR ALL MODELS")
    print("="*80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not args.rgb_file or not args.depth_file:
        print("❌ Please specify --rgb and --depth")
        return False
    
    zone_name = extract_zone_name(args.rgb_file)
    print(f"📍 Study area: {zone_name}")
    
    all_models = ['sdb_cnn', 'pi_cnn', 'cnn_bilstm', 'fcnn', 'seabed_net']
    
    # Step 1: Data collection
    print(f"\n{'='*80}")
    print("STEP 1: DATA COLLECTION")
    print(f"{'='*80}")
    
    if not collect_data(args):
        print("❌ Data collection failed")
        return False
    
    # Step 2: Training
    print(f"\n{'='*80}")
    print("STEP 2: TRAINING ALL MODELS")
    print(f"{'='*80}")
    
    for model_type in all_models:
        print(f"\n{'─'*60}")
        print(f"TRAINING: {model_type.upper()}")
        print(f"{'─'*60}")
        
        args.model_type = model_type
        if not train_model(args):
            print(f"⚠  Training failed for {model_type}, continuing...")
    
    # Step 3: Testing
    print(f"\n{'='*80}")
    print("STEP 3: TESTING ALL MODELS")
    print(f"{'='*80}")
    
    for model_type in all_models:
        print(f"\n{'─'*60}")
        print(f"TESTING: {model_type.upper()}")
        print(f"{'─'*60}")
        
        args.model_type = model_type
        if not test_model(args):
            print(f"⚠  Testing failed for {model_type}, continuing...")
    
    # Step 4: Map generation
    print(f"\n{'='*80}")
    print("STEP 4: MAP GENERATION")
    print(f"{'='*80}")
    
    for model_type in all_models:
        print(f"\n{'─'*60}")
        print(f"MAP GENERATION: {model_type.upper()}")
        print(f"{'─'*60}")
        
        args.model_type = model_type
        if not generate_map(args):
            print(f"⚠  Map generation failed for {model_type}, continuing...")
    
    # Step 5: Comparison
    print(f"\n{'='*80}")
    print("STEP 5: COMPARISON OF ALL MODELS")
    print(f"{'='*80}")
    
    if not compare_all_models(args):
        print("⚠  Model comparison failed")
    
    # Step 6: Final report
    print(f"\n{'='*80}")
    print("STEP 6: FINAL REPORT")
    print(f"{'='*80}")
    
    generate_final_report(zone_name)
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print(f"{'='*80}")
    
    return True

def generate_final_report(zone_name):
    """Generates final report with all metrics."""
    print(f"\n📋 Generating final report...")
    
    metrics_patterns = [
        os.path.join(METRICS_PATH, f"metrics_*_{zone_name}.csv"),
        os.path.join(METRICS_PATH, f"*{zone_name}*.csv"),
        os.path.join(RESULTS_PATH, f"*{zone_name}*.csv"),
        os.path.join(METRICS_PATH, f"all_models_summary_{zone_name}.csv")
    ]
    
    metrics_files = []
    for pattern in metrics_patterns:
        metrics_files.extend(glob.glob(pattern))
    
    metrics_files = list(set(metrics_files))
    
    if not metrics_files:
        print(f"❌ No metrics files found")
        return False
    
    print(f"   ✓ Metrics files found: {len(metrics_files)}")
    
    all_metrics = []
    for file in metrics_files:
        try:
            df = pd.read_csv(file)
            required_cols = ['model', 'mae', 'rmse', 'r2']
            if all(col in df.columns for col in required_cols):
                all_metrics.append(df)
                print(f"   • Loaded: {os.path.basename(file)} ({len(df)} rows)")
        except Exception as e:
            print(f"   ⚠  Error with {os.path.basename(file)}: {e}")
            continue
    
    if not all_metrics:
        print("❌ No valid metrics found")
        return False
    
    combined_df = pd.concat(all_metrics, ignore_index=True).drop_duplicates()
    
    report_file = os.path.join(METRICS_PATH, f'final_report_{zone_name}.csv')
    combined_df.to_csv(report_file, index=False)
    
    text_report = f"""
    ============================================================================
    FINAL REPORT - SDB MODEL ANALYSIS
    Zone: {zone_name}
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ============================================================================
    
    PERFORMANCE SUMMARY:
    """
    
    for model_type in combined_df['model'].unique():
        model_df = combined_df[combined_df['model'] == model_type]
        if len(model_df) > 0:
            best_model = model_df.loc[model_df['rmse'].idxmin()]
            text_report += f"""
    {model_type.upper()}:
      • MAE:   {best_model['mae']:.3f} m
      • RMSE:  {best_model['rmse']:.3f} m
      • R²:    {best_model['r2']:.3f}
      • Samples: {best_model.get('samples', 'N/A')}
            """
    
    if len(combined_df) > 0:
        best_rmse = combined_df.loc[combined_df['rmse'].idxmin()]
        best_mae = combined_df.loc[combined_df['mae'].idxmin()]
        best_r2 = combined_df.loc[combined_df['r2'].idxmax()]
        
        text_report += f"""
    ============================================================================
    RECOMMENDATIONS:
    
    Best model for RMSE: {best_rmse['model'].upper()} ({best_rmse['rmse']:.3f} m)
    Best model for MAE:  {best_mae['model'].upper()} ({best_mae['mae']:.3f} m)
    Best model for R²:   {best_r2['model'].upper()} ({best_r2['r2']:.3f})
    
    For production, use: {best_rmse['model'].upper()} (best overall accuracy)
    ============================================================================
    
    AVAILABLE VISUALIZATIONS:
    • 2D bathymetric maps: /figs/map_comparison_*.png
    • 3D visualizations: /visualizations_3d/3d_visualization_*.png
    • Analysis dashboards: /figs/dashboard_*.png
    • Advanced comparisons: /figs/advanced_comparison_*.png
    • Depth profiles: /figs/depth_profiles_*.png
    """
    
    text_report_file = os.path.join(METRICS_PATH, f'final_report_{zone_name}.txt')
    with open(text_report_file, 'w', encoding='utf-8') as f:
        f.write(text_report)
    
    print(f"   ✓ Final report saved: {report_file}")
    print(f"   ✓ Text report saved: {text_report_file}")
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    summary_cols = ['model', 'mae', 'rmse', 'r2']
    if 'samples' in combined_df.columns:
        summary_cols.append('samples')
    
    summary_df = combined_df.loc[combined_df.groupby('model')['rmse'].idxmin()]
    print(summary_df[summary_cols].to_string(index=False))
    
    return True

# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description='SDB pipeline with advanced and 3D visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:

# Run complete pipeline in one command (RECOMMENDED)
python SDB_Advanced_Visualization.py --complete --rgb ANZA_2021_rgb.tif --depth ANZA_2021_crt.tif

# Data collection
python SDB_Advanced_Visualization.py --collect --rgb ANZA_2021_rgb.tif --depth ANZA_2021_crt.tif

# Training specific model
python SDB_Advanced_Visualization.py --train --model pi_cnn --rgb ANZA_2021_rgb.tif

# Testing with advanced visualizations
python SDB_Advanced_Visualization.py --test --model pi_cnn --rgb ANZA_2021_rgb.tif

# Map generation with 3D visualizations
python SDB_Advanced_Visualization.py --generate --model pi_cnn --rgb ANZA_2021_rgb.tif --depth ANZA_2021_crt.tif

# Detailed evaluation with dashboard
python SDB_Advanced_Visualization.py --evaluate --model pi_cnn --rgb ANZA_2021_rgb.tif

# Comparison of all models
python SDB_Advanced_Visualization.py --compare --rgb ANZA_2021_rgb.tif
        """
    )
    
    parser.add_argument('--rgb', dest='rgb_file',
                       help='RGB image file (e.g., ANZA_2021_rgb.tif)')
    parser.add_argument('--depth', dest='depth_file',
                       help='Bathymetry file (e.g., ANZA_2021_crt.tif)')
    
    parser.add_argument('--model', dest='model_type',
                       choices=['sdb_cnn', 'pi_cnn', 'cnn_bilstm', 'fcnn', 'seabed_net'],
                       default='sdb_cnn',
                       help='Model type to use')
    
    parser.add_argument('--collect', action='store_true',
                       help='Phase 1: Data collection')
    parser.add_argument('--train', action='store_true',
                       help='Phase 2: Model training')
    parser.add_argument('--train_all', action='store_true',
                       help='Train all models')
    parser.add_argument('--test', action='store_true',
                       help='Phase 3: Model testing')
    parser.add_argument('--generate', action='store_true',
                       help='Phase 4: Map generation')
    parser.add_argument('--evaluate', action='store_true',
                       help='Phase 5: Detailed evaluation')
    parser.add_argument('--compare', action='store_true',
                       help='Phase 6: Comparison of all models')
    parser.add_argument('--all', action='store_true',
                       help='Run all phases for a model')
    parser.add_argument('--complete', action='store_true',
                       help='Run complete pipeline for all models')
    
    parser.add_argument('--window', type=int, default=WINDOW,
                       help=f'Window size (default: {WINDOW})')
    parser.add_argument('--stride', type=int, default=STRIDE,
                       help=f'Stride (default: {STRIDE})')
    parser.add_argument('--channels', type=int, default=CHANNELS,
                       help=f'Number of channels (default: {CHANNELS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help=f'Number of epochs (default: {EPOCHS})')
    parser.add_argument('--dropout', type=float, default=DROPOUT_RATE,
                       help=f'Dropout rate (default: {DROPOUT_RATE})')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                       help=f'Early stopping patience (default: {PATIENCE})')
    
    args = parser.parse_args()
    
    if not any([args.collect, args.train, args.train_all, args.test, 
                args.generate, args.evaluate, args.compare, args.all, args.complete]):
        parser.print_help()
        print("\n❌ Specify a phase to execute.")
        return
    
    print("\n" + "="*80)
    print("🚀 SDB PIPELINE - ADVANCED AND 3D VISUALIZATIONS")
    print("="*80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.model_type and not args.compare and not args.complete:
        model_names = {
            'sdb_cnn': 'SDB-CNN (Original)',
            'pi_cnn': 'PI-CNN (Physics-Informed)',
            'cnn_bilstm': 'BOA-CNN-BiLSTM',
            'fcnn': 'FCNN (Fully Convolutional)',
            'seabed_net': 'Seabed-Net (Multi-task)'
        }
        print(f"🧠 Selected model: {model_names[args.model_type]}")
    
    if args.rgb_file:
        zone_name = extract_zone_name(args.rgb_file)
        print(f"📍 Study area: {zone_name}")
    
    success = True
    
    if args.complete:
        if not run_complete_pipeline(args):
            success = False
    else:
        if args.collect or args.all:
            if not collect_data(args):
                success = False
                if not args.all:
                    return
        
        if args.train or args.all:
            if success or args.train:
                if not train_model(args):
                    success = False
                    if not args.all:
                        return
        
        if args.train_all:
            original_model = args.model_type
            for model_type in ['sdb_cnn', 'pi_cnn', 'cnn_bilstm', 'fcnn', 'seabed_net']:
                args.model_type = model_type
                print(f"\n{'='*80}")
                print(f"TRAINING MODEL: {model_type.upper()}")
                print(f"{'='*80}")
                train_model(args)
            args.model_type = original_model
        
        if args.test or args.all:
            if success or args.test:
                if not test_model(args):
                    success = False
                    if not args.all:
                        return
        
        if args.generate or args.all:
            if success or args.generate:
                if not generate_map(args):
                    success = False
                    if not args.all:
                        return
        
        if args.evaluate or args.all:
            if success or args.evaluate:
                if not evaluate_model(args):
                    success = False
                    if not args.all:
                        return
        
        if args.compare:
            if not compare_all_models(args):
                success = False
    
    print("\n" + "="*80)
    if success:
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        print("⚠  PIPELINE COMPLETED WITH ERRORS")
    print("="*80)

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()