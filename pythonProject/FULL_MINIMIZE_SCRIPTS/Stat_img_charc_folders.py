"""
===============================================
Microstructure Properties Processing Module
===============================================

Description:
------------
This module processes CSV files containing microstructure properties.
(Use after script ImgProcessing.py)
It provides functionality for:

1. Reading CSV files with filtering of missing and infinite values.
2. Data processing:
    - filtering by Shape Factor,
    - normalizing Orientation values (mod π),
    - removing outliers using 6*MAD threshold.
3. Computing statistical characteristics:
    - regular statistics (mean, std, median, mode, Q1, Q3, IQR, range),
    - circular statistics for Orientation (circmean, circstd, median, Q1, Q3, IQR, range).
4. Plotting:
    - histograms for numerical columns,
    - polar histogram for Orientation with mean, median, Q1, Q3 indicators.
5. Saving results:
    - processed DataFrame,
    - statistics table,
    - plots in a separate output folder.

Libraries used:
---------------
- os
- pandas
- numpy
- matplotlib
- scipy (stats, circmean, circstd)

Example usage:
--------------
"""

import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import circmean, circstd

input_properties = [
    'Area', 'Shape Factor', 'ECR', 'Scale Factor', 'Aspect Ratio',
    'Compactness Ratio', 'area-to-ellipse Ratio', 'Orientation',
    'Inertia Tensor XX', 'Inertia Tensor XY', 'Inertia Tensor YY',
    'Inertia Tensor/Area XX', 'Inertia Tensor/Area XY', 'Inertia Tensor/Area YY'
]

derived_properties = [
    'I_Principal_Max', 'I_Principal_Min', 'I_Anisotropy',
    'I_Area_Principal_Max', 'I_Area_Principal_Min', 'I_Area_Anisotropy'
]

# Повний список для статистики та графіків
all_properties = input_properties + derived_properties


def calculate_principal_moments(df):
    """
    Обчислює власні значення тензора інерції (I1, I2) та їх відношення.
    """

    def get_eigenvalues(xx, yy, xy):
        # Формула власних значень для симетричної матриці 2x2
        avg = (xx + yy) / 2
        diff = (xx - yy) / 2
        radius = np.sqrt(diff ** 2 + xy ** 2)
        l1 = avg + radius  # Max
        l2 = avg - radius  # Min
        return l1, l2

    # Для звичайного тензора
    if 'Inertia Tensor XX' in df.columns and 'Inertia Tensor YY' in df.columns:
        l1, l2 = get_eigenvalues(df['Inertia Tensor XX'], df['Inertia Tensor YY'], df['Inertia Tensor XY'])
        df['I_Principal_Max'] = l1
        df['I_Principal_Min'] = l2
        df['I_Anisotropy'] = l1 / l2.replace(0, np.nan)  # Відношення осей (анізотропія розподілу маси)

    # Для тензора, нормованого на площу
    if 'Inertia Tensor/Area XX' in df.columns and 'Inertia Tensor/Area YY' in df.columns:
        l1_a, l2_a = get_eigenvalues(df['Inertia Tensor/Area XX'], df['Inertia Tensor/Area YY'],
                                     df['Inertia Tensor/Area XY'])
        df['I_Area_Principal_Max'] = l1_a
        df['I_Area_Principal_Min'] = l2_a
        df['I_Area_Anisotropy'] = l1_a / l2_a.replace(0, np.nan)

    return df


# ===== CSV =====
def read_csv(filename):
    """
    Read a CSV file and return a DataFrame with selected properties.

    Parameters
    ----------
    filename : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing only specified properties with NaNs removed.
        Returns None if reading fails.
    """
    try:
        df = pd.read_csv(filename, usecols=input_properties)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        # Одразу додаємо розраховані значення
        df = calculate_principal_moments(df)
        return df
    except Exception as e:
        print(f"[{filename}] Error reading file: {e}")
        return None


# ===== Circular utils =====
def circular_median(data):
    """
    Compute circular median of angles in radians.

    Parameters
    ----------
    data : array-like
        Array of angles in radians.

    Returns
    -------
    float
        Circular median (wrapped between 0 and 2*pi).
    """
    return np.mod(np.median(np.unwrap(data)), 2 * np.pi)


def circular_quantiles(data, q=[0.25, 0.5, 0.75]):
    """
    Compute circular quantiles of angles.

    Parameters
    ----------
    data : array-like
        Array of angles in radians.
    q : list of float, optional
        Quantiles to compute (default: [0.25, 0.5, 0.75]).

    Returns
    -------
    np.ndarray
        Circular quantiles (wrapped between 0 and 2*pi).
    """
    return np.mod(np.quantile(np.unwrap(data), q), 2 * np.pi)


def compute_circular_statistics(data):
    """
    Compute circular statistics for an array of angles.

    Parameters
    ----------
    data : array-like
        Array of angles in radians.

    Returns
    -------
    dict
        Dictionary containing mean, std, median, Q1, Q3, IQR, and range.
    """
    mean_circ = circmean(data, high=np.pi, low=0)
    std_circ = circstd(data, high=np.pi, low=0)
    q1, median, q3 = circular_quantiles(data)
    iqr = (q3 - q1) % (2 * np.pi)
    data_range = (np.max(np.unwrap(data)) - np.min(np.unwrap(data))) % (2 * np.pi)

    return {
        'Circular Mean': mean_circ,
        'Circular Std': std_circ,
        'Circular Median': median,
        'Circular Q1': q1,
        'Circular Q3': q3,
        'Circular IQR': iqr,
        'Circular Range': data_range
    }


# ===== Regular stats =====
def compute_regular_statistics(data):
    """
    Compute standard descriptive statistics for numeric data.

    Parameters
    ----------
    data : array-like
        Array of numeric values.

    Returns
    -------
    dict
        Dictionary with mean, std, median, mode, Q1, Q3, IQR, and range.
    """
    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)
    mode_val = float(stats.mode(data, keepdims=True)[0])
    q1, q3 = np.percentile(data, [25, 75])

    return {
        'Mean': mean,
        'Std': std,
        'Median': median,
        'Mode': mode_val,
        'Q1': q1,
        'Q3': q3,
        'IQR': q3 - q1,
        'Range': np.ptp(data)
    }


# ===== Data processing =====
def filter_shape_factor(df):
    """
    Filter DataFrame rows by Shape Factor <= 1.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    return df[df['Shape Factor'] <= 1]


def mod_orientation(df):
    """
    Apply modulo pi to Orientation column to wrap angles.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'Orientation' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with wrapped Orientation.
    """
    df['Orientation'] = np.mod(df['Orientation'], np.pi)
    return df


def threshold_mask(df, cols):
    """
    Compute boolean mask for rows within threshold (6*MAD) from median.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : list of str
        Columns to apply thresholding.

    Returns
    -------
    pd.Series
        Boolean mask indicating valid rows.
    """
    mask = pd.Series(True, index=df.index)
    for col in cols:
        med = df[col].median()
        mad = (df[col] - med).abs().median()
        mask &= (df[col] - med).abs() <= 6 * mad
    return mask


def process_data(df):
    """
    Process DataFrame: filter shape factor, wrap Orientation, apply MAD threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame.
    """
    df = df.copy()
    df = filter_shape_factor(df)
    df = mod_orientation(df)
    cols = [c for c in df.columns if c != 'Orientation']
    return df[threshold_mask(df, cols)]


# ===== Stats DF =====
def generate_statistics_df(df):
    """
    Generate a DataFrame of statistics for all properties.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Table with columns ['Property', 'Statistic', 'Value'].
    """
    rows = []
    for prop in all_properties:
        data = df[prop].values
        stats_dict = compute_circular_statistics(data) if prop == "Orientation" else compute_regular_statistics(data)
        rows.extend(format_stats_rows(prop, stats_dict))
    return pd.DataFrame(rows, columns=["Property", "Statistic", "Value"])


def format_stats_rows(prop, stats_dict):
    """
    Convert statistics dictionary to table rows.

    Parameters
    ----------
    prop : str
        Name of the property.
    stats_dict : dict
        Dictionary of statistics.

    Returns
    -------
    list of lists
        Table rows: [Property, Statistic, Value]
    """
    first = True
    rows = []
    for stat, val in stats_dict.items():
        rows.append([prop if first else '', stat, f"{val:.4f}"])
        first = False
    return rows


# ===== Histograms =====
def safe_filename(name: str) -> str:
    """
    Convert column name to a filesystem-safe filename.
    """
    name = name.strip()
    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace(":", "_")
    name = name.replace("-", "_")
    name = re.sub(r"__+", "_", name)  # прибрати подвійні _
    return name


def plot_regular_histogram(df_before, df_after, col, output_dir):
    """
    Plot overlaid histogram before and after processing for a numeric column.

    Parameters
    ----------
    df_before : pd.DataFrame
        Original DataFrame.
    df_after : pd.DataFrame
        Processed DataFrame.
    col : str
        Column name to plot.
    output_dir : str
        Directory to save figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    safe_col = safe_filename(col)

    data_all = np.concatenate([df_before[col], df_after[col]])
    bins = np.linspace(
        data_all.min(),
        data_all.max(),
        int(np.sqrt(len(data_all)))
    )

    plt.figure(figsize=(10, 5))
    plt.hist(df_before[col], bins=bins, alpha=0.7, label='Before')
    plt.hist(df_after[col], bins=bins, alpha=0.5, label='After')
    plt.title(f'Histogram: {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{safe_col}_hist.png"),
        dpi=300
    )
    plt.close()


def plot_orientation_polar(df_before, df_after, output_dir):
    """
    Plot polar histogram for Orientation before and after processing.

    Parameters
    ----------
    df_before : pd.DataFrame
        Original DataFrame.
    df_after : pd.DataFrame
        Processed DataFrame.
    output_dir : str
        Directory to save figure.
    """
    data_before = np.mod(df_before['Orientation'].values, np.pi)
    data_after = np.mod(df_after['Orientation'].values, np.pi)
    bins_before = np.linspace(0, np.pi, max(10, len(data_before) // 5) + 1)
    bins_after = np.linspace(0, np.pi, max(10, len(data_after) // 5) + 1)
    plot_polar(data_before, data_after, bins_before, bins_after, output_dir)


def plot_polar(data_before, data_after, bins_before, bins_after, output_dir):
    """
    Plot polar histogram using precomputed bins.

    Parameters
    ----------
    data_before, data_after : array-like
        Angles in radians.
    bins_before, bins_after : array-like
        Bin edges for histograms.
    output_dir : str
        Directory to save figure.
    """
    hist_before, hist_after, edges_before, edges_after = prepare_hist(data_before, data_after, bins_before, bins_after)
    theta_before, theta_after, width_before, width_after, r_max, eps = prepare_polar_params(
        hist_before, hist_after, edges_before, edges_after
    )

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, projection='polar')
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.bar(theta_before, hist_before, width=width_before, alpha=0.5, color='teal', label='Before')
    ax.bar(theta_after, hist_after, width=width_after, alpha=0.5, color='coral', label='After')
    plot_polar_lines(ax, data_after, eps, r_max)
    ax.set_xticks(np.linspace(0, np.pi, 5))
    ax.set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    # ax.set_yscale('log')
    ax.set_ylim(eps, r_max)
    ax.set_title('Polar Histogram: Orientation')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Orientation_polar.png"))
    plt.close()


def prepare_hist(data_before, data_after, bins_before, bins_after):
    """
    Compute histogram counts and edges for polar plot.

    Returns
    -------
    tuple
        hist_before, hist_after, edges_before, edges_after
    """
    hist_before, edges_before = np.histogram(data_before, bins=bins_before)
    hist_after, edges_after = np.histogram(data_after, bins=bins_after)
    epsilon = 1e-2
    hist_before = np.maximum(hist_before.astype(float), epsilon)
    hist_after = np.maximum(hist_after.astype(float), epsilon)
    return hist_before, hist_after, edges_before, edges_after


def prepare_polar_params(hist_before, hist_after, edges_before, edges_after):
    """
    Compute polar plot parameters: theta, width, r_max, epsilon.

    Returns
    -------
    tuple
        theta_before, theta_after, width_before, width_after, r_max, eps
    """
    theta_before = (edges_before[:-1] + edges_before[1:]) / 2
    theta_after = (edges_after[:-1] + edges_after[1:]) / 2
    width_before = np.diff(edges_before)
    width_after = np.diff(edges_after)
    r_max = max(hist_before.max(), hist_after.max()) * 1.1
    eps = 1e-2
    return theta_before, theta_after, width_before, width_after, r_max, eps


def plot_polar_lines(ax, data, eps, r_max):
    """
    Plot circular statistics lines on polar histogram.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        Polar axis to plot lines on.
    data : array-like
        Angle values.
    eps : float
        Minimum radius.
    r_max : float
        Maximum radius.
    """
    stats = compute_circular_statistics(data)
    ax.plot([stats['Circular Mean']] * 2, [eps, r_max], color='yellow', linewidth=2, label='Mean')
    ax.plot([stats['Circular Median']] * 2, [eps, r_max], color='green', linewidth=2, label='Median')
    ax.plot([stats['Circular Q1']] * 2, [eps, r_max], color='orange', linestyle='--', linewidth=1.5, label='Q1')
    ax.plot([stats['Circular Q3']] * 2, [eps, r_max], color='purple', linestyle='--', linewidth=1.5, label='Q3')


def plot_histograms(df_before, df_after, output_dir):
    """
    Plot histograms for all properties, orientation as polar plot.

    Parameters
    ----------
    df_before : pd.DataFrame
        Original data.
    df_after : pd.DataFrame
        Processed data.
    output_dir : str
        Directory to save plots.
    """
    for col in all_properties:
        if col == "Orientation":
            plot_orientation_polar(df_before, df_after, output_dir)
        else:
            plot_regular_histogram(df_before, df_after, col, output_dir)


# ===== Folder processing =====
def process_folder(folder_path):
    """
    Process all CSV files in folder.

    Parameters
    ----------
    folder_path : str
        Path to folder containing CSV files.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not csv_files:
        print(f"[{folder_path}] CSV file not found.")
        return
    for csv_file in csv_files:
        process_file(folder_path, csv_file)


def process_file(folder_path, csv_file):
    """
    Process single CSV file: read, clean, stats, save, plot.

    Parameters
    ----------
    folder_path : str
        Folder containing CSV.
    csv_file : str
        CSV file name.
    """
    full_path = os.path.join(folder_path, csv_file)
    df_raw = read_csv(full_path)
    if df_raw is None: return

    df_processed = process_data(df_raw.copy())
    stats_df = generate_statistics_df(df_processed)

    output_dir = os.path.join(folder_path, 'processed_output')
    os.makedirs(output_dir, exist_ok=True)

    df_processed.to_csv(os.path.join(output_dir, f"processed_{csv_file}"), index=False)
    stats_df.to_csv(os.path.join(output_dir, f"statistics_{csv_file}"), index=False)
    plot_histograms(df_raw, df_processed, output_dir)
    print(f"File '{csv_file}' processed. Saved to: {output_dir}")


# ===== Main =====
folders_to_process = [
    # r"D:\\Універ\\ДИПЛОМ\\TEST_alg\\optimize_models_update\\AZ31_iA"
    r"D:\University\MatViz\1FULL_MINIMIZE_SCRIPTS\AZ31_iA"
]

for folder in folders_to_process:
    process_folder(folder)
