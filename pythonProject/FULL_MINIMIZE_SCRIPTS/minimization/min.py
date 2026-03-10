import csv
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import count
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from scipy.optimize import (
    minimize,
    Bounds,
    basinhopping,
    differential_evolution,
    dual_annealing
)
from scipy.spatial.distance import pdist, cdist
from scipy.stats import circmean, circstd
from skimage.measure import regionprops

from MatViz3DLauncher import MatViz3DLauncher

# TODO:  Метод оптимізації, ('SLSQP', 'L-BFGS-B', 'Dual Annealing', 'basinhopping', 'Differential Evolution', 'Manual Sweep')
selected_method = 'Dual Annealing'
selected_metric_type = 'Energy Distance'  # або 'MSE' 'SMAPE' 'MSPE' 'Energy Distance'

# TODO:   Глобальні змінні
selected_features = [
    'Inertia Tensor/Area XX', 'Inertia Tensor/Area XY', 'Inertia Tensor/Area YY'
]

# selected_features = [
#     'Inertia Tensor/Area XX', 'Inertia Tensor/Area YY'
# ]

# selected_features = [
#     'ECR', 'scale_factor', 'Compactness Ratio',
#     'Inertia Tensor XX', 'Inertia Tensor XY', 'Inertia Tensor YY',
#     'Inertia Tensor/Area XX', 'Inertia Tensor/Area XY', 'Inertia Tensor/Area YY',
#     'I_Principal_Max', 'I_Principal_Min', 'I_Anisotropy',
#     'I_Area_Principal_Max', 'I_Area_Principal_Min', 'I_Area_Anisotropy'
# ]

# selected_metrics = ['Mean', 'Std', 'Median', 'Q1', 'Q3']
selected_metrics = ['Mean', 'Q1', 'Q3']

# TODO: Фіксований розмір куба
FIXED_SIZE = 130

# TODO: Обмеження
bounds = [(0.01, 0.2), (0.65, 1.5), (0.65, 1.5), (1, 6), (1, 60), (0.1, 50), (0, 1000)]

base_params = np.array([
    0.1,  # concentration
    1.5,  # halfaxis_b
    1.5,  # halfaxis_c
    1.7,  # ellipse_order
    15.0,  # wave_coefficient
    2.0,  # wave_spread
    1  # initial_nuclei_count
], dtype=float)

# TODO: Ініціалізація запуска MatViz3D
exe_path = r".\debug\MatViz3D.exe"
launcher = MatViz3DLauncher(exe_path)

# TODO:  Створення каталогу для виведення
# output_folder = r"D:\Project(MatViz3D)\Random\Paper_Optimisation\test\iter\EnergyDistance_norm_allTensor_LOG"
output_folder = r".\test"
os.makedirs(output_folder, exist_ok=True)

# TODO: Опції аналізу шарів
# mode: 'all' - аналіз усіх шарів (поточна поведінка)
#       'sample' - аналізувати 1 шар, пропустити SAMPLE_SKIP шарів
#       'single' - аналізувати тільки один шар із індексом SINGLE_LAYER_INDEX
ANALYSIS_MODE = 'sample'  # 'all', 'sample', 'single'
SAMPLE_SKIP = 4  # якщо mode == 'sample', то після кожного проаналізованого шару пропускаємо SAMPLE_SKIP шарів
SAMPLE_OFFSET = 0  # початковий індекс для sample (0..cube_size-1)
SINGLE_LAYER_INDEX = 0  # індекс шару для mode == 'single'

# TODO: Обрати осі для розрізання
# ['x'], ['y'], ['z'], ['x', 'y'], ['x', 'z'], ['y', 'z'], або ['x', 'y', 'z']
SELECTED_AXES = ['x']  # за замовчуванням — тільки по осі Z

# лічильник ітерацій для назв файлів
ITER_COUNTER = count(1)

# True = працюємо в логарифмічному просторі (Log-Normal distribution)
# False = звичайний лінійний простір
USE_LOG_SPACE = False

# Шляхи до файлів з цільовими значеннями
TARGET_FILE_NORMAL = r".\FULL_MINIMIZE_SCRIPTS\AZ31_iA\processed_output_Arcsinh\statistics_image_properties_(AZ31_imgA).csv"

TARGET_FILE_LOG = r".\FULL_MINIMIZE_SCRIPTS\AZ31_iA\processed_output_Arcsinh\Arcsinh_statistics_image_properties_(AZ31_imgA).csv"

TARGET_FILE_DIST = r".\FULL_MINIMIZE_SCRIPTS\AZ31_iA\processed_output_Arcsinh\processed_image_properties_(AZ31_imgA).csv"

TARGET_FILE_DIST_LOG = r".\FULL_MINIMIZE_SCRIPTS\AZ31_iA\processed_output_Arcsinh\processed_Arcsinh_image_properties_(AZ31_imgA).csv"

if selected_metric_type == 'Energy Distance':
    target_csv_path = TARGET_FILE_DIST_LOG if USE_LOG_SPACE else TARGET_FILE_DIST
    print(f"Обрана метрика: Energy Distance. Завантажуємо повний розподіл.")
else:
    target_csv_path = TARGET_FILE_LOG if USE_LOG_SPACE else TARGET_FILE_NORMAL
    print(f"Обрана метрика: {selected_metric_type}. Завантажуємо статистику.")

print(f"Цільовий файл: {target_csv_path}")

# ------------------------
inertia_needed = {
    'Inertia Tensor XX', 'Inertia Tensor XY', 'Inertia Tensor YY',
    'Inertia Tensor/Area XX', 'Inertia Tensor/Area XY', 'Inertia Tensor/Area YY'
}

principal_needed = {
    'I_Principal_Max', 'I_Principal_Min', 'I_Anisotropy',
    'I_Area_Principal_Max', 'I_Area_Principal_Min', 'I_Area_Anisotropy'
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)

# CSV файл для параметрів запуску
csv_params_path = os.path.join(output_folder, "run_parameters_and_selected_features.csv")
csv_headers = [
    "Iteration",
    "wave_coefficient", "wave_spread", "initial_nuclei_count", "concentration",
    "halfaxis_a", "halfaxis_b", "halfaxis_c",
    "ellipse_order",
]

# значення selected features
for f in selected_features:
    for m in selected_metrics:
        csv_headers.append(f"{f}_{m}")

csv_headers.append("Total_Error")

if selected_metric_type == 'Energy Distance':
    for f in selected_features:
        csv_headers.append(f"ED_{f}")  # Energy Distance для кожної фічі
    csv_headers.append("Energy_Distance")  # Загальна похибка по всім фічам разом
else:
    for f in selected_features:
        for m in selected_metrics:
            csv_headers.append(f"Error_{f}_{m}")


def log_iteration_to_csv(
        iteration: int,
        params,
        stats,
        individual_errors,
        total_error
):
    file_exists = os.path.isfile(csv_params_path)

    concentration, halfaxis_b, halfaxis_c, ellipse_order, wave_coefficient, wave_spread, initial_nuclei_count = params
    halfaxis_a = halfaxis_b

    initial_nuclei_count = int(round(initial_nuclei_count))

    row = {
        "Iteration": iteration,
        "wave_coefficient": wave_coefficient,
        "wave_spread": wave_spread,
        "initial_nuclei_count": initial_nuclei_count,
        "concentration": concentration,
        "halfaxis_a": halfaxis_a,
        "halfaxis_b": halfaxis_b,
        "halfaxis_c": halfaxis_c,
        "ellipse_order": ellipse_order,
        "Total_Error": total_error
    }

    for f in selected_features:
        for m in selected_metrics:
            row[f"{f}_{m}"] = stats[m].get(f, np.nan)

    if selected_metric_type == 'Energy Distance':
        for f in selected_features:
            row[f"ED_{f}"] = individual_errors.get(f, {}).get("EnergyDist", np.nan)

        row["Energy_Distance"] = individual_errors.get("Energy_Distance", {}).get("All", np.nan)
    else:
        for f in selected_features:
            for m in selected_metrics:
                row[f"Error_{f}_{m}"] = individual_errors.get(f, {}).get(m, np.nan)

    with open(csv_params_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


param_names = [
    "concentration",
    "halfaxis_b",
    "halfaxis_c",
    "ellipse_order",
    "wave_coefficient",
    "wave_spread",
    "initial_nuclei_count"
]


def manual_parameter_sweep(
        param_index: int,
        base_params: np.ndarray,
        bounds: list,
        n_points: int = 15
):
    low, high = bounds[param_index]
    sweep_values = np.linspace(low, high, n_points)

    logging.info(
        f"Manual sweep for '{param_names[param_index]}' "
        f"from {low} to {high} ({n_points} points)"
    )

    results = []

    for i, val in enumerate(sweep_values, start=1):
        params = base_params.copy()
        params[param_index] = val

        logging.info(
            f"[SWEEP {i}/{n_points}] "
            f"{param_names[param_index]} = {val:.6f}"
        )

        try:
            error = minimize_properties(params)
            results.append((val, error))
        except Exception as e:
            logging.error(f"Sweep failed at {val}: {e}")
            results.append((val, np.nan))

    return results


def start(size, concentration, halfaxis_a, halfaxis_b, halfaxis_c, orientation_angle_a, orientation_angle_b,
          orientation_angle_c, wave_coefficient, wave_spread, initial_nuclei_count, ellipse_order, output_file):
    print(
        f"Запуск MatViz3D з параметрами: [{size}, conc = {concentration}, "
        f"h_a = {halfaxis_a}, h_b = {halfaxis_b}, h_c = {halfaxis_c}, "
        f"or_a = {orientation_angle_a}, or_b = {orientation_angle_b}, or_c = {orientation_angle_c}, "
        f"wc = {wave_coefficient}, wave_spread = {wave_spread}, initial_nuclei_count = {initial_nuclei_count}, "
        f"ellipse_order = {ellipse_order}]")

    return launcher.start(
        size=size,
        concentration=concentration,
        halfaxis_a=halfaxis_a,
        halfaxis_b=halfaxis_b,
        halfaxis_c=halfaxis_c,
        orientation_angle_a=orientation_angle_a,
        orientation_angle_b=orientation_angle_b,
        orientation_angle_c=orientation_angle_c,
        wave_coefficient=wave_coefficient,
        wave_spread=wave_spread,
        initial_nuclei_count=initial_nuclei_count,
        ellipse_order=ellipse_order,
        output_file=output_file
    )


def process_layer(index, layers, axis):
    try:
        if axis == 'z':
            layer = layers[:, :, index]
        elif axis == 'x':
            layer = layers[index, :, :]
        elif axis == 'y':
            layer = layers[:, index, :]
        else:
            raise ValueError(f"Invalid axis: {axis}")

        layer_area = np.prod(layer.shape)
        unique_colors = set(layer.flatten())
        properties = []

        for grain_color in unique_colors:
            grain_mask = (layer == grain_color)
            labeled_grains = skimage.measure.label(grain_mask, connectivity=2)
            for region in regionprops(labeled_grains):
                if region.area <= 5:
                    continue
                if np.any(region.coords[:, 0] == 0) or np.any(region.coords[:, 0] == layer.shape[0] - 1) \
                        or np.any(region.coords[:, 1] == 0) or np.any(region.coords[:, 1] == layer.shape[1] - 1):
                    continue

                area = region.area
                norm_area = area / layer_area
                props = {}

                if 'Norm Area' in selected_features:
                    props['Norm Area'] = norm_area
                if 'ECR' in selected_features:
                    props['ECR'] = np.sqrt(norm_area / np.pi)
                if 'Aspect Ratio' in selected_features or 'Compactness Ratio' in selected_features or 'scale_factor' in selected_features:
                    if region.minor_axis_length == 0:
                        continue
                    major = region.major_axis_length
                    minor = region.minor_axis_length
                    if 'Aspect Ratio' in selected_features:
                        props['Aspect Ratio'] = major / minor
                    if 'Compactness Ratio' in selected_features:
                        props['Compactness Ratio'] = region.convex_area / area
                    if 'scale_factor' in selected_features:
                        props['scale_factor'] = major / minor

                if 'Orientation' in selected_features:
                    props['Orientation'] = float(region.orientation)

                if any(f in selected_features for f in inertia_needed.union(principal_needed)):
                    it = region.inertia_tensor

                    if 'Inertia Tensor XX' in selected_features:
                        props['Inertia Tensor XX'] = it[0, 0]
                    if 'Inertia Tensor XY' in selected_features:
                        props['Inertia Tensor XY'] = it[0, 1]
                    if 'Inertia Tensor YY' in selected_features:
                        props['Inertia Tensor YY'] = it[1, 1]

                    if 'Inertia Tensor/Area XX' in selected_features:
                        props['Inertia Tensor/Area XX'] = it[0, 0] / area
                    if 'Inertia Tensor/Area XY' in selected_features:
                        props['Inertia Tensor/Area XY'] = it[0, 1] / area
                    if 'Inertia Tensor/Area YY' in selected_features:
                        props['Inertia Tensor/Area YY'] = it[1, 1] / area

                    if any(f in selected_features for f in ['I_Principal_Max', 'I_Principal_Min', 'I_Anisotropy']):
                        xx, yy, xy = it[0, 0], it[1, 1], it[0, 1]
                        l1 = (xx + yy) / 2 + np.sqrt(((xx - yy) / 2) ** 2 + xy ** 2)
                        l2 = (xx + yy) / 2 - np.sqrt(((xx - yy) / 2) ** 2 + xy ** 2)
                        if 'I_Principal_Max' in selected_features:
                            props['I_Principal_Max'] = l1
                        if 'I_Principal_Min' in selected_features:
                            props['I_Principal_Min'] = l2
                        if 'I_Anisotropy' in selected_features:
                            props['I_Anisotropy'] = l1 / max(l2, 1e-12)

                    if any(f in selected_features for f in
                           ['I_Area_Principal_Max', 'I_Area_Principal_Min', 'I_Area_Anisotropy']):
                        xx, yy, xy = it[0, 0] / area, it[1, 1] / area, it[0, 1] / area
                        l1 = (xx + yy) / 2 + np.sqrt(((xx - yy) / 2) ** 2 + xy ** 2)
                        l2 = (xx + yy) / 2 - np.sqrt(((xx - yy) / 2) ** 2 + xy ** 2)
                        if 'I_Area_Principal_Max' in selected_features:
                            props['I_Area_Principal_Max'] = l1
                        if 'I_Area_Principal_Min' in selected_features:
                            props['I_Area_Principal_Min'] = l2
                        if 'I_Area_Anisotropy' in selected_features:
                            props['I_Area_Anisotropy'] = l1 / max(l2, 1e-12)

                    if USE_LOG_SPACE:
                        for key in list(props.keys()):
                            props[key] = np.arcsinh(props[key])

                if all(np.isfinite(list(props.values()))):
                    properties.append(props)

        return properties

    except Exception as e:
        logging.error(f"Error processing layer {index} along {axis}: {e}")
        return []


def process_data(df):
    df = df.copy()

    if 'Compactness Ratio' in df.columns:
        df = df[df['Compactness Ratio'] != 1]

    if 'Shape Factor' in df.columns:
        df = df[df['Shape Factor'] <= 1]

    if 'Orientation' in df.columns:
        df['Orientation'] = np.mod(df['Orientation'], np.pi)

    cols = [c for c in df.columns if c != 'Orientation']
    mask = pd.Series(True, index=df.index)

    for col in cols:
        med = df[col].median()
        mad = (df[col] - med).abs().median()
        if mad > 0:
            mask &= (df[col] - med).abs() <= 6 * mad

    return df[mask]


def process_layers(
        layers: np.ndarray,
        mode: str = 'all',
        sample_skip: int = 0,
        sample_offset: int = 0,
        single_index: int = 0,
        selected_axes: List[str] = None,
        max_workers: int | None = None
) -> List[Dict]:
    if selected_axes is None:
        selected_axes = ['z']

    cube_size = layers.shape[0]
    all_grains: List[Dict] = []

    logging.info(
        f"process_layers: mode={mode}, axes={selected_axes}, cube_size={cube_size}"
    )

    if mode == 'all':
        indices = list(range(cube_size))

    elif mode == 'sample':
        indices = list(range(sample_offset, cube_size, sample_skip + 1))

    elif mode == 'single':
        indices = [single_index]

    else:
        raise ValueError(f"Unknown analysis mode: {mode}")

    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for axis in selected_axes:
            for idx in indices:
                tasks.append(
                    executor.submit(process_layer, idx, layers, axis)
                )

        for future in as_completed(tasks):
            try:
                grains = future.result()

                if not isinstance(grains, list):
                    logging.warning("process_layer returned non-list")
                    continue

                for grain in grains:
                    if isinstance(grain, dict):
                        all_grains.append(grain)

            except Exception as exc:
                logging.exception(f"Layer processing failed: {exc}")

    logging.info(f"Total grains collected: {len(all_grains)}")

    return all_grains


def compute_circular_stats(values):
    values = np.mod(values, np.pi)

    q1, median, q3 = np.mod(np.quantile(np.unwrap(values), [0.25, 0.5, 0.75]), np.pi)
    iqr = (q3 - q1) % np.pi

    return {
        "Mean": circmean(values, high=np.pi, low=0),
        "Std": circstd(values, high=np.pi, low=0),
        "Median": median,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr
    }


def ensure_hist_folder(output_dir):
    hist_root = os.path.join(output_dir, "histograms")
    os.makedirs(hist_root, exist_ok=True)
    return hist_root


def save_iteration_histograms(stats, df, target_values, iteration, output_dir):
    hist_root = ensure_hist_folder(output_dir)
    iter_folder = os.path.join(hist_root, f"{iteration:04d}")
    os.makedirs(iter_folder, exist_ok=True)

    for col in selected_features:
        try:
            if col not in df.columns:
                logging.warning(f"{col} немає в df -> пропускаємо гістограму.")
                continue

            if col == "Orientation":
                current_vals = np.mod(df[col].values, np.pi)
                target_stats = target_values.get("Orientation", None)
                plot_orientation_polar_iteration(current_vals, iter_folder, iteration, current_stats=None,
                                                 target_stats=target_stats)
            else:
                current_vals = df[col].values
                target_stats = target_values.get(col, None)
                current_stats = {
                    'Mean': float(stats['Mean'].get(col, np.nan)),
                    'Q1': float(stats['Q1'].get(col, np.nan)),
                    'Q3': float(stats['Q3'].get(col, np.nan)),
                    'IQR': float(stats['IQR'].get(col, np.nan)) if 'IQR' in stats else np.nan
                }
                plot_regular_histogram_iteration(None, current_vals, col, iter_folder, iteration, current_stats,
                                                 target_stats)
        except Exception as e:
            logging.error(f"Помилка при збереженні гістограми {col} для ітерації {iteration}: {e}")


def safe_filename(name: str) -> str:
    name = name.strip()
    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace(":", "_")
    name = name.replace("-", "_")
    name = re.sub(r"__+", "_", name)
    return name


def save_iteration_cube(cube_file_path: str, iteration: int, output_dir: str):
    cubes_root = os.path.join(output_dir, "cubes")

    iter_folder = os.path.join(cubes_root, f"{iteration:04d}")
    os.makedirs(iter_folder, exist_ok=True)

    cube_name = f"{iteration:04d}_cube.csv"
    target_path = os.path.join(iter_folder, cube_name)

    try:
        import shutil
        shutil.copy2(cube_file_path, target_path)
        logging.info(f"Saved cube CSV to separate folder: {target_path}")
    except Exception as e:
        logging.error(f"Не вдалося зберегти cube CSV для ітерації {iteration}: {e}")


def plot_regular_histogram_iteration(df_before, df_after_vals, col, output_dir, iteration,
                                     current_stats=None, target_stats=None):
    data_all = np.asarray(df_after_vals)
    if data_all.size == 0:
        logging.warning(f"No data for {col} in iteration {iteration}")
        return

    bins_count = int(np.sqrt(len(data_all))) if len(data_all) > 1 else 1
    bins = np.linspace(data_all.min(), data_all.max(), max(10, bins_count))

    plt.figure(figsize=(10, 5))

    plt.hist(data_all, bins=bins, color='teal', alpha=0.7, label='Current')

    if current_stats is not None:
        try:
            if np.isfinite(current_stats.get('Mean', np.nan)):
                plt.axvline(current_stats['Mean'], color='yellow', linestyle='-', linewidth=2, label='Current Mean')
            if np.isfinite(current_stats.get('Q1', np.nan)):
                plt.axvline(current_stats['Q1'], color='orange', linestyle='--', linewidth=1.5, label='Current Q1')
            if np.isfinite(current_stats.get('Q3', np.nan)):
                plt.axvline(current_stats['Q3'], color='purple', linestyle='--', linewidth=1.5, label='Current Q3')
        except Exception:
            pass

    if target_stats is not None:
        try:
            if 'Mean' in target_stats and np.isfinite(target_stats['Mean']):
                plt.axvline(target_stats['Mean'], color='red', linestyle='-', linewidth=2, label='Target Mean')
            if 'Q1' in target_stats and np.isfinite(target_stats['Q1']):
                plt.axvline(target_stats['Q1'], color='red', linestyle=':', linewidth=1.5, label='Target Q1')
            if 'Q3' in target_stats and np.isfinite(target_stats['Q3']):
                plt.axvline(target_stats['Q3'], color='red', linestyle='-.', linewidth=1.5, label='Target Q3')
        except Exception:
            pass

    plt.title(f'Histogram: {col} (iter {iteration:04d})')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(loc='best')
    plt.tight_layout()
    safe_col = safe_filename(col)
    filename = f"{iteration:04d}_{safe_col}_hist.png"

    full_path = os.path.join(output_dir, filename)

    plt.savefig(full_path)
    plt.close()
    logging.info(f"Saved histogram: {full_path}")


def prepare_polar_params_single(data, bins):
    hist, edges = np.histogram(data, bins=bins)
    epsilon = 1e-2
    hist = np.maximum(hist.astype(float), epsilon)

    theta = (edges[:-1] + edges[1:]) / 2
    width = np.diff(edges)
    r_max = hist.max() * 1.1

    return theta, width, hist, r_max, epsilon


def plot_orientation_polar_iteration(data_after, output_dir, iteration,
                                     current_stats=None, target_stats=None):
    data = np.mod(data_after, np.pi)

    n_bins = int(np.sqrt(len(data)))
    n_bins = max(12, min(n_bins, 45))

    bins = np.linspace(0, np.pi, n_bins + 1)

    theta, width, hist, r_max, eps = prepare_polar_params_single(data, bins)

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, projection='polar')
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    ax.bar(theta, hist, width=width, alpha=0.6, color='teal', label='Current', edgecolor='white', linewidth=0.5)

    if current_stats is None:
        try:
            cs = compute_circular_stats(data)
            curr_mean = cs['Mean']
            curr_q1 = cs['Q1']
            curr_q3 = cs['Q3']
        except Exception:
            curr_mean, curr_q1, curr_q3 = None, None, None
    else:
        curr_mean = current_stats.get('Mean', None)
        curr_q1 = current_stats.get('Q1', None)
        curr_q3 = current_stats.get('Q3', None)

    # Current
    if curr_mean is not None and np.isfinite(curr_mean):
        ax.plot([curr_mean, curr_mean], [eps, r_max], color='yellow', linewidth=2.5, label='Current Mean', zorder=10)
    if curr_q1 is not None and np.isfinite(curr_q1):
        ax.plot([curr_q1, curr_q1], [eps, r_max], color='orange', linestyle='--', linewidth=2, label='Current Q1',
                zorder=10)
    if curr_q3 is not None and np.isfinite(curr_q3):
        ax.plot([curr_q3, curr_q3], [eps, r_max], color='purple', linestyle='--', linewidth=2, label='Current Q3',
                zorder=10)

    # Target
    if target_stats is not None:
        try:
            t_mean = target_stats.get('Mean', None)
            t_q1 = target_stats.get('Q1', None)
            t_q3 = target_stats.get('Q3', None)
            if t_mean is not None and np.isfinite(t_mean):
                ax.plot([t_mean, t_mean], [eps, r_max], color='red', linewidth=2.5, label='Target Mean', zorder=10)
            if t_q1 is not None and np.isfinite(t_q1):
                ax.plot([t_q1, t_q1], [eps, r_max], color='red', linestyle=':', linewidth=2, label='Target Q1',
                        zorder=10)
            if t_q3 is not None and np.isfinite(t_q3):
                ax.plot([t_q3, t_q3], [eps, r_max], color='red', linestyle='-.', linewidth=2, label='Target Q3',
                        zorder=10)
        except Exception:
            pass

    ax.set_xticks(np.linspace(0, np.pi, 5))
    ax.set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax.set_ylim(eps, r_max)
    ax.set_title(f'Polar Histogram: Orientation (iter {iteration:04d})', y=1.1)

    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    fname = os.path.join(output_dir, f"{iteration:04d}_Orientation_polar.png")
    plt.savefig(fname)
    plt.close()
    logging.info(f"Saved polar histogram: {fname}")


# Функція для обробки файлу та обчислення середніх значень властивостей
def process_and_calculate(file_path, output_props_file,
                          analysis_mode=ANALYSIS_MODE,
                          sample_skip=SAMPLE_SKIP,
                          sample_offset=SAMPLE_OFFSET,
                          single_index=SINGLE_LAYER_INDEX):
    try:
        data = np.genfromtxt(file_path, delimiter=';', skip_header=1, dtype=int)
    except Exception as e:
        logging.error(f"Помилка зчитування файлу: {e}")
        return None, None

    cube_size = np.max(data[:, :3]) + 1
    layers = np.zeros((cube_size, cube_size, cube_size), dtype=int)
    layers[data[:, 0], data[:, 1], data[:, 2]] = data[:, 3]

    grains = process_layers(
        layers,
        mode=analysis_mode,
        sample_skip=sample_skip,
        sample_offset=sample_offset,
        single_index=single_index,
        selected_axes=SELECTED_AXES
    )

    df = pd.DataFrame(grains)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = process_data(df)

    # Лінійна статистика
    stats = {
        'Mean': df.mean(),
        'Std': df.std(),
        'Median': df.median(),
        'Q1': df.quantile(0.25),
        'Q3': df.quantile(0.75),
        'IQR': df.quantile(0.75) - df.quantile(0.25)
    }

    # Circular статистики для Orientation
    if "Orientation" in df.columns:
        circ = compute_circular_stats(df["Orientation"].values)
        stats['Mean']["Orientation"] = circ["Mean"]
        stats['Std']["Orientation"] = circ["Std"]
        stats['Median']["Orientation"] = circ["Median"]
        stats['Q1']["Orientation"] = circ["Q1"]
        stats['Q3']["Orientation"] = circ["Q3"]
        stats['IQR']["Orientation"] = circ["IQR"]

    print(f"Статистика властивостей:\n{stats}")
    df.to_csv(output_props_file, index=False)
    logging.info(f"Файл з властивостями збережено як: {output_props_file}")
    return stats, df


def load_target_data(file_path, metric_type, features_list):
    if not os.path.exists(file_path):
        logging.error(f"Файл цільових значень не знайдено: {file_path}")
        sys.exit(1)

    if metric_type == 'Energy Distance':
        df = pd.read_csv(file_path)
        available_cols = [c for c in features_list if c in df.columns]

        if not available_cols:
            logging.error(f"У файлі {file_path} не знайдено потрібних колонок з {features_list}")
            sys.exit(1)

        logging.info(f"Завантажено розподіл для Energy Distance. Columns: {available_cols}, Rows: {len(df)}")
        return df[available_cols]

    else:
        df = pd.read_csv(file_path)
        df['Property'] = df['Property'].ffill()
        selected_set = set(features_list)
        name_mapping = {"Scale Factor": "scale_factor"}

        targets = {}
        for _, row in df.iterrows():
            prop_raw = str(row['Property']).strip()
            stat_raw = str(row['Statistic']).strip()
            value = float(row['Value'])

            prop = name_mapping.get(prop_raw, prop_raw)
            if prop not in selected_set: continue

            stat = stat_raw.replace("Circular ", "")
            if prop not in targets: targets[prop] = {}
            targets[prop][stat] = value

        return targets


GLOBAL_TARGET = load_target_data(target_csv_path, selected_metric_type, selected_features)


def calculate_smape(actual, predicted):
    smape_values = [
        100 * abs(a - p) / ((abs(a) + abs(p)) / 2)
        for a, p in zip(actual, predicted) if (a != 0 or p != 0)
    ]
    return np.mean(smape_values)


def calculate_mse(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return float(np.mean((actual - predicted) ** 2))


def calculate_mspe(actual, predicted):
    vals = []
    for a, p in zip(actual, predicted):
        if a == 0:
            continue
        vals.append(((a - p) / a) ** 2)
    if len(vals) == 0:
        return 0.0
    return 100.0 * float(np.mean(vals))


def calculate_energy_distance(model_data, exp_data):
    if len(model_data) == 0 or len(exp_data) == 0:
        return np.inf

    if model_data.ndim == 1:
        model_data = model_data.reshape(-1, 1)
    if exp_data.ndim == 1:
        exp_data = exp_data.reshape(-1, 1)

    MAX_SAMPLES = 2000
    if len(model_data) > MAX_SAMPLES:
        indices = np.random.choice(len(model_data), MAX_SAMPLES, replace=False)
        model_data = model_data[indices]
    if len(exp_data) > MAX_SAMPLES:
        indices = np.random.choice(len(exp_data), MAX_SAMPLES, replace=False)
        exp_data = exp_data[indices]

    avg_dist_xy = np.mean(cdist(model_data, exp_data, metric='euclidean'))
    avg_dist_xx = np.mean(pdist(model_data, metric='euclidean'))
    avg_dist_yy = np.mean(pdist(exp_data, metric='euclidean'))

    # 2*E|X-Y| - E|X-X'| - E|Y-Y'|
    e_dist_sq = 2 * avg_dist_xy - avg_dist_xx - avg_dist_yy

    return np.sqrt(max(e_dist_sq, 0))


def minimize_properties(params):
    concentration, halfaxis_b, halfaxis_c, ellipse_order, wave_coefficient, wave_spread, initial_nuclei_count = params
    halfaxis_a = halfaxis_b

    initial_nuclei_count = int(round(initial_nuclei_count))

    # TODO: шлях до файлів
    output_file = r".\cube_output.csv"
    output_props_file = r".\properties_output.csv"

    generated_file = start(FIXED_SIZE, concentration, halfaxis_a, halfaxis_b, halfaxis_c,
                           0, 0, 0,
                           wave_coefficient, wave_spread, initial_nuclei_count, ellipse_order, output_file)
    if not generated_file:
        logging.error("Не вдалося створити файл.")
        return np.inf

    stats, df = process_and_calculate(generated_file, output_props_file)

    if stats is None or df is None:
        logging.error("Не вдалося обробити статистику.")
        return np.inf

    target_data = GLOBAL_TARGET

    # Визначаємо дані для графіків (графіки вміють працювати тільки зі словником)
    target_values_for_plot = target_data if isinstance(target_data, dict) else None

    try:
        iteration = next(ITER_COUNTER)
        save_iteration_histograms(stats, df, target_data, iteration, output_folder)
        save_iteration_cube(generated_file, iteration, output_folder)
    except Exception as e:
        logging.error(f"Не вдалося зберегти гістограми для ітерації: {e}")

    individual_errors = {}
    total_error = 0
    final_score = np.inf

    if selected_metric_type == 'Energy Distance':
        if not isinstance(target_data, pd.DataFrame):
            logging.error("Обрано Energy Distance, але цільові дані не є DataFrame")
            return np.inf

        valid_features = [f for f in selected_features if f in df.columns and f in target_data.columns]

        if not valid_features:
            logging.error("Немає спільних колонок між симуляцією і ціллю для Energy Distance.")
            return np.inf

        for feat in valid_features:
            sim_feat = df[feat].values
            exp_feat = target_data[feat].values

            exp_feat_mean = np.mean(exp_feat)
            exp_feat_std = np.std(exp_feat)
            if exp_feat_std == 0:
                exp_feat_std = 1.0

            sim_feat_norm = (sim_feat - exp_feat_mean) / exp_feat_std
            exp_feat_norm = (exp_feat - exp_feat_mean) / exp_feat_std

            ed_feat_val = calculate_energy_distance(sim_feat_norm, exp_feat_norm)

            individual_errors[feat] = {"EnergyDist": ed_feat_val}

        sim_values = df[valid_features].values
        exp_values = target_data[valid_features].values

        exp_mean = np.mean(exp_values, axis=0)
        exp_std = np.std(exp_values, axis=0)
        exp_std[exp_std == 0] = 1.0

        sim_values_norm = (sim_values - exp_mean) / exp_std
        exp_values_norm = (exp_values - exp_mean) / exp_std

        ed_value = calculate_energy_distance(sim_values_norm, exp_values_norm)

        total_error = ed_value
        final_score = total_error

        individual_errors["Multivariate_Energy_Distance"] = {"All": ed_value}
        print(f"Total Multivariate Energy Distance (Normalized): {total_error:.4f}")

        # --- MSE / SMAPE  ---
    else:
        count = 0
        # target_data тут це словник (dictionary)
        for feature_name in selected_features:
            if feature_name not in stats['Mean'] or feature_name not in target_data:
                continue

            if feature_name == 'Orientation':
                current_metrics = ['Mean', 'Std']
            elif feature_name == 'Inertia Tensor/Area XY':
                current_metrics = [m for m in selected_metrics if m != 'Median']
            else:
                current_metrics = selected_metrics

            individual_errors[feature_name] = {}

            for metric in current_metrics:
                try:
                    actual = stats[metric][feature_name]
                    predicted = target_data[feature_name][metric]

                    # Логіка помилки
                    if USE_LOG_SPACE:
                        error = (actual - predicted) ** 2
                    else:
                        if feature_name == 'Orientation':
                            error = abs(actual - predicted) * 30.0
                        elif selected_metric_type == 'SMAPE':
                            error = calculate_smape([actual], [predicted])
                        elif selected_metric_type == 'MSE':
                            target_std = target_data[feature_name].get('Std', 1.0)

                            if target_std == 0 or np.isnan(target_std):
                                target_std = 1.0

                            norm_actual = actual / target_std
                            norm_predicted = predicted / target_std
                            error = calculate_mse([norm_actual], [norm_predicted])
                        elif selected_metric_type == 'MSPE':
                            error = calculate_mspe([actual], [predicted])
                        else:
                            error = abs(actual - predicted)

                    individual_errors[feature_name][metric] = error
                    total_error += error
                    count += 1
                except KeyError:
                    continue

        if count > 0:
            total_error = total_error / count
            final_score = total_error
        else:
            final_score = np.inf

    print(f"Загальна помилка: {final_score:.4f}")
    print(f"Помилки по параметрах:")
    for feature, metrics in individual_errors.items():
        if feature == "Multivariate_Energy_Distance" and selected_metric_type != 'Energy Distance':
            continue

        for metric, value in metrics.items():
            try:
                val_float = float(value)
                unit = " (abs*100)" if feature == 'Orientation' else ""

                if selected_metric_type == 'Energy Distance':
                    print(f"{feature} - {metric}: {val_float:.4f}")
                else:
                    print(f"{feature} - {metric}: {val_float:.4f}{unit}%")
            except:
                print(f"{feature} - {metric}: {value}")

    log_iteration_to_csv(
        iteration=iteration,
        params=params,
        stats=stats,
        individual_errors=individual_errors,
        total_error=total_error
    )

    summary_file = os.path.join(output_folder, "summary.txt")
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write("\n===========================\n")
        f.write(f"№ Ітерації: {iteration}\n")
        f.write(f"Загальна похибка: {total_error}\n")
        f.write(f"Параметри: {params}\n")
        # f.write(f"Загальна помилка: {final_score:.2f}\n") # Можна закоментувати, якщо Energy Distance це і є загальна помилка

        for feature, metrics in individual_errors.items():
            # --- СПЕЦІАЛЬНИЙ ВИВІД ДЛЯ ENERGY DISTANCE ---
            if feature == "Multivariate_Energy_Distance":
                # Беремо значення помилки
                val = metrics.get('All', 0.0)
                # Використовуємо змінну valid_features, яка була визначена вище (рядок ~733)
                f.write(f"Energy Distance calculated on features {valid_features}: {val:.4f}\n")
                continue  # Пропускаємо стандартний вивід для цього ключа
            # ---------------------------------------------

            # Стандартний вивід для інших метрик (MSE, SMAPE і т.д.)
            for metric, value in metrics.items():
                # unit = " (abs*100)" if feature == 'Orientation' else "%"
                f.write(f"{feature} - {metric}: {value:.2f}\n")

    return final_score


def find_best_starting_point(bounds):
    x0_list = [
        [(low + high) / 2 for (low, high) in bounds],
        [low + (high - low) * 0.25 for (low, high) in bounds],
        [low + (high - low) * 0.75 for (low, high) in bounds],
        [np.random.uniform(low, high) for (low, high) in bounds]
    ]

    best_x0 = None
    best_fun = float('inf')

    for i, x0 in enumerate(x0_list):
        logging.info(f"Оцінка стартової точки #{i + 1}: {x0}")

        fun = minimize_properties(x0)
        logging.info(f"→ SMAPE = {fun:.6f}")

        if fun < best_fun:
            best_fun = fun
            best_x0 = x0

    return best_x0


def optimize_properties():
    scipy_bounds = Bounds([b[0] for b in bounds], [b[1] for b in bounds])

    print(f"Вибраний метод: {selected_method}")
    start_time = time.time()

    if selected_method in ['SLSQP', 'L-BFGS-B']:
        print("Вибір найкращої стартової точки...")
        # x0 = find_best_starting_point(bounds)
        x0 = [0.075, 0.9, 1, 2.7, 15, 2.0, 7]

        result = minimize(
            minimize_properties,
            x0=x0,
            method=selected_method,
            bounds=bounds,
            options={'disp': True, 'maxiter': 100}
        )

    elif selected_method == 'basinhopping':
        print("Вибір найкращої стартової точки...")
        # x0 = find_best_starting_point(bounds)
        x0 = [0.075, 0.9, 1, 2.7, 15, 2.0, 7]

        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "options": {"disp": True, "maxiter": 20}
        }

        result = basinhopping(
            func=minimize_properties,
            x0=x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=5,
            niter_success=2,
            disp=True,
            accept_test=lambda f_new, x_new, f_old, x_old:
            np.all(scipy_bounds.lb <= x_new) and np.all(x_new <= scipy_bounds.ub)
        )

    elif selected_method == 'Dual Annealing':
        x0 = [0.075, 0.9, 1, 2.7, 15, 2.0, 7]
        result = dual_annealing(
            func=minimize_properties,
            bounds=bounds,
            x0=x0,
            maxiter=1,
            no_local_search=True
        )

    elif selected_method == 'Differential Evolution':
        result = differential_evolution(
            func=minimize_properties,
            bounds=bounds,
            strategy='best1bin',
            maxiter=20,  # Обмеження за кількістю поколінь
            popsize=3,  # 3 * 6 параметрів = 18 прорахунків за ітерацію
            tol=0.05,  # Досить точний, але не надмірний поріг зупинки
            mutation=(0.5, 1),  # Стандарт для збереження різноманітності
            recombination=0.7,  # Швидкість змішування параметрів
            disp=True,  # Щоб ви бачили прогрес у консолі
            polish=False  # Додає довгий локальний пошук в кінці
        )

    elif selected_method == 'Manual Sweep':
        # TODO: sweep_param
        sweep_param = "wave_spread"
        sweep_index = param_names.index(sweep_param)

        sweep_results = manual_parameter_sweep(
            param_index=sweep_index,
            base_params=base_params,
            bounds=bounds,
            n_points=30
        )
        logging.info(f"Manual sweep finished")

    else:
        raise ValueError(f"Невідомий метод: {selected_method}")

    elapsed_time = time.time() - start_time

    print(f"Optimization completed in {elapsed_time:.2f} seconds.")
    print(f"Best parameters found: {result.x}")
    print(f"Final SMAPE error: {result.fun}")

    summary_file = os.path.join(output_folder, "summary.txt")
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"Optimization method: {selected_method}\n")
        f.write(f"Initial point: {result.x.tolist() if hasattr(result, 'x') else 'N/A'}\n")
        f.write(f"Final SMAPE error: {result.fun:.6f}\n")
        f.write(f"Total optimization time: {elapsed_time:.2f} seconds\n")

    return result


def main():
    log_file_path = os.path.join(output_folder, "full_output.txt")

    log_file_stream = open(log_file_path, "w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(sys.__stdout__, log_file_stream)
    sys.stderr = Tee(sys.__stderr__, log_file_stream)

    try:
        result = optimize_properties()
        print(result)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_stream.close()


if __name__ == "__main__":
    main()

