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
import skimage
from scipy.optimize import (
    minimize,
    Bounds,
    basinhopping,
    differential_evolution,
    dual_annealing
)
from scipy.stats import circmean, circstd
from skimage.measure import regionprops

from MatViz3DLauncher import MatViz3DLauncher

# TODO:  Метод оптимізації, ('SLSQP', 'L-BFGS-B', 'Dual Annealing', 'basinhopping', 'Differential Evolution', 'Manual Sweep')
selected_method = 'Dual Annealing'
selected_metric_type = 'SMAPE'  # або 'MSE' 'SMAPE' 'MSPE'

# TODO:   Глобальні змінні
# selected_features = ['ECR', 'scale_factor', 'Compactness Ratio', 'Orientation']
# selected_features = [
#    'ECR', 'scale_factor', 'Compactness Ratio',
#    'I_Anisotropy'
# ]

selected_features = [
    'Inertia Tensor/Area XX', 'Inertia Tensor/Area XY', 'Inertia Tensor/Area YY'
]

# selected_features = [
#     'ECR', 'scale_factor', 'Compactness Ratio',
#     'Inertia Tensor XX', 'Inertia Tensor XY', 'Inertia Tensor YY',
#     'Inertia Tensor/Area XX', 'Inertia Tensor/Area XY', 'Inertia Tensor/Area YY',
#     'I_Principal_Max', 'I_Principal_Min', 'I_Anisotropy',
#     'I_Area_Principal_Max', 'I_Area_Principal_Min', 'I_Area_Anisotropy'
# ]
# selected_features = ['ECR', 'scale_factor', 'Compactness Ratio']

# selected_metrics = ['Mean', 'Std', 'Q1', 'Q3']
# selected_metrics = ['Mean', 'Q1', 'Q3']
# selected_metrics = ['Mean', 'Std']
selected_metrics = ['Mean']
# TODO: Фіксований розмір куба
FIXED_SIZE = 130

# TODO: Ініціалізація запуска MatViz3D
exe_path = r"D:\Project(MatViz3D)\Random\current_build\debug\MatViz3D.exe"
launcher = MatViz3DLauncher(exe_path)

# TODO:  Створення каталогу для виведення
output_folder = r"D:\Project(MatViz3D)\Random\Paper_Optimisation\test\iter\test_6"
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
            process_and_calculate

    def flush(self):
        for s in self.streams:
            s.flush()


# Створюємо шлях до файлу для логів
log_file_path = os.path.join(output_folder, "full_output.txt")  # Вкажи свій шлях
log_file_stream = open(log_file_path, "w", encoding="utf-8")

logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Перенаправляємо stdout і stderr на файл і консоль
sys.stdout = Tee(sys.__stdout__, log_file_stream)
sys.stderr = Tee(sys.__stderr__, log_file_stream)

# Налаштування логування (запис в той самий файл)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Логування в консоль (stdout)
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8')  # Логування в файл
    ]
)

# CSV файл для параметрів запуску
csv_params_path = os.path.join(output_folder, "run_parameters_and_selected_features.csv")
csv_headers = [
    "Iteration",
    "wave_coefficient", "concentration",
    "halfaxis_a", "halfaxis_b", "halfaxis_c",
    "ellipse_order",
]

# значення selected features
for f in selected_features:
    for m in selected_metrics:
        csv_headers.append(f"{f}_{m}")

# загальна похибка
csv_headers.append("Total_Error")

# окремі похибки
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

    concentration, halfaxis_b, halfaxis_c, ellipse_order = params
    wave_coefficient = 1.0
    halfaxis_a = (halfaxis_b + halfaxis_c) / 2

    row = {
        "Iteration": iteration,
        "wave_coefficient": wave_coefficient,
        "concentration": concentration,
        "halfaxis_a": halfaxis_a,
        "halfaxis_b": halfaxis_b,
        "halfaxis_c": halfaxis_c,
        "ellipse_order": ellipse_order,
        "Total_Error": total_error
    }

    # значення фіч
    for f in selected_features:
        for m in selected_metrics:
            row[f"{f}_{m}"] = stats[m].get(f, np.nan)

    # окремі похибки
    for f in selected_features:
        for m in selected_metrics:
            row[f"Error_{f}_{m}"] = (
                individual_errors.get(f, {}).get(m, np.nan)
            )

    with open(csv_params_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


param_names = [
    "concentration",
    "halfaxis_b",
    "halfaxis_c",
    "ellipse_order"
]

# TODO: Обмеження
bounds = [(0.01, 0.2), (0.65, 1.5), (0.65, 1.5), (0.5, 6)]

base_params = np.array([
    0.1,  # concentration
    1.5,  # halfaxis_b
    1.5,  # halfaxis_c
    1.7  # ellipse_order
], dtype=float)


def manual_parameter_sweep(
        param_index: int,
        base_params: np.ndarray,
        bounds: list,
        n_points: int = 15
):
    """
    param_index — індекс параметра, який варіюємо (0..4)
    base_params — фіксовані значення
    bounds — bounds як у оптимізації
    n_points — кількість точок
    """

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
          orientation_angle_c, wave_coefficient, ellipse_order, output_file):
    print(
        f"Запуск MatViz3D з параметрами: [{size}, conc = {concentration}, h_a = {halfaxis_a}, h_b = {halfaxis_b}, h_c = {halfaxis_c}, "
        f"or_a = {orientation_angle_a}, or_b = {orientation_angle_b}, or_c = {orientation_angle_c}, wc = {wave_coefficient}, ellipse_order = {ellipse_order}]")

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
        ellipse_order=ellipse_order,
        output_file=output_file
    )


def process_layer(index, layers, axis):
    """Обробка одного шару і обчислення всіх властивостей, включно з похідними."""
    try:
        # Вибір шару по осі
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
                # виключаємо зерна на границі
                if np.any(region.coords[:, 0] == 0) or np.any(region.coords[:, 0] == layer.shape[0] - 1) \
                        or np.any(region.coords[:, 1] == 0) or np.any(region.coords[:, 1] == layer.shape[1] - 1):
                    continue

                area = region.area
                norm_area = area / layer_area
                props = {}

                # базові властивості
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

                # інерційні властивості
                if any(f in selected_features for f in inertia_needed.union(principal_needed)):
                    it = region.inertia_tensor

                    # базові компоненти (за потребою)
                    if 'Inertia Tensor XX' in selected_features:
                        props['Inertia Tensor XX'] = it[0, 0]
                    if 'Inertia Tensor XY' in selected_features:
                        props['Inertia Tensor XY'] = it[0, 1]
                    if 'Inertia Tensor YY' in selected_features:
                        props['Inertia Tensor YY'] = it[1, 1]

                    # нормовані на площу компоненти (за потребою)
                    if 'Inertia Tensor/Area XX' in selected_features:
                        props['Inertia Tensor/Area XX'] = it[0, 0] / area
                    if 'Inertia Tensor/Area XY' in selected_features:
                        props['Inertia Tensor/Area XY'] = it[0, 1] / area
                    if 'Inertia Tensor/Area YY' in selected_features:
                        props['Inertia Tensor/Area YY'] = it[1, 1] / area

                    # похідні властивості (звичайний тензор)
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

                    # похідні властивості для тензора, нормованого на площу
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

                if all(np.isfinite(list(props.values()))):
                    properties.append(props)

        return properties

    except Exception as e:
        logging.error(f"Error processing layer {index} along {axis}: {e}")
        return []


import pandas as pd
import numpy as np


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
    """
    Process layers according to selected mode and axes.
    Returns list of grains (one dict = one grain).
    """

    if selected_axes is None:
        selected_axes = ['z']

    cube_size = layers.shape[0]
    all_grains: List[Dict] = []

    logging.info(
        f"process_layers: mode={mode}, axes={selected_axes}, cube_size={cube_size}"
    )

    # ---- determine layer indices ----
    if mode == 'all':
        indices = list(range(cube_size))

    elif mode == 'sample':
        indices = list(range(sample_offset, cube_size, sample_skip + 1))

    elif mode == 'single':
        indices = [single_index]

    else:
        raise ValueError(f"Unknown analysis mode: {mode}")

    # ---- multiprocessing over (axis, layer_index) ----
    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for axis in selected_axes:
            for idx in indices:
                tasks.append(
                    executor.submit(process_layer, idx, layers, axis)
                )

        for future in as_completed(tasks):
            try:
                grains = future.result()  # List[dict]

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
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr
    }


def ensure_hist_folder(output_dir):
    hist_root = os.path.join(output_dir, "histograms")
    os.makedirs(hist_root, exist_ok=True)
    return hist_root


def save_iteration_histograms(stats, df, target_values, iteration, output_dir):
    """
    Зберігає гістограми для всіх властивостей (і поляр для Orientation) для даної ітерації.
    stats: словник статистик як у process_and_calculate
    df: DataFrame поточних значень
    target_values: ваш словник target_values
    iteration: int — номер ітерації
    output_dir: базова папка виводу
    """
    hist_root = ensure_hist_folder(output_dir)
    iter_folder = os.path.join(hist_root, f"{iteration:04d}")
    os.makedirs(iter_folder, exist_ok=True)

    for col in selected_features:
        try:
            if col not in df.columns:
                logging.warning(f"{col} немає в df -> пропускаємо гістограму.")
                continue

            if col == "Orientation":
                # polar: передаємо поточні значення (df[col]) й цільові stats для Orientation
                current_vals = np.mod(df[col].values, np.pi)
                target_stats = target_values.get("Orientation", None)
                plot_orientation_polar_iteration(current_vals, iter_folder, iteration, current_stats=None,
                                                 target_stats=target_stats)
            else:
                # звичайна гістограма: малюємо current distribution й додаємо поточні/цільові лінії
                current_vals = df[col].values
                target_stats = target_values.get(col, None)
                # Поточні статистики беремо з stats словників, якщо є
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


def plot_regular_histogram_iteration(df_before, df_after_vals, col, output_dir, iteration,
                                     current_stats=None, target_stats=None):
    """
    Оновлена функція: стиль відповідає Stat_img_charc_folders.py (Teal color, binning logic).
    """
    data_all = np.asarray(df_after_vals)
    if data_all.size == 0:
        logging.warning(f"No data for {col} in iteration {iteration}")
        return

    # Логіка бінів як у Stat_img_charc_folders.py (np.sqrt)
    # Якщо даних дуже мало, np.linspace може видати помилку, тому додамо страховку max(1, ...)
    bins_count = int(np.sqrt(len(data_all))) if len(data_all) > 1 else 1
    bins = np.linspace(data_all.min(), data_all.max(), max(10, bins_count))

    plt.figure(figsize=(10, 5))

    # Стиль як у "Before" з reference файлу (teal, alpha=0.7) або "After" (coral).
    # Використаємо 'teal' як основний колір для поточної ітерації.
    plt.hist(data_all, bins=bins, color='teal', alpha=0.7, label='Current')

    # Поточні статистики — вертикальні лінії
    if current_stats is not None:
        try:
            # Використовуємо яскраві кольори для ліній поверх teal гістограми
            if np.isfinite(current_stats.get('Mean', np.nan)):
                plt.axvline(current_stats['Mean'], color='yellow', linestyle='-', linewidth=2, label='Current Mean')
            if np.isfinite(current_stats.get('Q1', np.nan)):
                plt.axvline(current_stats['Q1'], color='orange', linestyle='--', linewidth=1.5, label='Current Q1')
            if np.isfinite(current_stats.get('Q3', np.nan)):
                plt.axvline(current_stats['Q3'], color='purple', linestyle='--', linewidth=1.5, label='Current Q3')
        except Exception:
            pass

    # Target статистики — червоні лінії
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
    """
    Адаптована допоміжна функція з Stat_img_charc_folders.py для одного набору даних.
    Розраховує параметри для полярної діаграми.
    """
    hist, edges = np.histogram(data, bins=bins)
    epsilon = 1e-2
    hist = np.maximum(hist.astype(float), epsilon)

    theta = (edges[:-1] + edges[1:]) / 2
    width = np.diff(edges)
    r_max = hist.max() * 1.1

    return theta, width, hist, r_max, epsilon


def plot_orientation_polar_iteration(data_after, output_dir, iteration,
                                     current_stats=None, target_stats=None):
    """
    Оновлена функція з виправленим розрахунком бінів (bins) для великих масивів даних.
    """
    # 1. Підготовка даних (радіани [0, pi])
    data = np.mod(data_after, np.pi)

    # 2. ВИПРАВЛЕННЯ: Розумний розрахунок кількості стовпчиків
    # Використовуємо корінь квадратний, але обмежуємо діапазон (від 12 до 45 стовпчиків),
    # щоб графік завжди був читабельним, незалежно від того, 100 у вас зерен чи 10000.
    n_bins = int(np.sqrt(len(data)))
    n_bins = max(12, min(n_bins, 45))

    bins = np.linspace(0, np.pi, n_bins + 1)

    # 3. Розрахунок параметрів для полярної діаграми
    # (Використовуємо ту ж логіку, що й раніше)
    theta, width, hist, r_max, eps = prepare_polar_params_single(data, bins)

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, projection='polar')
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # Стиль як у вашому референсі (Teal, напівпрозорий)
    ax.bar(theta, hist, width=width, alpha=0.6, color='teal', label='Current', edgecolor='white', linewidth=0.5)

    # --- Далі код малювання ліній статистики без змін ---

    # Поточні circular-статистики
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

    # Малюємо поточні лінії (Current)
    if curr_mean is not None and np.isfinite(curr_mean):
        ax.plot([curr_mean, curr_mean], [eps, r_max], color='yellow', linewidth=2.5, label='Current Mean', zorder=10)
    if curr_q1 is not None and np.isfinite(curr_q1):
        ax.plot([curr_q1, curr_q1], [eps, r_max], color='orange', linestyle='--', linewidth=2, label='Current Q1',
                zorder=10)
    if curr_q3 is not None and np.isfinite(curr_q3):
        ax.plot([curr_q3, curr_q3], [eps, r_max], color='purple', linestyle='--', linewidth=2, label='Current Q3',
                zorder=10)

    # Малюємо цільові лінії (Target)
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

    # Виносимо легенду трохи за межі кола, щоб не перекривала дані
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
        'Q1': df.quantile(0.25),
        'Q3': df.quantile(0.75),
        'IQR': df.quantile(0.75) - df.quantile(0.25)
    }

    # Circular статистики для Orientation
    if "Orientation" in df.columns:
        circ = compute_circular_stats(df["Orientation"].values)
        stats['Mean']["Orientation"] = circ["Mean"]
        stats['Std']["Orientation"] = circ["Std"]
        stats['Q1']["Orientation"] = circ["Q1"]
        stats['Q3']["Orientation"] = circ["Q3"]
        stats['IQR']["Orientation"] = circ["IQR"]

    print(f"Статистика властивостей:\n{stats}")
    df.to_csv(output_props_file, index=False)
    logging.info(f"Файл з властивостями збережено як: {output_props_file}")
    return stats, df


def calculate_smape(actual, predicted):
    smape_values = [
        100 * abs(a - p) / ((abs(a) + abs(p)) / 2)
        for a, p in zip(actual, predicted) if (a != 0 or p != 0)
    ]
    return np.mean(smape_values)


def calculate_mse(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mse = np.mean((actual - predicted) ** 2)
    denom = np.mean(actual ** 2)

    if denom == 0:
        return np.nan
    return 100 * mse / denom


def calculate_mspe(actual, predicted):
    vals = []
    for a, p in zip(actual, predicted):
        if a == 0:
            continue
        vals.append(((a - p) / a) ** 2)
    if len(vals) == 0:
        return 0.0
    return 100.0 * float(np.mean(vals))


def minimize_properties(params):
    concentration, halfaxis_b, halfaxis_c, ellipse_order = params
    halfaxis_a = (halfaxis_b + halfaxis_c) / 2

    # TODO: шлях до файлів
    output_file = r"D:\University\MatViz\1FULL_MINIMIZE_SCRIPTS\minimization\cube_output.csv"
    output_props_file = r"D:\University\MatViz\1FULL_MINIMIZE_SCRIPTS\minimization\properties_output.csv"

    # log_run_parameters(params)

    # Запуск генерації структури
    generated_file = start(FIXED_SIZE, concentration, halfaxis_a, halfaxis_b, halfaxis_c,
                           0, 0, 0,
                           1, ellipse_order, output_file)
    if not generated_file:
        logging.error("Не вдалося створити файл.")
        return np.inf

    stats, df = process_and_calculate(generated_file, output_props_file)

    if stats is None or df is None:
        logging.error("Не вдалося обробити статистику.")
        return np.inf

    # TODO: Повні цільові значення
    # AZ31_iA
    # region
    target_values = {

        'Area': {
            'Mean': 0.0032, 'Std': 0.0021, 'Median': 0.0030, 'Mode': 0.0004,
            'Q1': 0.0016, 'Q3': 0.0042, 'IQR': 0.0026, 'Range': 0.0114
        },

        'Shape Factor': {
            'Mean': 0.8199, 'Std': 0.0732, 'Median': 0.8168, 'Mode': 0.6264,
            'Q1': 0.7817, 'Q3': 0.8630, 'IQR': 0.0813, 'Range': 0.3700
        },

        'ECR': {
            'Mean': 0.0302, 'Std': 0.0109, 'Median': 0.0310, 'Mode': 0.0113,
            'Q1': 0.0226, 'Q3': 0.0366, 'IQR': 0.0140, 'Range': 0.0532
        },

        'scale_factor': {
            'Mean': 1.6718, 'Std': 0.3385, 'Median': 1.6185, 'Mode': 1.1149,
            'Q1': 1.4361, 'Q3': 1.8977, 'IQR': 0.4615, 'Range': 1.5233
        },

        'Aspect Ratio': {
            'Mean': 1.3377, 'Std': 0.1039, 'Median': 1.3254, 'Mode': 1.2500,
            'Q1': 1.2723, 'Q3': 1.4007, 'IQR': 0.1285, 'Range': 0.6195
        },

        'Compactness Ratio': {
            'Mean': 1.0523, 'Std': 0.0219, 'Median': 1.0468, 'Mode': 1.0400,
            'Q1': 1.0376, 'Q3': 1.0625, 'IQR': 0.0249, 'Range': 0.1149
        },

        'area-to-ellipse Ratio': {
            'Mean': 0.7689, 'Std': 0.0094, 'Median': 0.7706, 'Mode': 0.7383,
            'Q1': 0.7649, 'Q3': 0.7745, 'IQR': 0.0096, 'Range': 0.0527
        },

        'Orientation': {
            'Mean': 1.5427, 'Std': 0.2925, 'Median': 1.5541,
            'Q1': 1.3919, 'Q3': 1.7066, 'IQR': 0.3148, 'Range': 2.1750
        },

        'Inertia Tensor XX': {
            'Mean': 67.1908, 'Std': 44.4945, 'Median': 63.2568, 'Mode': 4.8010,
            'Q1': 32.7500, 'Q3': 93.1957, 'IQR': 60.4457, 'Range': 243.1506
        },

        'Inertia Tensor XY': {
            'Mean': -0.2818, 'Std': 9.3603, 'Median': -0.5265, 'Mode': -32.8073,
            'Q1': -5.4456, 'Q3': 3.9165, 'IQR': 9.3621, 'Range': 61.9791
        },

        'Inertia Tensor YY': {
            'Mean': 29.1213, 'Std': 20.1992, 'Median': 27.1806, 'Mode': 1.0957,
            'Q1': 15.1038, 'Q3': 37.7492, 'IQR': 22.6454, 'Range': 110.6628
        },

        'Inertia Tensor/Area XX': {
            'Mean': 0.1311, 'Std': 0.0287, 'Median': 0.1296, 'Mode': 0.0721,
            'Q1': 0.1112, 'Q3': 0.1501, 'IQR': 0.0389, 'Range': 0.1436
        },

        'Inertia Tensor/Area XY': {
            'Mean': -0.0007, 'Std': 0.0181, 'Median': -0.0017, 'Mode': -0.0386,
            'Q1': -0.0131, 'Q3': 0.0099, 'IQR': 0.0230, 'Range': 0.0993
        },

        'Inertia Tensor/Area YY': {
            'Mean': 0.0553, 'Std': 0.0123, 'Median': 0.0532, 'Mode': 0.0313,
            'Q1': 0.0469, 'Q3': 0.0629, 'IQR': 0.0160, 'Range': 0.0598
        },

        'I_Principal_Max': {
            'Mean': 69.2874, 'Std': 45.0863, 'Median': 64.6991, 'Mode': 4.8972,
            'Q1': 34.3859, 'Q3': 96.5441, 'IQR': 62.1582, 'Range': 243.6486
        },

        'I_Principal_Min': {
            'Mean': 27.0247, 'Std': 19.0252, 'Median': 25.3775, 'Mode': 0.9995,
            'Q1': 13.2388, 'Q3': 35.5454, 'IQR': 22.3065, 'Range': 93.0971
        },

        'I_Anisotropy': {
            'Mean': 2.9094, 'Std': 1.2091, 'Median': 2.6196, 'Mode': 1.2430,
            'Q1': 2.0625, 'Q3': 3.6011, 'IQR': 1.5386, 'Range': 5.7173
        },

        'I_Area_Principal_Max': {
            'Mean': 0.1359, 'Std': 0.0276, 'Median': 0.1316, 'Mode': 0.0894,
            'Q1': 0.1159, 'Q3': 0.1532, 'IQR': 0.0373, 'Range': 0.1266
        },

        'I_Area_Principal_Min': {
            'Mean': 0.0506, 'Std': 0.0099, 'Median': 0.0505, 'Mode': 0.0310,
            'Q1': 0.0427, 'Q3': 0.0567, 'IQR': 0.0140, 'Range': 0.0422
        },

        'I_Area_Anisotropy': {
            'Mean': 2.9094, 'Std': 1.2091, 'Median': 2.6196, 'Mode': 1.2430,
            'Q1': 2.0625, 'Q3': 3.6011, 'IQR': 1.5386, 'Range': 5.7173
        }
    }

    # endregion

    # AZ31_iB
    # region
    # target_values = {
    #     'Norm Area': {
    #         'Mean': 0.0048, 'Std': 0.0030, 'Q1': 0.0024, 'Q3': 0.0065
    #     },
    #     'ECR': {
    #         'Mean': 0.0370, 'Std': 0.0122, 'Q1': 0.0275, 'Q3': 0.0454
    #     },
    #     'Aspect Ratio': {
    #         'Mean': 1.4239, 'Std': 0.1086, 'Q1': 1.3393, 'Q3': 1.4772
    #     },
    #     'Compactness Ratio': {
    #         'Mean': 1.0565, 'Std': 0.0181, 'Q1': 1.0429, 'Q3': 1.0685
    #     },
    #     'area-to-ellipse Ratio': {
    #         'Mean': 0.7596, 'Std': 0.0158, 'Q1': 0.7531, 'Q3': 0.7711
    #     },
    #     'scale_factor': {
    #         'Mean': 1.5945, 'Std': 0.2721, 'Q1': 1.4023, 'Q3': 1.7235
    #     }
    # }
    # endregion

    # WE43-0P
    # region
    # target_values = {
    #     'Norm Area': {
    #         'Mean': 0.0075, 'Std': 0.0046, 'Q1': 0.0037, 'Q3': 0.0106
    #     },
    #     'ECR': {
    #         'Mean': 0.0463, 'Std': 0.0158, 'Q1': 0.0343, 'Q3': 0.0581
    #     },
    #     'Aspect Ratio': {
    #         'Mean': 1.6393, 'Std': 0.1759, 'Q1': 1.5237, 'Q3': 1.7548
    #     },
    #     'Compactness Ratio': {
    #         'Mean': 1.1545, 'Std': 0.0809, 'Q1': 1.1038, 'Q3': 1.1828
    #     },
    #     'area-to-ellipse Ratio': {
    #         'Mean': 0.7289, 'Std': 0.0353, 'Q1': 0.7130, 'Q3': 0.7520
    #     },
    #     'scale_factor': {
    #         'Mean': 1.3756, 'Std': 0.2041, 'Q1': 1.1900, 'Q3': 1.5599
    #     }
    # }
    # endregion

    # збереження гістограм для поточної ітерації
    try:
        iteration = next(ITER_COUNTER)
        save_iteration_histograms(stats, df, target_values, iteration, output_folder)
    except Exception as e:
        logging.error(f"Не вдалося зберегти гістограми для ітерації: {e}")

    individual_errors = {}
    total_error = 0
    count = 0

    for feature_name in selected_features:
        if feature_name not in stats['Mean'] or feature_name not in target_values:
            continue

        # Вибір метрик
        if feature_name == 'Orientation':
            current_metrics = ['Mean', 'Std']
            # current_metrics = ['Mean', 'Q1', 'Q3']
        else:
            current_metrics = selected_metrics

        individual_errors[feature_name] = {}

        for metric in current_metrics:
            try:
                actual = stats[metric][feature_name]
                predicted = target_values[feature_name][metric]

                # --- ГІБРИДНА ЛОГІКА ---
                if feature_name == 'Orientation':
                    # Для Орієнтації використовуємо Абсолютну Різницю (MAE) * 100.
                    # Це дозволяє уникнути проблем SMAPE з малими значеннями Std.
                    # Множник 100 приводить радіани (~0.1-0.2) до масштабу відсотків (10-20).
                    error = abs(actual - predicted) * 30.0
                else:
                    # Для інших — класичний SMAPE (або те, що вибрано глобально)
                    if selected_metric_type == 'SMAPE':
                        error = calculate_smape([actual], [predicted])
                    elif selected_metric_type == 'MSE':
                        error = calculate_mse([actual], [predicted])
                    elif selected_metric_type == 'MSPE':
                        error = calculate_mspe([actual], [predicted])
                    else:
                        error = abs(actual - predicted)  # Fallback

                individual_errors[feature_name][metric] = error
                total_error += error
                count += 1
            except KeyError:
                continue

    if count == 0:
        return np.inf

    final_score = total_error / count

    print(f"Загальна помилка (Hybrid Score): {final_score:.2f}")
    print(f"Помилки по параметрах:")
    for feature, metrics in individual_errors.items():
        for metric, value in metrics.items():
            # Додаємо позначку, якщо це не % SMAPE
            unit = " (abs*100)" if feature == 'Orientation' else "%"
            print(f"{feature} - {metric}: {value:.2f}{unit}")

    log_iteration_to_csv(
        iteration=iteration,
        params=params,
        stats=stats,
        individual_errors=individual_errors,
        total_error=total_error
    )

    # Запис у файл summary
    summary_file = os.path.join(output_folder, "summary.txt")
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write("\n===========================\n")
        f.write(f"Параметри: {params}\n")
        f.write(f"Загальна помилка: {final_score:.2f}\n")
        for feature, metrics in individual_errors.items():
            for metric, value in metrics.items():
                unit = " (abs*100)" if feature == 'Orientation' else "%"
                f.write(f"{feature} - {metric}: {value:.2f}{unit}\n")

    return final_score


def find_best_starting_point(bounds):
    # Створення списку стартових точок
    x0_list = [
        [(low + high) / 2 for (low, high) in bounds],  # Центр
        [low + (high - low) * 0.25 for (low, high) in bounds],
        [low + (high - low) * 0.75 for (low, high) in bounds],
        [np.random.uniform(low, high) for (low, high) in bounds]
    ]

    best_x0 = None
    best_fun = float('inf')

    # Проходимо по кожній стартовій точці
    for i, x0 in enumerate(x0_list):
        logging.info(f"Оцінка стартової точки #{i + 1}: {x0}")

        # Виконуємо одну оцінку для поточної точки без оптимізації
        fun = minimize_properties(x0)
        logging.info(f"→ SMAPE = {fun:.6f}")

        # Вибираємо найкращу точку
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
        x0 = find_best_starting_point(bounds)

        result = minimize(
            minimize_properties,
            x0=x0,
            method=selected_method,
            bounds=bounds,
            options={'disp': True, 'maxiter': 100}
        )

    elif selected_method == 'basinhopping':
        print("Вибір найкращої стартової точки...")
        x0 = find_best_starting_point(bounds)

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
        result = dual_annealing(
            func=minimize_properties,
            bounds=bounds,
            maxiter=500,
            no_local_search=True
        )

    elif selected_method == 'Differential Evolution':
        result = differential_evolution(
            func=minimize_properties,
            bounds=bounds,
            strategy='best1bin',
            maxiter=5,
            tol=10.0,
            popsize=5,
            disp=True
        )

    elif selected_method == 'Manual Sweep':
        # TODO: sweep_param
        sweep_param = "halfaxis_c"
        sweep_index = param_names.index(sweep_param)

        sweep_results = manual_parameter_sweep(
            param_index=sweep_index,
            base_params=base_params,
            bounds=bounds,
            n_points=15
        )
        logging.info(f"Manual sweep finished")

    else:
        raise ValueError(f"Невідомий метод: {selected_method}")

    elapsed_time = time.time() - start_time

    print(f"Optimization completed in {elapsed_time:.2f} seconds.")
    print(f"Best parameters found: {result.x}")
    print(f"Final SMAPE error: {result.fun}")

    summary_file = os.path.join(output_folder, "summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Optimization method: {selected_method}\n")
        f.write(f"Initial point: {result.x.tolist() if hasattr(result, 'x') else 'N/A'}\n")
        f.write(f"Final SMAPE error: {result.fun:.6f}\n")
        f.write(f"Total optimization time: {elapsed_time:.2f} seconds\n")

    return result


# Виклик основної функції
def main():
    result = optimize_properties()
    print(result)


if __name__ == "__main__":
    main()
