import csv
import logging
import os
import shutil

import uuid

import sys
import time
from multiprocessing import Manager
import threading

import matplotlib

matplotlib.use('Agg')
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
selected_method = 'Manual Sweep'
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
selected_metrics = ['Mean', 'Q1', 'Q3', 'Std']

# TODO: Фіксований розмір куба
FIXED_SIZE = 130

# ============================================================
# ГЛОБАЛЬНА ЗМІННА ДЛЯ ВИБОРУ ПАРАМЕТРІВ ОПТИМІЗАЦІЇ
# ============================================================
# Назви всіх параметрів (у тому ж порядку як в bounds та base_params)
ALL_PARAM_NAMES = [
    'concentration',
    'halfaxis_a',
    'halfaxis_b',
    'halfaxis_c',
    'ellipse_order',
    'wave_coefficient',
    'wave_spread',
    'initial_nuclei_count',
    'orientation_angle_a',
    'orientation_angle_b',
    'orientation_angle_c',
    'stefan_number'
]

# ВИБІР: які параметри оптимізувати
# Варіанти:
#   - None (або 'all'): оптимізувати ВСІ параметри
#   - список індексів: [0, 1, 5, 7] - оптимізувати тільки ці
#   - список імен: ['concentration', 'halfaxis_a', 'wave_coefficient']
OPTIMIZE_PARAMS = 'all'  # Оптимізувати ВСІ параметри

# OPTIMIZE_PARAMS = [0, 4, 5, 6]  # Приклад: оптимізувати індекси 0, 4, 5, 6
# OPTIMIZE_PARAMS = ['concentration', 'halfaxis_a', 'wave_coefficient', 'wave_spread']  # Приклад: по імені

# TODO: Обмеження
bounds = [
    (0.01, 0.2),  # 0: concentration
    (0.65, 1.7),  # 1: halfaxis_a
    (0.65, 1.7),  # 2: halfaxis_b
    (0.65, 1.7),  # 3: halfaxis_c
    (1, 6),  # 4: ellipse_order
    (1, 60),  # 5: wave_coefficient
    (0.1, 50),  # 6: wave_spread
    (1, 1000),  # 7: initial_nuclei_count
    (0, 360),  # 8: orientation_angle_a
    (0, 360),  # 9: orientation_angle_b
    (0, 360),  # 10: orientation_angle_c
    (0.01, 1000.0)  # 11: stefan_number
]

base_params = np.array([
    0.07,  # 0: concentration
    1.7,  # 1: halfaxis_a
    1.7,  # 2: halfaxis_b
    0.8,  # 3: halfaxis_c
    6,  # 4: ellipse_order
    20.0,  # 5: wave_coefficient
    50,  # 6: wave_spread
    1,  # 7: initial_nuclei_count
    0.0,  # 8: orientation_angle_a
    0.0,  # 9: orientation_angle_b
    0.0,  # 10: orientation_angle_c
    100.0  # 11: stefan_number
], dtype=float)


# ============================================================
# НОВІ ФУНКЦІЇ ДЛЯ ОБРОБКИ OPTIMIZE_PARAMS
# ============================================================
def get_optimize_indices():
    """
    Повертає список індексів параметрів, які потрібно оптимізувати
    """
    global OPTIMIZE_PARAMS, ALL_PARAM_NAMES

    if OPTIMIZE_PARAMS is None or OPTIMIZE_PARAMS == 'all':
        # Оптимізувати ВСІ параметри
        return list(range(len(ALL_PARAM_NAMES)))

    elif isinstance(OPTIMIZE_PARAMS, list):
        if all(isinstance(x, int) for x in OPTIMIZE_PARAMS):
            # Список індексів
            return OPTIMIZE_PARAMS
        elif all(isinstance(x, str) for x in OPTIMIZE_PARAMS):
            # Список імен - конвертуємо на індекси
            indices = []
            for name in OPTIMIZE_PARAMS:
                if name in ALL_PARAM_NAMES:
                    indices.append(ALL_PARAM_NAMES.index(name))
                else:
                    raise ValueError(f"Невідомий параметр: {name}")
            return indices
        else:
            raise ValueError("OPTIMIZE_PARAMS повинен містити тільки числа або тільки строки")
    else:
        raise ValueError("OPTIMIZE_PARAMS повинен бути None, 'all', список індексів або список імен")


def get_optimizable_bounds_and_indices():
    """
    Повертає:
    - bounds для оптимізації (тільки для вибраних параметрів)
    - indices (які індекси від всіх параметрів ми оптимізуємо)
    """
    indices = get_optimize_indices()
    opt_bounds = [bounds[i] for i in indices]
    return opt_bounds, indices


def reconstruct_full_params(optimized_values, optimize_indices):
    """
    Реконструює повний масив параметрів з оптимізованих значень

    Args:
        optimized_values: масив значень, які були оптимізовані
        optimize_indices: індекси параметрів, які були оптимізовані

    Returns:
        повний масив всіх параметрів
    """
    full_params = base_params.copy()
    for i, idx in enumerate(optimize_indices):
        full_params[idx] = optimized_values[i]
    return full_params


# TODO: Ініціалізація запуска MatViz3D
exe_path = r"D:\Project(MatViz3D)\Random\current_build\test\MatViz3D.exe"
launcher = MatViz3DLauncher(exe_path)

# TODO:  Створення каталогу для виведення
output_folder = r"D:\Project(MatViz3D)\Random\Test\stephan"
os.makedirs(output_folder, exist_ok=True)

# TODO: Опції аналізу шарів
# mode: 'all' - аналіз усіх шарів (поточна поведінка)
#       'sample' - аналізувати 1 шар, пропустити SAMPLE_SKIP шарів
#       'single' - аналізувати тільки один шар із індексом SINGLE_LAYER_INDEX
ANALYSIS_MODE = 'sample'  # 'all', 'sample', 'single'
SAMPLE_SKIP = 10  # якщо mode == 'sample', то після кожного проаналізованого шару пропускаємо SAMPLE_SKIP шарів
SAMPLE_OFFSET = 0  # початковий індекс для sample (0..cube_size-1)
SINGLE_LAYER_INDEX = 0  # індекс шару для mode == 'single'

# TODO: Обрати осі для розрізання
# ['x'], ['y'], ['z'], ['x', 'y'], ['x', 'z'], ['y', 'z'], або ['x', 'y', 'z']
SELECTED_AXES = ['x']  # за замовчуванням — тільки по осі Z
# SELECTED_AXES = ['y']

# ========== НОВИЙ ПІДХІД ДЛЯ СИНХРОНІЗАЦІЇ ==========
_iteration_manager = None
ITER_COUNTER_DICT = None
FILE_WRITE_LOCK = threading.Lock()
ITER_COUNTER_LOCK = None

# TODO: Перемикач для копіювання результатів симуляції (cube та properties)
SAVE_FULL_ITERATION_DATA = True  # True — копіювати файли, False — ні

# True = працюємо в логарифмічному просторі (Log-Normal distribution)
# False = звичайний лінійний простір
USE_LOG_SPACE = False

# Шляхи до файлів з цільовими значеннями
TARGET_FILE_NORMAL = r"D:\University\MatViz\1FULL_MINIMIZE_SCRIPTS\Target_img\AZ31_iA\processed_output\statistics_image_properties_(AZ31_imgA).csv"

TARGET_FILE_LOG = r"D:\University\MatViz\1FULL_MINIMIZE_SCRIPTS\Target_img\AZ31_iA\processed_output\Arcsinh_statistics_image_properties_(AZ31_imgA).csv"

TARGET_FILE_DIST = r"D:\University\MatViz\1FULL_MINIMIZE_SCRIPTS\Target_img\AZ31_iA\processed_output\processed_image_properties_(AZ31_imgA).csv"

TARGET_FILE_DIST_LOG = r"D:\University\MatViz\1FULL_MINIMIZE_SCRIPTS\Target_img\AZ31_iA\processed_output\processed_Arcsinh_image_properties_(AZ31_imgA).csv"

if selected_metric_type == 'Energy Distance':
    target_csv_path = TARGET_FILE_DIST_LOG if USE_LOG_SPACE else TARGET_FILE_DIST
    print(f"Обрана метрика: Energy Distance. Завантажуємо повний розподіл.")
else:
    target_csv_path = TARGET_FILE_LOG if USE_LOG_SPACE else TARGET_FILE_NORMAL
    print(f"Обрана метрика: {selected_metric_type}. Завантажуємо статистику.")

print(f"Цільовий файл: {target_csv_path}")

# Виведення інформації про оптимізацію
optimize_indices = get_optimize_indices()
print(f"\n{'=' * 60}")
print(f"ПАРАМЕТРИ ОПТИМІЗАЦІЇ:")
print(f"{'=' * 60}")
print(f"Всього параметрів: {len(ALL_PARAM_NAMES)}")
print(f"Параметрів для оптимізації: {len(optimize_indices)}")
print(f"Параметри для оптимізації:")
for idx in optimize_indices:
    print(f"  - {idx}: {ALL_PARAM_NAMES[idx]} (діапазон: {bounds[idx]})")
print(f"\nЗафіксовані параметри (на значеннях з base_params):")
for idx in range(len(ALL_PARAM_NAMES)):
    if idx not in optimize_indices:
        print(f"  - {idx}: {ALL_PARAM_NAMES[idx]} = {base_params[idx]}")
print(f"{'=' * 60}\n")

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
    "angle_a", "angle_b", "angle_c",
    "stefan_number",
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


def get_next_iteration():
    """Безпечно отримує наступний номер ітерації з синхронізацією між процесами"""
    with ITER_COUNTER_LOCK:
        current_value = ITER_COUNTER_DICT['value']
        ITER_COUNTER_DICT['value'] = current_value + 1
        return current_value + 1


def log_iteration_to_csv(
        iteration: int,
        params,
        stats,
        individual_errors,
        total_error
):
    # Використовуємо Lock для безпечного запису в файл при паралельному виконанні
    with FILE_WRITE_LOCK:
        file_exists = os.path.isfile(csv_params_path)

        (concentration, halfaxis_a, halfaxis_b, halfaxis_c, ellipse_order,
         wave_coefficient, wave_spread, initial_nuclei_count,
         angle_a, angle_b, angle_c, stefan_number) = params

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
            "angle_a": angle_a,
            "angle_b": angle_b,
            "angle_c": angle_c,
            "stefan_number": stefan_number,
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
    "concentration", "halfaxis_a", "halfaxis_b", "halfaxis_c",
    "ellipse_order", "wave_coefficient", "wave_spread", "initial_nuclei_count",
    "orientation_angle_a", "orientation_angle_b", "orientation_angle_c",
    "stefan_number"
]


# ============================================================
# ФУНКЦІЇ ОБРОБЛЕННЯ ДАНИХ
# ============================================================

def process_data(df):
    """Фільтрує дані для видалення виключень та неправильних значень"""
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


def compute_circular_stats(values):
    """Обчислює статистику для кутових даних"""
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


def save_iteration_files(iteration: int, cube_src: str, props_src: str, output_dir: str):
    """Копіює вихідний куб та розраховані властивості в папку ітерації."""
    if not SAVE_FULL_ITERATION_DATA:
        return

    # Створюємо шлях: output_folder/iteration_data/0001/
    data_root = os.path.join(output_dir, "iteration_data")
    iter_folder = os.path.join(data_root, f"{iteration:04d}")
    os.makedirs(iter_folder, exist_ok=True)

    try:
        # Копіюємо cube_output.csv
        if os.path.exists(cube_src):
            shutil.copy2(cube_src, os.path.join(iter_folder, "cube_output.csv"))

        # Копіюємо properties_output.csv
        if os.path.exists(props_src):
            shutil.copy2(props_src, os.path.join(iter_folder, "properties_output.csv"))

        logging.info(f"Iteration {iteration}: Full data copied to {iter_folder}")
    except Exception as e:
        logging.error(f"Не вдалося скопіювати файли для ітерації {iteration}: {e}")


def manual_parameter_sweep(
        param_index: int,
        base_params: np.ndarray,
        bounds: list,
        optimize_indices: list,
        n_points: int = 15
):
    """Проводит ручной поиск параметра с учетом оптимизируемых параметров"""
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
            # Если параметр в списке оптимизации, передаём только оптимизированные значения
            if param_index in optimize_indices:
                opt_values = [params[idx] for idx in optimize_indices]
                error = minimize_properties(np.array(opt_values))
            else:
                # Если параметр не в списке оптимизации, используем полный массив параметров
                # но передаём пустой массив, так как sweep не требует оптимизации
                error = minimize_properties(np.array([params[idx] for idx in optimize_indices]))

            results.append((val, error))
        except Exception as e:
            logging.error(f"Sweep failed at {val}: {e}")
            results.append((val, np.nan))

    return results


def start(size, concentration, halfaxis_a, halfaxis_b, halfaxis_c, orientation_angle_a, orientation_angle_b,
          orientation_angle_c, wave_coefficient, wave_spread, initial_nuclei_count, ellipse_order, stefan_number,
          output_file):
    print(
        f"Запуск MatViz3D з параметрами: [{size}, conc = {concentration}, "
        f"h_a = {halfaxis_a}, h_b = {halfaxis_b}, h_c = {halfaxis_c}, "
        f"or_a = {orientation_angle_a}, or_b = {orientation_angle_b}, or_c = {orientation_angle_c}, "
        f"wc = {wave_coefficient}, wave_spread = {wave_spread}, initial_nuclei_count = {initial_nuclei_count}, "
        f"ellipse_order = {ellipse_order}, stefan_number = {stefan_number}]")

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
        stefan_number=stefan_number,
        output_file=output_file
    )


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

    grains = []

    if analysis_mode == 'all':
        indices = list(range(cube_size))
    elif analysis_mode == 'sample':
        indices = list(range(sample_offset, cube_size, sample_skip + 1))
    elif analysis_mode == 'single':
        indices = [single_index]
    else:
        indices = list(range(cube_size))

    for axis in SELECTED_AXES:
        for idx in indices:
            try:
                if axis == 'z':
                    layer = layers[:, :, idx]
                elif axis == 'x':
                    layer = layers[idx, :, :]
                elif axis == 'y':
                    layer = layers[:, idx, :]
                else:
                    continue

                layer_area = np.prod(layer.shape)
                unique_colors = set(layer.flatten())

                for grain_color in unique_colors:
                    if grain_color == 0:
                        continue

                    grain_mask = (layer == grain_color)
                    labeled_grains = skimage.measure.label(grain_mask, connectivity=2)

                    for region in regionprops(labeled_grains):
                        if region.area <= 20:
                            continue

                        props = {}
                        area = region.area
                        norm_area = area / layer_area

                        if 'Norm Area' in selected_features:
                            props['Norm Area'] = norm_area
                        if 'ECR' in selected_features:
                            props['ECR'] = np.sqrt(norm_area / np.pi)
                        if 'Orientation' in selected_features:
                            props['Orientation'] = np.degrees(float(region.orientation))

                        # Inertia tensor
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

                        if all(np.isfinite(list(props.values()))):
                            grains.append(props)

            except Exception as e:
                logging.error(f"Error processing layer {idx} along {axis}: {e}")
                continue

    df = pd.DataFrame(grains)
    if df.empty:
        logging.error("Не вдалося збирти дані про зерна")
        return None, None

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Застосовуємо фільтр process_data
    df = process_data(df)

    if df.empty:
        logging.error("Дані порожні після фільтрування")
        return None, None

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


def save_iteration_histograms(stats, df, target_data, iteration, output_folder):
    """Зберігає гістограми розподілу для ітерації"""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    try:
        # Обираємо 2-3 ключові властивості для графіків
        selected_cols = [col for col in df.columns if col in selected_features][:3]

        if not selected_cols:
            logging.warning("Немає колонок для графіків")
            return

        fig, axes = plt.subplots(1, len(selected_cols), figsize=(15, 5))
        if len(selected_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, selected_cols):
            if col in df.columns:
                ax.hist(df[col].values, bins=30, alpha=0.7, label='Simulation')

                if isinstance(target_data, pd.DataFrame) and col in target_data.columns:
                    ax.hist(target_data[col].values, bins=30, alpha=0.7, label='Target')

                ax.set_title(f'{col}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.legend()

        plt.tight_layout()
        output_file = os.path.join(output_folder, f"iteration_{iteration}_histograms.png")
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logging.error(f"Не вдалося зберегти гістограми: {e}")


def minimize_properties(x):
    """
    Функція мінімізації - автоматично обробляє вибірку параметрів

    Args:
        x: масив значень для оптимізації (тільки для вибраних параметрів)

    Returns:
        значення функції помилки
    """
    # Реконструюємо повний набір параметрів
    optimize_indices = get_optimize_indices()
    full_params = reconstruct_full_params(x, optimize_indices)

    # ВАЖЛИВО: Отримуємо унікальний номер ітерації на початку функції
    iteration = get_next_iteration()

    (concentration, halfaxis_a, halfaxis_b, halfaxis_c, ellipse_order,
     wave_coefficient, wave_spread, initial_nuclei_count,
     angle_a, angle_b, angle_c, stefan_number) = full_params

    initial_nuclei_count = int(round(initial_nuclei_count))

    # Створюємо унікальний ідентифікатор для цього запуску
    run_id = str(uuid.uuid4())[:8]
    output_file = os.path.join(output_folder, f"cube_output_{run_id}.csv")
    output_props_file = os.path.join(output_folder, f"properties_output_{run_id}.csv")

    generated_file = start(
        FIXED_SIZE, concentration, halfaxis_a, halfaxis_b, halfaxis_c,
        angle_a, angle_b, angle_c,
        wave_coefficient, wave_spread, initial_nuclei_count, ellipse_order, stefan_number,
        output_file
    )

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
        save_iteration_histograms(stats, df, target_data, iteration, output_folder)
        save_iteration_files(
            iteration=iteration,
            cube_src=generated_file,
            props_src=output_props_file,
            output_dir=output_folder
        )
    except Exception as e:
        logging.error(f"Не вдалося зберегти дані для ітерації: {e}")

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
        params=full_params,
        stats=stats,
        individual_errors=individual_errors,
        total_error=total_error
    )

    summary_file = os.path.join(output_folder, "summary.txt")
    with FILE_WRITE_LOCK:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write("\n===========================\n")
            f.write(f"№ Ітерації: {iteration}\n")
            f.write(f"Загальна похибка: {total_error}\n")
            f.write(f"Параметри: {full_params}\n")

            for feature, metrics in individual_errors.items():
                if feature == "Multivariate_Energy_Distance":
                    val = metrics.get('All', 0.0)
                    f.write(f"Energy Distance calculated on features {valid_features}: {val:.4f}\n")
                    continue

                for metric, value in metrics.items():
                    f.write(f"{feature} - {metric}: {value:.2f}\n")

        # Видалення тимчасових файлів під захистом Lock
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
            if os.path.exists(output_props_file):
                os.remove(output_props_file)
            if generated_file and os.path.exists(generated_file):
                os.remove(generated_file)
        except Exception as e:
            logging.warning(f"Не вдалося видалити тимчасові файли: {e}")

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

        fun = minimize_properties(np.array(x0))
        logging.info(f"→ SMAPE = {fun:.6f}")

        if fun < best_fun:
            best_fun = fun
            best_x0 = x0

    return best_x0


def optimize_properties():
    """
    Основна функція оптимізації з підтримкою вибірки параметрів
    """
    opt_bounds, optimize_indices = get_optimizable_bounds_and_indices()
    scipy_bounds = Bounds([b[0] for b in opt_bounds], [b[1] for b in opt_bounds])

    print(f"Вибраний метод: {selected_method}")
    start_time = time.time()

    if selected_method in ['SLSQP', 'L-BFGS-B']:
        print("Вибір найкращої стартової точки...")
        # x0 = find_best_starting_point(opt_bounds)
        x0 = [base_params[i] for i in optimize_indices]

        result = minimize(
            minimize_properties,
            x0=x0,
            method=selected_method,
            bounds=opt_bounds,
            options={'disp': True, 'maxiter': 100}
        )

    elif selected_method == 'basinhopping':
        print("Вибір найкращої стартової точки...")
        # x0 = find_best_starting_point(opt_bounds)
        x0 = [base_params[i] for i in optimize_indices]

        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": opt_bounds,
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
        x0 = [base_params[i] for i in optimize_indices]
        result = dual_annealing(
            func=minimize_properties,
            bounds=opt_bounds,
            x0=x0,
            maxiter=1,
            no_local_search=True
        )

    elif selected_method == 'Differential Evolution':
        result = differential_evolution(
            func=minimize_properties,
            bounds=opt_bounds,
            strategy='best1bin',
            maxiter=30,
            popsize=5,
            mutation=(0.5, 1),
            recombination=0.7,
            tol=0.1,
            polish=False,
            disp=True,
            workers=-1,
            updating='deferred'
        )

    elif selected_method == 'Manual Sweep':
        # TODO: sweep_param
        sweep_param = "stefan_number"
        if sweep_param in ALL_PARAM_NAMES:
            sweep_index = ALL_PARAM_NAMES.index(sweep_param)
            if sweep_index in optimize_indices:
                local_index = optimize_indices.index(sweep_index)
            else:
                raise ValueError(f"Параметр {sweep_param} не в списку для оптимізації")
        else:
            raise ValueError(f"Невідомий параметр: {sweep_param}")

        sweep_results = manual_parameter_sweep(
            param_index=sweep_index,
            base_params=base_params,
            bounds=bounds,
            optimize_indices=optimize_indices,
            n_points=20
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
        f.write(f"Optimized parameters: {[ALL_PARAM_NAMES[i] for i in optimize_indices]}\n")
        f.write(f"Initial point: {result.x.tolist() if hasattr(result, 'x') else 'N/A'}\n")
        f.write(f"Final SMAPE error: {result.fun:.6f}\n")
        f.write(f"Total optimization time: {elapsed_time:.2f} seconds\n")

    return result


def main():
    global _iteration_manager, ITER_COUNTER_DICT, ITER_COUNTER_LOCK

    # Ініціалізація менеджера та локів ТІЛЬКИ в головному процесі
    _iteration_manager = Manager()
    ITER_COUNTER_DICT = _iteration_manager.dict()
    ITER_COUNTER_DICT['value'] = 0
    ITER_COUNTER_LOCK = _iteration_manager.Lock()

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
        # Закриваємо менеджер після завершення роботи
        _iteration_manager.shutdown()


if __name__ == "__main__":
    main()
