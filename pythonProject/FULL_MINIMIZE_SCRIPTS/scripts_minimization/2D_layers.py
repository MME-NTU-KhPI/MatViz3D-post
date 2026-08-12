import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from skimage.measure import regionprops, label

# --- ВЛАСТИВОСТІ ДЛЯ АНАЛІЗУ ---
# Перемикач: True - використовувати локальні властивості, False - імпортувати з min.py
USE_LOCAL_FEATURES = True
LOCAL_FEATURES = ['Area', 'Inertia Tensor/Area XX', 'Inertia Tensor/Area XY', 'Inertia Tensor/Area YY']

if USE_LOCAL_FEATURES:
    selected_features = LOCAL_FEATURES
    print(f"Використовуються локальні налаштування властивостей: {selected_features}")
else:
    try:
        from min import selected_features

        print(f"Успішно імпортовано selected_features з min.py: {selected_features}")
    except ImportError:
        print("Попередження: Не вдалося знайти min.py. Використовуються локальні налаштування за замовчуванням.")
        selected_features = LOCAL_FEATURES

# --- ANALYSIS SETTINGS ---
# Намагаємося імпортувати налаштування зрізів з min.py
try:
    from min import ANALYSIS_MODE, SAMPLE_SKIP, SAMPLE_OFFSET, SINGLE_LAYER_INDEX, SELECTED_AXES

    print("Успішно імпортовано налаштування зрізів (ANALYSIS SETTINGS) з min.py")
except ImportError:
    print("Налаштування зрізів не знайдені в min.py. Використовуються значення за замовчуванням.")
    ANALYSIS_MODE = 'sample'  # 'all', 'sample', 'single'
    SAMPLE_SKIP = 10
    SAMPLE_OFFSET = 0
    SINGLE_LAYER_INDEX = 0

    SELECTED_AXES = ['x']  # 'x', 'y', 'z'


# --- ФУНКЦІЯ ОБЧИСЛЕННЯ ВЛАСТИВОСТЕЙ ---
def get_property_value(prop, feature_name, layer_area):
    """
    Обчислює конкретну властивість за назвою.
    Ця версія 100% повторює логіку обчислення властивостей з min.py.
    """
    try:
        area = prop.area
        if area == 0: return None

        if feature_name == 'Area':
            return area

        norm_area = area / layer_area if layer_area else 0

        if feature_name == 'Norm Area':
            return norm_area
        if feature_name == 'ECR':
            return np.sqrt(norm_area / np.pi)

        if feature_name in ['Aspect Ratio', 'Compactness Ratio', 'scale_factor']:
            if feature_name == 'Compactness Ratio':
                return prop.convex_area / area
            else:
                minor = prop.minor_axis_length
                if minor == 0: return None
                major = prop.major_axis_length
                return major / minor

        if feature_name == 'Orientation':
            return np.degrees(float(prop.orientation))

        # Розрахунок Inertia Tensor
        it = prop.inertia_tensor
        xx, yy, xy = it[0, 0], it[1, 1], it[0, 1]

        if feature_name == 'Inertia Tensor XX': return xx
        if feature_name == 'Inertia Tensor XY': return xy
        if feature_name == 'Inertia Tensor YY': return yy

        if feature_name == 'Inertia Tensor/Area XX': return xx / area
        if feature_name == 'Inertia Tensor/Area XY': return xy / area
        if feature_name == 'Inertia Tensor/Area YY': return yy / area

        if feature_name in ['I_Principal_Max', 'I_Principal_Min', 'I_Anisotropy']:
            l1 = (xx + yy) / 2 + np.sqrt(((xx - yy) / 2) ** 2 + xy ** 2)
            l2 = (xx + yy) / 2 - np.sqrt(((xx - yy) / 2) ** 2 + xy ** 2)
            if feature_name == 'I_Principal_Max': return l1
            if feature_name == 'I_Principal_Min': return l2
            if feature_name == 'I_Anisotropy': return l1 / max(l2, 1e-12)

        if feature_name in ['I_Area_Principal_Max', 'I_Area_Principal_Min', 'I_Area_Anisotropy']:
            axx, ayy, axy = xx / area, yy / area, xy / area
            l1 = (axx + ayy) / 2 + np.sqrt(((axx - ayy) / 2) ** 2 + axy ** 2)
            l2 = (axx + ayy) / 2 - np.sqrt(((axx - ayy) / 2) ** 2 + axy ** 2)
            if feature_name == 'I_Area_Principal_Max': return l1
            if feature_name == 'I_Area_Principal_Min': return l2
            if feature_name == 'I_Area_Anisotropy': return l1 / max(l2, 1e-12)

        return None
    except Exception:
        return None


# --- 3D SETTINGS ---
ENABLE_3D_STACK_VISUALIZATION = False
SAVE_INDIVIDUAL_2D_SLICES = True
SLICE_THICKNESS = 1

# --- LABEL SETTINGS ---
ENABLE_GRAIN_LABELS = True
FONT_SIZE = 7  # Розмір шрифту
FONT_COLOR = 'black'  # Колір шрифту

# Словник для коротких і зрозумілих підписів властивостей на зображеннях
FEATURE_SHORT_NAMES = {
    'Area': 'A',
    'Norm Area': 'N.Area',
    'ECR': 'ECR',
    'Aspect Ratio': 'AR',
    'Compactness Ratio': 'Comp',
    'scale_factor': 'Scale',
    'Orientation': 'Ang(°)',
    'Inertia Tensor XX': 'Ixx',
    'Inertia Tensor XY': 'Ixy',
    'Inertia Tensor YY': 'Iyy',
    'Inertia Tensor/Area XX': 'Ixx/A',
    'Inertia Tensor/Area XY': 'Ixy/A',
    'Inertia Tensor/Area YY': 'Iyy/A',
    'I_Principal_Max': 'I_max',
    'I_Principal_Min': 'I_min',
    'I_Anisotropy': 'I_anis',
    'I_Area_Principal_Max': 'Ia_max',
    'I_Area_Principal_Min': 'Ia_min',
    'I_Area_Anisotropy': 'Ia_anis'
}

# --- EXPORT SETTINGS ---
FORMATS_3D = ['png']  # Formats for 3D stack (e.g., ['png', 'svg', 'pdf'])
FORMATS_2D = ['png']  # Formats for 2D slices
DPI_SETTING = 600  # Resolution (300 is good, 600 is excellent for print)

# --- CAMERA ANGLE SETTINGS ---
CAMERA_ANGLES = {
    'default': (30, -60),
    'isometric': (35.264, 45),
    'dimetric': (20, 60),
    'trimetric': (25, -50),
    'top_down': (90, -90),
    'front': (0, -90),
    'side': (0, 0)
}
SELECTED_ANGLES = ['isometric']


def load_volume(csv_path):
    """
    Завантажує 3D об'єм з CSV файлу (формат X;Y;Z;Colors).
    """
    print(f"Завантаження об'єму з {csv_path}...")
    try:
        df = pd.read_csv(csv_path, sep=';')

        if not all(col in df.columns for col in ['X', 'Y', 'Z', 'Colors']):
            print("Помилка: CSV файл повинен містити колонки X, Y, Z, Colors")
            return None

        max_coords = df[['X', 'Y', 'Z']].max()
        shape = (int(max_coords['X']) + 1, int(max_coords['Y']) + 1, int(max_coords['Z']) + 1)

        volume = np.zeros(shape, dtype=int)
        volume[df['X'].values, df['Y'].values, df['Z'].values] = df['Colors'].values
        return volume
    except Exception as e:
        print(f"Помилка завантаження: {e}")
        return None


def create_old_colormap(n_colors):
    hues = np.random.rand(n_colors)
    sats = np.random.uniform(0.6, 1.0, n_colors)
    vals = np.random.uniform(0.85, 1.0, n_colors)

    hsv_colors = np.column_stack((hues, sats, vals))
    rgb_colors = mcolors.hsv_to_rgb(hsv_colors)

    np.random.shuffle(rgb_colors)

    # фон чорний
    rgb_colors[0] = [0, 0, 0]

    return ListedColormap(rgb_colors)


def visualize_slices(volume, mode='sample', sample_skip=10, sample_offset=0, single_index=0, selected_axes=None,
                     output_dir='slices'):
    if selected_axes is None:
        selected_axes = ['x']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    axes_map = {'x': 0, 'y': 1, 'z': 2}

    # --- Відновлюємо оригінальну палітру зі старого коду ---
    max_grain_id = int(np.max(volume)) if np.max(volume) > 0 else 1
    original_cmap = create_old_colormap(max_grain_id + 1)

    for axis in selected_axes:
        if axis not in axes_map:
            print(f"Невідома вісь: {axis}")
            continue

        axis_idx = axes_map[axis]
        num_slices = volume.shape[axis_idx]

        if mode == 'all':
            indices = range(num_slices)
        elif mode == 'sample':
            indices = range(sample_offset, num_slices, sample_skip)
        elif mode == 'single':
            indices = [single_index] if single_index < num_slices else []
        else:
            indices = []

        count = 1
        for i in indices:
            if axis == 'x':
                slice_data = volume[i, :, :]
                xlabel, ylabel = 'Z', 'Y'
            elif axis == 'y':
                slice_data = volume[:, i, :]
                xlabel, ylabel = 'Z', 'X'
            else:
                slice_data = volume[:, :, i]
                xlabel, ylabel = 'Y', 'X'

            fig, ax = plt.subplots(figsize=(10, 10))

            # Відображення мапи з оригінальною кастомною палітрою
            ax.imshow(slice_data, cmap=original_cmap, interpolation='nearest', vmin=0, vmax=max_grain_id)

            if ENABLE_GRAIN_LABELS:
                unique_ids = np.unique(slice_data)
                image_shape = slice_data.shape
                layer_area = image_shape[0] * image_shape[1]

                for grain_id in unique_ids:
                    if grain_id == 0:
                        continue

                    grain_mask = (slice_data == grain_id)
                    labeled = label(grain_mask, connectivity=2)

                    props = regionprops(labeled)

                    for prop in props:
                        local_id = prop.label

                        if prop.area <= 20:
                            continue

                        centroid = prop.centroid

                        min_row, min_col, max_row, max_col = prop.bbox
                        is_on_edge = (
                                min_row == 0 or min_col == 0 or
                                max_row == image_shape[0] or max_col == image_shape[1]
                        )

                        if is_on_edge:
                            # Якщо зерно на краю, помічаємо його і пропускаємо розрахунки
                            label_text = f"ø"
                            ax.text(centroid[1], centroid[0], label_text, color='black',
                                    fontsize=FONT_SIZE, ha='center', va='center', weight='bold',
                                    bbox=dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none'))
                            continue  # Переходимо до наступного зерна

                        # Якщо зерно повне (не на краю), розраховуємо його властивості
                        label_text = ""
                        for feature in selected_features:
                            val = get_property_value(prop, feature, layer_area)

                            # Беремо коротке позначення зі словника (або скорочуємо автоматично, якщо його там немає)
                            short_name = FEATURE_SHORT_NAMES.get(feature,
                                                                 feature.split('/')[-1] if '/' in feature else feature)

                            if val is not None:
                                label_text += f"\n{short_name}: {val:.4f}"
                            else:
                                label_text += f"\n{short_name}: N/A"

                        # Накладання тексту на зображення
                        # Автоматично змінюємо фон підпису, щоб чорний текст було видно
                        bg_color = 'white' if FONT_COLOR.lower() == 'black' else 'black'

                        ax.text(centroid[1], centroid[0], label_text, color=FONT_COLOR,
                                fontsize=FONT_SIZE, ha='center', va='center',
                                bbox=dict(facecolor=bg_color, alpha=0.6, pad=0.1, edgecolor='none'))

            ax.set_title(f"Slice {axis.upper()} = {i}")
            ax.set_xlabel(f"{xlabel} axis")
            ax.set_ylabel(f"{ylabel} axis")

            for fmt in FORMATS_2D:
                filename = f"slice_{axis}_{i:04d}.{fmt}"
                plt.savefig(os.path.join(output_dir, filename), dpi=DPI_SETTING, bbox_inches='tight')

            plt.close()
            count += 1


def process_all_csv_in_folder(root_folder):
    print(f"Starting folder traversal: {root_folder}")
    for dirpath, dirnames, filenames in os.walk(root_folder):
        csv_files = [f for f in filenames if f.lower().endswith(".csv")]

        for csv_file in csv_files:
            csv_path = os.path.join(dirpath, csv_file)
            print(f"\nFound CSV: {csv_path}")

            volume = load_volume(csv_path)
            if volume is None: continue

            csv_name = os.path.splitext(csv_file)[0]
            output_dir = os.path.join(dirpath, f"{csv_name}_slices")

            if SAVE_INDIVIDUAL_2D_SLICES:
                visualize_slices(
                    volume,
                    mode=ANALYSIS_MODE,
                    sample_skip=SAMPLE_SKIP,
                    sample_offset=SAMPLE_OFFSET,
                    single_index=SINGLE_LAYER_INDEX,
                    selected_axes=SELECTED_AXES,
                    output_dir=output_dir
                )


if __name__ == "__main__":
    FOLDER = r'D:\University\MatViz\1FULL_MINIMIZE_SCRIPTS\scripts_minimization\test'
    process_all_csv_in_folder(FOLDER)
