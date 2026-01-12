import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

INPUT_FILE = r"D:\Project(MatViz3D)\Random\Paper_Optimisation\test\Radial_2.csv"
OUTPUT_DIR = r"./visualized_layers"

# TODO: Опції аналізу шарів (як у min.py)
# mode: 'all' - аналіз усіх шарів
#       'sample' - аналізувати 1 шар, пропустити SAMPLE_SKIP шарів
#       'single' - аналізувати тільки один шар із індексом SINGLE_LAYER_INDEX
ANALYSIS_MODE = 'sample'  # 'all', 'sample', 'single'
SAMPLE_SKIP = 4  # якщо mode == 'sample', крок = SAMPLE_SKIP + 1
SAMPLE_OFFSET = 0  # початковий індекс для sample
SINGLE_LAYER_INDEX = 0  # індекс шару для mode == 'single'

# TODO: Обрати осі для розрізання (як у min.py)
# ['x'], ['y'], ['z'], ['x', 'y'], ['x', 'z'], ['y', 'z'], або ['x', 'y', 'z']
SELECTED_AXES = ['x']  # Можна вказати кілька осей, наприклад ['x', 'y']


def load_volume(file_path):
    print(f"Зчитування файлу: {file_path}...")
    try:
        data = np.genfromtxt(file_path, delimiter=';', skip_header=1, dtype=int)
    except Exception as e:
        print(f"Помилка відкриття файлу: {e}")
        return None

    if data.size == 0:
        print("Файл порожній.")
        return None

    max_coords = np.max(data[:, :3], axis=0)
    cube_size_x, cube_size_y, cube_size_z = max_coords + 1

    print(f"Виявлено розміри куба: {cube_size_x}x{cube_size_y}x{cube_size_z}")

    volume = np.zeros((cube_size_x, cube_size_y, cube_size_z), dtype=int)
    volume[data[:, 0], data[:, 1], data[:, 2]] = data[:, 3]

    return volume


def create_random_colormap(n_colors):
    hues = np.random.rand(n_colors)
    sats = np.random.uniform(0.6, 1.0, n_colors)
    vals = np.random.uniform(0.85, 1.0, n_colors)

    hsv_colors = np.column_stack((hues, sats, vals))
    rgb_colors = mcolors.hsv_to_rgb(hsv_colors)

    np.random.shuffle(rgb_colors)
    return ListedColormap(rgb_colors)


def visualize_slices(volume,
                     mode=ANALYSIS_MODE,
                     sample_skip=SAMPLE_SKIP,
                     sample_offset=SAMPLE_OFFSET,
                     single_index=SINGLE_LAYER_INDEX,
                     selected_axes=SELECTED_AXES,
                     output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    cube_size = volume.shape[0]

    if mode == 'all':
        indices = list(range(cube_size))
    elif mode == 'sample':
        indices = list(range(sample_offset, cube_size, sample_skip + 1))
    elif mode == 'single':
        indices = [single_index]
    else:
        raise ValueError(f"Unknown analysis mode: {mode}")

    max_id = np.max(volume)
    cmap = create_random_colormap(max_id + 1)

    print(f"Запуск візуалізації: Mode={mode}, Axes={selected_axes}, Indices count={len(indices)}")

    count = 0

    for axis in selected_axes:
        if axis == 'x':
            axis_limit = volume.shape[0]
        elif axis == 'y':
            axis_limit = volume.shape[1]
        elif axis == 'z':
            axis_limit = volume.shape[2]
        else:
            continue

        for i in indices:
            if i >= axis_limit:
                continue

            # Нарізання шару
            if axis == 'x':
                layer = volume[i, :, :]
                xlabel, ylabel = 'Y', 'Z'
            elif axis == 'y':
                layer = volume[:, i, :]
                xlabel, ylabel = 'X', 'Z'
            elif axis == 'z':
                layer = volume[:, :, i]
                xlabel, ylabel = 'X', 'Y'

            plt.figure(figsize=(8, 8))
            plt.imshow(layer, cmap=cmap, interpolation='nearest', origin='upper')

            plt.title(f"Axis: {axis.upper()}, Layer: {i}")
            plt.xlabel(f"{xlabel} axes")
            plt.ylabel(f"{ylabel} axes")
            plt.tight_layout()

            filename = f"slice_{axis}_{i:04d}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=150)
            plt.close()

            count += 1

    print(f"Готово! Збережено {count} зображень у папку '{output_dir}'.")


if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"УВАГА: Файл {INPUT_FILE} не знайдено.")
    else:
        vol = load_volume(INPUT_FILE)
        if vol is not None:
            visualize_slices(
                vol,
                mode=ANALYSIS_MODE,
                sample_skip=SAMPLE_SKIP,
                sample_offset=SAMPLE_OFFSET,
                single_index=SINGLE_LAYER_INDEX,
                selected_axes=SELECTED_AXES,
                output_dir=OUTPUT_DIR
            )