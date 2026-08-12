import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import proj3d

# --- НАЛАШТУВАННЯ ЕКСПОРТУ ---
DPI_SETTING = 600
CAMERA_ELEV = 30  # Ізометричний кут (висота)
CAMERA_AZIM = 45  # Ізометричний кут (азимут)


def load_volume(csv_path):
    """Завантажує 3D об'єм з CSV файлу (формат X;Y;Z;Colors)."""
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
    """Створює кастомну палітру кольорів."""
    hues = np.random.rand(n_colors)
    sats = np.random.uniform(0.6, 1.0, n_colors)
    vals = np.random.uniform(0.85, 1.0, n_colors)

    hsv_colors = np.column_stack((hues, sats, vals))
    rgb_colors = mcolors.hsv_to_rgb(hsv_colors)

    np.random.shuffle(rgb_colors)
    rgb_colors[0] = [0, 0, 0]  # Фон чорний (хоча ми його не будемо малювати)
    return ListedColormap(rgb_colors)


def draw_colored_axes(ax, volume):
    """Малює кольорові вісі X (червона), Y (зелена), Z (синя)."""
    # Визначаємо масштаб осей на основі розміру куба
    max_size = max(volume.shape)
    axis_length = max_size * 0.4

    # Координати початку осей (трохи від краю куба)
    origin = np.array([0, 0, 0])

    # Напрямки осей
    x_axis = np.array([axis_length, 0, 0])
    y_axis = np.array([0, axis_length, 0])
    z_axis = np.array([0, 0, axis_length])

    # Малюємо осі як стрілки
    # X-вісь (червона)
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0], x_axis[1], x_axis[2],
              color='red', arrow_length_ratio=0.15, linewidth=2.5)

    # Y-вісь (зелена)
    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0], y_axis[1], y_axis[2],
              color='green', arrow_length_ratio=0.15, linewidth=2.5)

    # Z-вісь (синя)
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color='blue', arrow_length_ratio=0.15, linewidth=2.5)

    # Додаємо текстові підписи
    ax.text(x_axis[0], x_axis[1], x_axis[2], 'X', color='red', fontsize=12, fontweight='bold')
    ax.text(y_axis[0], y_axis[1], y_axis[2], 'Y', color='green', fontsize=12, fontweight='bold')
    ax.text(z_axis[0], z_axis[1], z_axis[2], 'Z', color='blue', fontsize=12, fontweight='bold')


def render_pure_3d_cube(volume, output_path):
    """Будує та зберігає 3D-кубик з кольоровими осями."""
    print("Генерація 3D-моделі (це може зайняти деякий час для великих масивів)...")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    max_grain_id = int(np.max(volume)) if np.max(volume) > 0 else 1
    cmap = create_old_colormap(max_grain_id + 1)

    # Визначаємо, де є зерна (виключаємо нульовий фон)
    filled = volume > 0

    # Створюємо масив кольорів для кожного вокселя
    # Нормалізуємо значення ідентифікаторів зерен під палітру
    norm = plt.Normalize(vmin=0, vmax=max_grain_id)
    colors = cmap(norm(volume))

    # Малюємо вокселі. edgecolors=None прибирає лінії між вокселями для монолітності
    ax.voxels(filled, facecolors=colors, edgecolors=None)

    # Встановлюємо ракурс камери (оптимізований для ізометричної проекції)
    ax.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM)

    # Малюємо кольорові осі
    draw_colored_axes(ax, volume)

    # Налаштування осей
    ax.set_xlabel('X', color='red', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y', color='green', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z', color='blue', fontsize=10, fontweight='bold')

    # Приховуємо основні осі, але залишаємо сітку мінімальною
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)

    # Встановлюємо однакові межі для всіх осей (кубічна перспектива)
    max_range = max(volume.shape)
    ax.set_xlim(0, max_range)
    ax.set_ylim(0, max_range)
    ax.set_zlim(0, max_range)

    # Зберігаємо файл із прозорим фоном
    plt.savefig(output_path, dpi=DPI_SETTING, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    print(f"Збережено 3D-куб з кольоровими осями: {output_path}")


def process_all_csv_in_folder(root_folder):
    """Проходить по папках і рендерить 3D для кожного CSV."""
    for dirpath, dirnames, filenames in os.walk(root_folder):
        csv_files = [f for f in filenames if f.lower().endswith(".csv")]

        for csv_file in csv_files:
            csv_path = os.path.join(dirpath, csv_file)
            volume = load_volume(csv_path)

            if volume is None: continue

            csv_name = os.path.splitext(csv_file)[0]
            output_file = os.path.join(dirpath, f"{csv_name}_pure_3d.png")

            render_pure_3d_cube(volume, output_file)


if __name__ == "__main__":
    FOLDER = r'D:\Project(MatViz3D)\Random\Paper_Optimisation\cube'
    process_all_csv_in_folder(FOLDER)