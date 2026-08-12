import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ================= CONFIGURATION =================
SHOW_BASELINE = False  # Показ базової лінії
NORMALIZE_X = True  # НОРМАЛІЗАЦІЯ: відображати X від 0 до 1 (0-100% діапазону)
MARKER_SIZE = 4.0  # Розмір точок
LINE_WIDTH = 1.4  # Товщина ліній
DPI = 300
SAVE_INDIVIDUAL = True
A4_SIZE = (8.27, 11.69)

# Логічне групування параметрів
PARAM_GROUPS = {
    "Size": ["halfaxis_a", "halfaxis_b", "halfaxis_c"],
    "Orientation": ["orientation_angle_a", "orientation_angle_b", "orientation_angle_c"],
    "Shape": ["ellipse_order", "wave_coefficient", "wave_spread"],
    "Model": ["concentration", "stefan_number", "initial_nuclei_count"]
}

STATISTICS = ["Std", "Median", "Q1", "Q3"]
COMPONENTS = ["XX", "XY", "YY"]

# Baseline (використовується, якщо SHOW_BASELINE = True)
REFERENCE_VALUES = {
    "Inertia Tensor/Area XX_Std": 0.0147, "Inertia Tensor/Area XY_Std": 0.0157, "Inertia Tensor/Area YY_Std": 0.0151,
    "Inertia Tensor/Area XX_Median": 0.0829, "Inertia Tensor/Area XY_Median": -0.0012,
    "Inertia Tensor/Area YY_Median": 0.0827,
    "Inertia Tensor/Area XX_Q1": 0.0747, "Inertia Tensor/Area XY_Q1": -0.0093, "Inertia Tensor/Area YY_Q1": 0.0759,
    "Inertia Tensor/Area XX_Q3": 0.0952, "Inertia Tensor/Area XY_Q3": 0.0083, "Inertia Tensor/Area YY_Q3": 0.0923,
}


def load_sweep_data(root_dir):
    data_frames = {}
    for group_name, params in PARAM_GROUPS.items():
        for param_folder in params:
            folder_path = os.path.join(root_dir, param_folder)
            csv_path = os.path.join(folder_path, "run_parameters_and_selected_features.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                data_frames[param_folder] = df
            else:
                print(f"Warning: File not found -> {csv_path}")
    return data_frames


def plot_comprehensive(data_frames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    indiv_dir = os.path.join(output_dir, "Individual_Plots")
    if SAVE_INDIVIDUAL: os.makedirs(indiv_dir, exist_ok=True)

    # Налаштування стилю для наукової публікації
    plt.rcParams.update({
        'font.size': 8.5,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'font.family': 'serif',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.titlepad': 4  # Додатковий відступ для заголовків сабплотів
    })

    for comp in COMPONENTS:
        n_rows = len(PARAM_GROUPS) * 2
        fig, axs = plt.subplots(n_rows, 2, figsize=A4_SIZE)

        # Збільшено hspace до 0.65, щоб заголовки не торкалися осей графіків вище
        plt.subplots_adjust(hspace=0.45, wspace=0.25, top=0.96, bottom=0.06, left=0.1, right=0.95)

        for g_idx, (group_name, params) in enumerate(PARAM_GROUPS.items()):
            base_row = g_idx * 2

            for s_idx, stat in enumerate(STATISTICS):
                row = base_row + (s_idx // 2)
                col = s_idx % 2
                ax = axs[row, col]
                target_col = f"Inertia Tensor/Area {comp}_{stat}"

                for p in params:
                    if p in data_frames:
                        df = data_frames[p]
                        x_col = p
                        if p.startswith("orientation_angle_") and p.replace("orientation_", "") in df.columns:
                            x_col = p.replace("orientation_", "")

                        if x_col in df.columns:
                            df_s = df.sort_values(x_col).copy()

                            x_min, x_max = df_s[x_col].min(), df_s[x_col].max()
                            if NORMALIZE_X and (x_max != x_min):
                                x_data = (df_s[x_col] - x_min) / (x_max - x_min)
                                x_label = f"{x_min:.1g} \u2192 {x_max:.1g}"
                            else:
                                x_data = df_s[x_col]
                                x_label = f"[{x_min:.1g}, {x_max:.1g}]"

                            clean_name = p.replace("halfaxis_", "").replace("orientation_angle_", "").replace("wave_",
                                                                                                              "")
                            label = f"{clean_name} {x_label}"

                            ax.plot(x_data, df_s[target_col], marker='o',
                                    markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=label)

                if SHOW_BASELINE:
                    base_val = REFERENCE_VALUES.get(target_col)
                    if base_val is not None:
                        ax.axhline(y=base_val, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')

                ax.set_title(f"{group_name} | {stat}", fontweight='bold')

                # ВИПРАВЛЕННЯ: Підпис осі X тільки для останнього ряду на сторінці
                if row == n_rows - 1:
                    if NORMALIZE_X:
                        ax.set_xlabel("Relative Sweep [0-1]")
                    else:
                        ax.set_xlabel("Parameter Value")
                else:
                    # Прибираємо текст підпису для всіх інших рядів
                    ax.set_xlabel("")

                if col == 0:
                    ax.set_ylabel(f"{comp} Value")

                if NORMALIZE_X:
                    ax.set_xlim(-0.05, 1.05)

                ax.legend(loc='best', frameon=True, framealpha=0.8, fontsize=6)

        output_path = os.path.join(output_dir, f"Summary_{comp}_Final.png")
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"Збережено: {output_path}")


def main():
    # Шлях до кореневої папки з результатами
    root_dir = r"D:\Project(MatViz3D)\Random\Paper_Optimisation\Full_Sweep_Results"
    output_dir = os.path.join(root_dir, "Aggregated_Plots_Normalized")

    data = load_sweep_data(root_dir)
    if data:
        print(f"Завантажено даних для {len(data)} параметрів. Починаємо малювання...")
        plot_comprehensive(data, output_dir)
        print("Готово! Графіки збережені в папку Aggregated_Plots_Final.")


if __name__ == "__main__":
    main()