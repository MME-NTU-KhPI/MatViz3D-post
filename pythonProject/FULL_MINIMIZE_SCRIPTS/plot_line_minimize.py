import os

import matplotlib.pyplot as plt
import pandas as pd

PARAMETERS = [
    # "wave_coefficient",
    # "concentration",
    # "halfaxis_a",
    # "halfaxis_b",
    "halfaxis_c",
    # "ellipse_order",
]

TARGETS = [
    "Inertia Tensor/Area XX_Mean",
    "Inertia Tensor/Area XY_Mean",
    "Inertia Tensor/Area YY_Mean",
]

MEAN_VALUES = {
    "Inertia Tensor/Area XX_Mean": 0.1311,
    "Inertia Tensor/Area XY_Mean": -0.0007,
    "Inertia Tensor/Area YY_Mean": 0.0553,
}


def normalize_column(df, col):
    min_val, max_val = df[col].min(), df[col].max()
    if min_val != max_val:
        if min_val >= 0 and max_val <= 1:
            return df[col]
        return (df[col] - min_val) / (max_val - min_val)
    return 0.5


def normalize_x0(x0, df):
    normalized_x0 = []
    for idx, (col, val) in enumerate(zip(df.columns, x0)):
        min_val, max_val = df[col].min(), df[col].max()
        if min_val != max_val:
            if min_val >= 0 and max_val <= 1:
                normalized_val = val
            else:
                normalized_val = (val - min_val) / (max_val - min_val)
        else:
            normalized_val = 0.5

        # Переконуємося, що значення знаходиться в межах [0, 1]
        normalized_val = max(0, min(1, normalized_val))

        normalized_x0.append(normalized_val)

    for i in range(1, len(normalized_x0)):
        if x0[i] == x0[i - 1]:
            normalized_x0[i] = normalized_x0[i - 1]

    return normalized_x0


def plot_optimization(csv_path, output_dir, x0=None):
    file_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_filename = os.path.join(output_dir, f"plot_line({file_name}).png")

    df = pd.read_csv(csv_path)
    columns_to_exclude = ["fun", "mean_ecr", "relative_error(%)"]
    df = df[[col for col in df.columns if col not in columns_to_exclude]]

    df_normalized = df.copy()
    for col in df.columns:
        df_normalized[col] = normalize_column(df, col)

    plt.rcParams.update({
        'font.size': 24,  # базовий розмір шрифту
        'axes.titlesize': 26,  # заголовок графіка
        'axes.labelsize': 24,  # підписи осей
        'xtick.labelsize': 22,  # мітки по X
        'ytick.labelsize': 22,  # мітки по Y
        'legend.fontsize': 20,  # легенда
    })

    plt.figure(figsize=(13, 8))

    for column in df_normalized.columns:
        plt.plot(df.index + 1, df_normalized[column], label=column)

    if x0 is not None:
        normalized_x0 = normalize_x0(x0, df)
        last_iteration = df.index.max() + 1

        shift_x0 = []
        seen_values = {}
        for idx, val in enumerate(normalized_x0):
            if val in seen_values:
                shift_x0.append(last_iteration + 0.5 * (seen_values[val] + 1))
                seen_values[val] += 1
            else:
                shift_x0.append(last_iteration)
                seen_values[val] = 0

        # Пізніше використовуємо ці зміщені значення для відображення точок на графіку
        for idx, (col, val) in enumerate(zip(df.columns, shift_x0)):
            plt.scatter(val, normalized_x0[idx], marker='o', s=50, label=f"{col} (x0)")

    plt.xlabel('Iteration')
    plt.ylabel('Normalized Parameter Value')
    plt.title('Parameter Changes During Optimization')
    plt.legend(loc='upper right', markerscale=0.3)
    plt.grid(True)
    plt.tight_layout()

    plt.ylim(0, 1.05)
    plt.xlim(df.index.min() + 1, df.index.max() + 5)

    plt.savefig(output_filename)
    plt.show()

    print(f"Plot saved at: {output_filename}")
    plt.close()


def plot_inertia_dependencies(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
    })

    for target in TARGETS:
        plt.figure(figsize=(12, 7))

        for param in PARAMETERS:
            # Сортуємо для коректної лінії
            df_sorted = df.sort_values(param)

            plt.plot(
                df_sorted[param],
                df_sorted[target],
                marker='o',
                linewidth=2,
                label=param
            )

        mean_value = MEAN_VALUES[target]
        plt.axhline(
            y=mean_value,
            linestyle="--",
            linewidth=2,
            color="black",
            label=f"Mean = {mean_value:.4f}"
        )

        plt.xlabel("Parameter value")
        plt.ylabel(target)
        plt.title(f"{target} dependence on parameters")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(
            output_dir,
            f"{target.replace('/', '_').replace(' ', '_')}.png"
        )
        plt.savefig(output_path)
        plt.show()
        plt.close()

        print(f"Saved: {output_path}")


def main():
    csv_path = r"D:\Project(MatViz3D)\Random\Paper_Optimisation\test\iter\halfaxis_c\run_parameters_and_selected_features.csv"
    output_dir = r"D:\Project(MatViz3D)\Random\Paper_Optimisation\test\iter\halfaxis_c\plot_line_minimize"
    x0 = [0.1, 1, 1.5, 1.5, 1.5, 90, 90, 90]
    # plot_optimization(csv_path, output_dir, x0=None)
    plot_inertia_dependencies(csv_path, output_dir)


if __name__ == "__main__":
    main()
