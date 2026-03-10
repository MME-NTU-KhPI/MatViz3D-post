import sys
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from MatViz3DLauncher import MatViz3DLauncher

try:
    import min as core
except ImportError:
    print("Помилка: Не знайдено файл min.py. Переконайтеся, що він у тій же папці.")
    sys.exit(1)


class ManualOptimizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MatViz3D Manual Optimizer")
        self.root.geometry("1400x900")

        self.exe_path = core.exe_path
        self.launcher = MatViz3DLauncher(self.exe_path)
        self.output_folder = core.output_folder

        self.run_counter = 0
        self.setup_ui()

    def setup_ui(self):
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # --- Ліва панель ---
        self.sidebar = ttk.Frame(self.paned, width=400, padding=10)
        self.paned.add(self.sidebar)

        ttk.Label(self.sidebar, text="Параметри моделювання", font=('Arial', 11, 'bold')).pack(pady=5)

        self.param_entries = {}
        params_list = [
            ("concentration", core.base_params[0]),
            ("halfaxis_b", core.base_params[1]),
            ("halfaxis_c", core.base_params[2]),
            ("ellipse_order", core.base_params[3]),
            ("wave_coefficient", core.base_params[4]),
            ("wave_spread", core.base_params[5]),
            ("initial_nuclei_count", core.base_params[6])
        ]

        for label, val in params_list:
            frame = ttk.Frame(self.sidebar)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label, width=22).pack(side=tk.LEFT)
            entry = ttk.Entry(frame)
            entry.insert(0, str(val))
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            self.param_entries[label] = entry

        ttk.Label(self.sidebar, text="Метод метрики:").pack(pady=(15, 0))
        self.metric_combo = ttk.Combobox(self.sidebar, values=['Energy Distance', 'MSE', 'SMAPE', 'MSPE'])
        self.metric_combo.set(core.selected_metric_type)
        self.metric_combo.pack(fill=tk.X, pady=5)

        self.run_btn = ttk.Button(self.sidebar, text="ЗАПУСТИТИ РОЗРАХУНОК", command=self.run_calculation)
        self.run_btn.pack(fill=tk.X, pady=10)

        ttk.Label(self.sidebar, text="Журнал порівняння:").pack(anchor=tk.W)
        self.result_text = tk.Text(self.sidebar, height=30, width=50, font=('Consolas', 9))
        self.result_text.pack(fill=tk.BOTH, expand=True)

        ttk.Button(self.sidebar, text="Очистити історію", command=lambda: self.result_text.delete(1.0, tk.END)).pack(
            fill=tk.X)

        # --- Права панель ---
        self.plot_frame = ttk.Frame(self.paned, padding=10)
        self.paned.add(self.plot_frame)

        self.fig = plt.figure(figsize=(8, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

    def run_calculation(self):
        try:
            self.run_counter += 1
            p_vals = [float(self.param_entries[name].get()) for name in [
                "concentration", "halfaxis_b", "halfaxis_c", "ellipse_order",
                "wave_coefficient", "wave_spread", "initial_nuclei_count"
            ]]
            current_metric = self.metric_combo.get()

            # Визначаємо шлях для РОЗРАХУНКУ похибки
            if current_metric == 'Energy Distance':
                target_path = core.TARGET_FILE_DIST_LOG if core.USE_LOG_SPACE else core.TARGET_FILE_DIST
            else:
                target_path = core.TARGET_FILE_LOG if core.USE_LOG_SPACE else core.TARGET_FILE_NORMAL

            # ЗАВЖДИ визначаємо шлях до розподілу для ГІСТОГРАМИ
            dist_path = core.TARGET_FILE_DIST_LOG if core.USE_LOG_SPACE else core.TARGET_FILE_DIST

            self.run_btn.config(state=tk.DISABLED, text="Обробка...")
            self.root.update()

            output_file = r".\gui_cube_output.csv"
            output_props_file = r".\gui_properties_output.csv"

            halfaxis_a = p_vals[1]
            rounded_nuclei = int(round(p_vals[6]))

            generated = core.start(core.FIXED_SIZE, p_vals[0], halfaxis_a, p_vals[1], p_vals[2],
                                   0, 0, 0, p_vals[4], p_vals[5], rounded_nuclei, p_vals[3], output_file)

            if not generated:
                raise Exception("MatViz3D не зміг створити файл.")

            stats, sim_df = core.process_and_calculate(generated, output_props_file)

            # Завантажуємо дані для розрахунку та окремо для малювання гістограми
            target_data = core.load_target_data(target_path, current_metric, core.selected_features)
            dist_data = core.load_target_data(dist_path, 'Energy Distance', core.selected_features)

            self.display_results(sim_df, stats, target_data, dist_data, current_metric, p_vals, rounded_nuclei)

        except Exception as e:
            messagebox.showerror("Помилка", f"Стався збій: {str(e)}")
        finally:
            self.run_btn.config(state=tk.NORMAL, text="ЗАПУСТИТИ РОЗРАХУНОК")

    def display_results(self, sim_df, sim_stats, target_data, dist_data, metric_type, params, rounded_nuclei):
        self.fig.clear()
        valid_features = [f for f in core.selected_features if f in sim_df.columns]
        n_plots = len(valid_features)

        report = f"\n>>> ЗАПУСК №{self.run_counter}\n"
        report += (f"Параметри: conc={params[0]}, h_a={params[1]}, h_b={params[1]}, h_c={params[2]}, "
                   f"order={params[3]}, wc={params[4]}, ws={params[5]}, nuclei={rounded_nuclei}\n")
        report += f"Метрика: {metric_type}\n"

        # 1. Розрахунок похибок (ідентично min.py)
        if metric_type == 'Energy Distance':
            report += "--- Індивідуальні Energy Distance ---\n"
            for feat in valid_features:
                sim_feat = sim_df[feat].values
                exp_feat = dist_data[feat].values
                exp_feat_mean, exp_feat_std = np.mean(exp_feat), (np.std(exp_feat) if np.std(exp_feat) != 0 else 1.0)
                ed_feat_val = core.calculate_energy_distance((sim_feat - exp_feat_mean) / exp_feat_std,
                                                             (exp_feat - exp_feat_mean) / exp_feat_std)
                report += f"  {feat}: {ed_feat_val:.6f}\n"

            sim_values, exp_values = sim_df[valid_features].values, dist_data[valid_features].values
            exp_mean, exp_std = np.mean(exp_values, axis=0), np.std(exp_values, axis=0)
            exp_std[exp_std == 0] = 1.0
            total_error = core.calculate_energy_distance((sim_values - exp_mean) / exp_std,
                                                         (exp_values - exp_mean) / exp_std)
            report += f">> Total Multivariate ED: {total_error:.6f}\n"
        else:
            err_count, total_error = 0, 0
            for feat in valid_features:
                if feat not in target_data: continue
                report += f"--- {feat} ---\n"
                for m in core.selected_metrics:
                    if m in sim_stats and m in target_data[feat]:
                        actual, predicted = sim_stats[m][feat], target_data[feat][m]
                        if metric_type == 'SMAPE':
                            err = core.calculate_smape([actual], [predicted])
                        elif metric_type == 'MSE':
                            t_std = target_data[feat].get('Std', 1.0)
                            err = core.calculate_mse([actual / (t_std if t_std != 0 else 1.0)],
                                                     [predicted / (t_std if t_std != 0 else 1.0)])
                        elif metric_type == 'MSPE':
                            err = core.calculate_mspe([actual], [predicted])
                        else:
                            err = abs(actual - predicted)
                        total_error += err
                        err_count += 1
                        report += f"  {m}: {err:.4f}\n"
            if err_count > 0: total_error /= err_count
            report += f">> Загальна похибка ({metric_type}): {total_error:.4f}\n"

        self.result_text.insert(tk.END, report + "=" * 30 + "\n")
        self.result_text.see(tk.END)

        # 2. Побудова графіків (завжди з гістограмою розподілу цілі)
        for i, feat in enumerate(valid_features):
            ax = self.fig.add_subplot(n_plots, 1, i + 1)

            # Гістограми (Тепер таргет завжди на фоні)
            if feat in dist_data.columns:
                ax.hist(dist_data[feat], bins=40, alpha=0.3, color='red', label='Target Dist', density=True)
            ax.hist(sim_df[feat], bins=40, alpha=0.5, color='teal', label='Sim', density=True)

            # Лінії статистики для симуляції
            ax.axvline(sim_stats['Mean'][feat], color='yellow', linestyle='-', linewidth=2, label='Sim Mean')
            ax.axvline(sim_stats['Q1'][feat], color='orange', linestyle='--', linewidth=1.5, label='Sim Q1')
            ax.axvline(sim_stats['Q3'][feat], color='purple', linestyle='--', linewidth=1.5, label='Sim Q3')

            # Лінії статистики для таргету (з повного розподілу)
            t_vals = dist_data[feat].values
            t_mean = np.mean(t_vals)
            t_q1, t_q3 = np.percentile(t_vals, [25, 75])

            ax.axvline(t_mean, color='red', linestyle='-', linewidth=2, label='Targ Mean')
            ax.axvline(t_q1, color='red', linestyle=':', linewidth=1.5, label='Targ Q1')
            ax.axvline(t_q3, color='red', linestyle='-.', linewidth=1.5, label='Targ Q3')

            ax.set_title(f"Feature: {feat}")
            ax.legend(prop={'size': 7}, loc='upper right', bbox_to_anchor=(1.15, 1))

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ManualOptimizerApp(root)
    root.mainloop()