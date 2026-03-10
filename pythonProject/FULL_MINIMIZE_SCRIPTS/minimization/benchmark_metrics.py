import os
import sys
import time

import numpy as np
import pandas as pd

try:
    from min import (
        start, process_and_calculate, load_target_data,
        calculate_mse, calculate_smape, calculate_mspe, calculate_energy_distance,
        selected_features, selected_metrics,
        TARGET_FILE_NORMAL, TARGET_FILE_LOG, TARGET_FILE_DIST, TARGET_FILE_DIST_LOG,
        output_folder, FIXED_SIZE, dual_annealing, bounds
    )
except ImportError:
    print("Помилка: Не знайдено файл min.py або в ньому відсутні потрібні функції/змінні.")
    sys.exit(1)

# --- НАЛАШТУВАННЯ БЕНЧМАРКУ ---
RUN_MODE = 'optimization'  # 'optimization' 'single'
MAX_ITER = 3
TEST_PARAMS = [0.1, 1.5, 1.5, 1.7, 40, 2.0]

RESULT_CSV = os.path.join(output_folder, "benchmark_detailed_results.csv")
RESULT_TXT = os.path.join(output_folder, "benchmark_report.txt")

all_benchmark_data = []
iteration_counter = 0


def display_iteration_table(iter_data, iteration_label, params):
    """Формує структурований звіт для консолі та файлу"""
    df = pd.DataFrame(iter_data)

    # 1. Окрема таблиця для параметрів
    df_params = df[df['Space'] == 'N/A'].copy()
    param_table = df_params.pivot_table(index="Feature", values="Value", sort=False)
    param_table.index.name = "Input Parameter"
    param_table.columns = ["Value"]

    # 2. Основна таблиця метрик
    df_metrics = df[df['Space'] != 'N/A'].copy()

    # Очищуємо назви метрик від префіксів (1_MSE -> MSE)
    df_metrics['Metric'] = df_metrics['Metric'].str.replace(r'^\d_', '', regex=True)

    # Створюємо ієрархічну таблицю: [Metric] -> [Space]
    # Тепер метрика буде головним стовпцем, а під нею Linear/Log
    pivot = df_metrics.pivot_table(
        index="Feature",
        columns=["Metric", "Space"],
        values="Value",
        sort=False
    )

    # Формування тексту
    output = f"\n" + "=" * 120 + "\n"
    output += f"  ІТЕРАЦІЯ № {iteration_label} | ЗВІТ ПОРІВНЯННЯ\n"
    output += "=" * 120 + "\n\n"

    output += "--- ПАРАМЕТРИ ЗАПУСКУ ---\n"
    output += param_table.to_string() + "\n\n"

    output += "--- ТАБЛИЦЯ ПОХИБОК ---\n"
    output += pivot.to_string(float_format=lambda x: f"{x:12.4f}") + "\n"
    output += "=" * 120 + "\n"

    print(output)
    return output


def run_full_benchmark(params, iteration_label="Single"):
    conc, h_b, h_c, order, wave_coefficient, wave_spread = params
    h_a = (h_b + h_c) / 2
    iter_results = []

    # 0. Записуємо параметри
    for name, val in zip(["Conc", "h_b", "h_c", "Order", "Wave_Coeff", "Wave_Spread"], params):
        iter_results.append(
            {"Iter": iteration_label, "Space": "N/A", "Metric": "0_Params", "Feature": name, "Value": val})

    # 1. Запуск симуляції
    temp_cube = os.path.join(output_folder, f"bench_cube_{iteration_label}.csv")
    temp_props = os.path.join(output_folder, f"bench_props_{iteration_label}.csv")

    generated_file = start(size=FIXED_SIZE, concentration=conc, halfaxis_a=h_a, halfaxis_b=h_b, halfaxis_c=h_c,
                           orientation_angle_a=0, orientation_angle_b=0, orientation_angle_c=0,
                           wave_coefficient=wave_coefficient, wave_spread=wave_spread, ellipse_order=order,
                           output_file=temp_cube)

    if not generated_file: return iter_results

    # 2. Обробка (отримуємо статистики stats та сирі дані df_sim)
    stats_sim, df_sim = process_and_calculate(generated_file, temp_props)
    if stats_sim is None: return iter_results

    # 3. Розрахунок у двох просторах
    for space in ["Linear", "Log"]:
        # --- НОВИЙ БЛОК: ПІДГОТОВКА СТАТИСТИКИ ДЛЯ КОНКРЕТНОГО ПРОСТОРУ ---
        if space == "Log":
            # Робимо логарифмування сирих даних ПЕРЕД розрахунком статистик
            df_space = df_sim.copy()
            for col in selected_features:
                if col in df_space.columns:
                    df_space[col] = np.arcsinh(df_space[col])

            # Перераховуємо статистики (Mean, Q1, Q3...) для логарифмічних значень
            current_stats_sim = {
                'Mean': df_space.mean(),
                'Std': df_space.std(),
                'Median': df_space.median(),
                'Q1': df_space.quantile(0.25),
                'Q3': df_space.quantile(0.75),
            }
        else:
            # Для Linear використовуємо вже готові лінійні статистики
            current_stats_sim = stats_sim

        # Тепер завантажуємо відповідні таргет-дані
        if space == "Linear":
            targets_stat = load_target_data(TARGET_FILE_NORMAL, 'SMAPE', selected_features)
            targets_dist = load_target_data(TARGET_FILE_DIST, 'Energy Distance', selected_features)
        else:
            targets_stat = load_target_data(TARGET_FILE_LOG, 'SMAPE', selected_features)
            targets_dist = load_target_data(TARGET_FILE_DIST_LOG, 'Energy Distance', selected_features)

        # --- РОЗРАХУНОК MSE, SMAPE, MSPE ---
        for m_type in ["MSE", "SMAPE", "MSPE"]:
            errors_list = []
            m_func = {"MSE": calculate_mse, "SMAPE": calculate_smape, "MSPE": calculate_mspe}[m_type]
            m_prefix = {"MSE": "1_MSE", "SMAPE": "2_SMAPE", "MSPE": "3_MSPE"}[m_type]

            for feat in selected_features:
                # Використовуємо current_stats_sim (яка тепер відповідає простору!)
                if feat in current_stats_sim['Mean'] and feat in targets_stat:
                    for stat_name in selected_metrics:
                        if stat_name in current_stats_sim and stat_name in targets_stat[feat]:
                            actual = current_stats_sim[stat_name][feat]
                            predicted = targets_stat[feat][stat_name]

                            err = m_func([actual], [predicted])
                            errors_list.append(err)

                            iter_results.append({
                                "Iter": iteration_label, "Space": space,
                                "Metric": m_prefix, "Feature": f"{feat} ({stat_name})", "Value": err
                            })

            if errors_list:
                iter_results.append({
                    "Iter": iteration_label, "Space": space,
                    "Metric": m_prefix, "Feature": "ALL (Average Error)", "Value": np.mean(errors_list)
                })

        # --- РОЗРАХУНОК ENERGY DISTANCE (на сирих розподілах) ---
        valid_cols = [c for c in selected_features if c in df_sim.columns and c in targets_dist.columns]
        if valid_cols:
            sim_data = df_sim[valid_cols].values
            exp_data = targets_dist[valid_cols].values

            if space == "Log":
                sim_data = np.arcsinh(sim_data)

            # Нормалізація (як у min.py)
            e_mean, e_std = np.mean(exp_data, axis=0), np.std(exp_data, axis=0)
            e_std[e_std == 0] = 1.0
            sim_norm, exp_norm = (sim_data - e_mean) / e_std, (exp_data - e_mean) / e_std

            # Multivariate
            ed_all = calculate_energy_distance(sim_norm, exp_norm)
            iter_results.append(
                {"Iter": iteration_label, "Space": space, "Metric": "4_EnergyDist", "Feature": "ALL (Multivariate)",
                 "Value": ed_all})

            # Univariate (по кожній фічі окремо)
            for i, col in enumerate(valid_cols):
                ed_single = calculate_energy_distance(sim_norm[:, i], exp_norm[:, i])
                iter_results.append({"Iter": iteration_label, "Space": space, "Metric": "4_EnergyDist", "Feature": col,
                                     "Value": ed_single})

    # Вивід таблиці в консоль
    table_text = display_iteration_table(iter_results, iteration_label, params)

    # Запис у TXT файл
    with open(RESULT_TXT, "a", encoding="utf-8") as f:
        f.write(table_text + "\n")

    return iter_results


def objective_wrapper(params):
    global iteration_counter, all_benchmark_data
    iteration_counter += 1
    res = run_full_benchmark(params, str(iteration_counter))
    all_benchmark_data.extend(res)

    # Для оптимізатора повертаємо "ALL (Multivariate) Energy Distance" у лінійному просторі
    for r in res:
        if r['Metric'] == '4_EnergyDist' and r['Space'] == 'Linear' and r['Feature'] == 'ALL (Multivariate)':
            return r['Value']
    return 999.0


def main():
    global all_benchmark_data
    # Фіксуємо час початку, як у min.py
    start_time = time.time()

    # Очищуємо файл звіту
    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write(f"FULL COMPARATIVE BENCHMARK LOG\nStarted: {time.ctime()}\n")

    result_obj = None
    if RUN_MODE == 'single':
        all_benchmark_data = run_full_benchmark(TEST_PARAMS, "Single")
    else:
        print(f"Запуск оптимізації на {MAX_ITER} ітерацій...")
        # Зберігаємо об'єкт результату dual_annealing
        result_obj = dual_annealing(
            objective_wrapper,
            bounds=bounds,
            x0=TEST_PARAMS,
            maxiter=MAX_ITER,
            no_local_search=True
        )

    # Розрахунок витраченого часу
    elapsed_time = time.time() - start_time

    # Збереження детального CSV
    pd.DataFrame(all_benchmark_data).to_csv(RESULT_CSV, index=False)

    # --- ВИВІД РЕЗУЛЬТАТІВ ЯК У MIN.PY ---
    if result_obj is not None:
        final_summary = (
                f"\n" + "=" * 50 + "\n"
                                   f"Optimization completed in {elapsed_time:.2f} seconds.\n"
                                   f"Best parameters found: {result_obj.x}\n"
                                   f"Final Energy Distance error: {result_obj.fun:.6f}\n"
                                   f"=" * 50 + "\n"
        )

        # Вивід у консоль
        print(final_summary)
        # Вивід повного об'єкта результату (як у main в min.py)
        print(result_obj)

        # Запис фінального результату у текстовий звіт
        with open(RESULT_TXT, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("         FINAL OPTIMIZATION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(final_summary)
            f.write(f"\nFull Scipy Result Object:\n{str(result_obj)}\n")

    print(f"\nБенчмарк завершено.\nCSV: {RESULT_CSV}\nTXT: {RESULT_TXT}")


if __name__ == "__main__":
    main()
