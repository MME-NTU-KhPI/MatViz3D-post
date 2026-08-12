import os
import subprocess
import sys

# --- НАЛАШТУВАННЯ ---
ORIGINAL_SCRIPT = "min.py"  # Назва вашого основного файлу
BASE_RESULT_DIR = r"D:\Project(MatViz3D)\Random\Paper_Optimisation\Full_Sweep_Results"

# Список всіх параметрів для перебору
PARAMS_TO_SWEEP = [
    "concentration", "halfaxis_a", "halfaxis_b", "halfaxis_c",
    "ellipse_order", "wave_coefficient", "wave_spread", "initial_nuclei_count",
    "orientation_angle_a", "orientation_angle_b", "orientation_angle_c",
    "stefan_number"
]


def run_sweep_for_param(param_name):
    print(f"\n" + "=" * 50)
    print(f"ПОЧАТОК СВІПУ ДЛЯ ПАРАМЕТРА: {param_name}")
    print("=" * 50)

    # 1. Читаємо оригінальний код
    with open(ORIGINAL_SCRIPT, 'r', encoding='utf-8') as f:
        code = f.read()

    # 2. Динамічно замінюємо налаштування в тексті коду
    # Замінюємо метод на Manual Sweep
    code = code.replace("selected_method = 'Differential Evolution'", "selected_method = 'Manual Sweep'")

    # Замінюємо метрику на Energy Distance (про всяк випадок)
    code = code.replace("selected_metric_type = 'MSE'", "selected_metric_type = 'Energy Distance'")

    # Замінюємо цільовий параметр всередині функції
    code = code.replace('sweep_param = "halfaxis_a"', f'sweep_param = "{param_name}"')

    # Замінюємо шлях виводу на унікальну папку для параметра
    target_folder = os.path.join(BASE_RESULT_DIR, param_name).replace("\\", "\\\\")
    # Шукаємо рядок з output_folder і замінюємо його повністю
    import re
    code = re.sub(r'output_folder = r".*?"', f'output_folder = r"{target_folder}"', code)

    # 3. Створюємо тимчасовий файл для запуску
    temp_script = f"temp_run_{param_name}.py"
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(code)

    # 4. Запускаємо тимчасовий файл
    try:
        subprocess.run([sys.executable, temp_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Помилка при виконанні свіпу для {param_name}: {e}")
    finally:
        # 5. Видаляємо тимчасовий файл
        if os.path.exists(temp_script):
            os.remove(temp_script)


if __name__ == "__main__":
    if not os.path.exists(BASE_RESULT_DIR):
        os.makedirs(BASE_RESULT_DIR)

    for param in PARAMS_TO_SWEEP:
        run_sweep_for_param(param)

    print("\nЗАВЕРШЕНО: Всі параметри оброблені.")