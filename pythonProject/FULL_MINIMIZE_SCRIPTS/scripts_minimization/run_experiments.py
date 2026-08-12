import os
import re
import sys
import subprocess

# Визначаємо поточну директорію, де лежить цей скрипт та min.py (scripts_minimization)
current_dir = os.path.abspath(os.path.dirname(__file__))

# ==============================================================================
# НАЛАШТУВАННЯ ПАРАМЕТРІВ ДЛЯ ПЕРЕБОРУ
# ==============================================================================
# ТУТ ВКАЗУЙТЕ ВАШІ РЕАЛЬНІ ПАПКИ НА WINDOWS
IMAGE_FOLDERS = ["AZ31_iA", "AZ31_iB"]
IMAGE_SUFFIXES = ["AZ31_imgA", "AZ31_imgB"]

# Якщо хочете запустити всі картинки одразу, закоментуйте рядки вище та розкоментуйте ці:
# IMAGE_FOLDERS = ["AZ31_iA", "AZ31_iB", "GW83", "Mg-Gd-Y-Zr_GW83", "SMPP-LT_imgb", "WE43-0P"]
# IMAGE_SUFFIXES = ["AZ31_imgA", "AZ31_imgB", "GW83", "Mg-Gd-Y-Zr_GW83", "SMPP-LT_imgb", "WE43-0P"]
# (Обов'язково перевірте, чи суфікси точно відповідають тим, що в дужках у файлах .csv)

# Методи оптимізації
METHODS = ["SLSQP", "L-BFGS-B", "Dual Annealing", "basinhopping", "Differential Evolution"]
METRIC = "SMAPE"

# Створюємо шлях до папки Result в поточній директорії
BASE_OUTPUT_DIR = os.path.join(current_dir, "Result")

# Точний шлях до папки з картинками (піднімаємося на 1 рівень і заходимо в Target_img)
TARGET_IMG_DIR = os.path.abspath(os.path.join(current_dir, "..", "Target_img"))

# Шляхи до базового та тимчасового файлів
base_py = os.path.join(current_dir, "min.py")
temp_py = os.path.join(current_dir, "temp_run_min.py")

if not os.path.exists(base_py):
    print(f"Помилка: Файл {base_py} не знайдено!")
    sys.exit(1)

# ==============================================================================
# ГОЛОВНИЙ ЦИКЛ ЗАПУСКУ
# ==============================================================================
img_count = len(IMAGE_FOLDERS)

for i in range(img_count):
    folder_name = IMAGE_FOLDERS[i]
    file_suffix = IMAGE_SUFFIXES[i]

    for method in METHODS:
        method_folder = method.replace(" ", "_")
        out_dir = os.path.join(BASE_OUTPUT_DIR, folder_name, method_folder, METRIC)
        os.makedirs(out_dir, exist_ok=True)

        print("=" * 70)
        print(f"Запуск: Зображення = {folder_name} | Метод = {method} | Метрика = {METRIC}")
        print(f"Папка виводу: {out_dir}")
        print("=" * 70)

        # 1. Зчитуємо оригінальний min.py
        with open(base_py, "r", encoding="utf-8") as f:
            content = f.read()

        # 2. Формуємо шляхи для цільових файлів
        target_normal = os.path.join(TARGET_IMG_DIR, folder_name, "processed_output",
                                     f"statistics_image_properties_({file_suffix}).csv")
        target_log = os.path.join(TARGET_IMG_DIR, folder_name, "processed_output",
                                  f"Arcsinh_statistics_image_properties_({file_suffix}).csv")
        target_dist = os.path.join(TARGET_IMG_DIR, folder_name, "processed_output",
                                   f"processed_image_properties_({file_suffix}).csv")
        target_dist_log = os.path.join(TARGET_IMG_DIR, folder_name, "processed_output",
                                       f"processed_Arcsinh_image_properties_({file_suffix}).csv")

        # ПЕРЕВІРКА: Чи існує взагалі цей файл на диску?
        if not os.path.exists(target_normal):
            print(f" ❌ УВАГА: Файл не знайдено -> {target_normal}")
            print(f"    Пропускаємо цей крок. Перевірте назву папки або наявність файлу!\n")
            continue

        escaped_out_dir = out_dir.replace("\\", "\\\\")
        escaped_normal = target_normal.replace("\\", "\\\\")
        escaped_log = target_log.replace("\\", "\\\\")
        escaped_dist = target_dist.replace("\\", "\\\\")
        escaped_dist_log = target_dist_log.replace("\\", "\\\\")

        # 3. Підміняємо параметри
        content = re.sub(r"^selected_method\s*=\s*.*", f"selected_method = '{method}'", content, flags=re.MULTILINE)
        content = re.sub(r"^selected_metric_type\s*=\s*.*", f"selected_metric_type = '{METRIC}'", content,
                         flags=re.MULTILINE)
        content = re.sub(r"^output_folder\s*=\s*.*", f"output_folder = r'{escaped_out_dir}'", content,
                         flags=re.MULTILINE)

        content = re.sub(r"^TARGET_FILE_NORMAL\s*=\s*.*", f"TARGET_FILE_NORMAL = r'{escaped_normal}'", content,
                         flags=re.MULTILINE)
        content = re.sub(r"^TARGET_FILE_LOG\s*=\s*.*", f"TARGET_FILE_LOG = r'{escaped_log}'", content,
                         flags=re.MULTILINE)
        content = re.sub(r"^TARGET_FILE_DIST\s*=\s*.*", f"TARGET_FILE_DIST = r'{escaped_dist}'", content,
                         flags=re.MULTILINE)
        content = re.sub(r"^TARGET_FILE_DIST_LOG\s*=\s*.*", f"TARGET_FILE_DIST_LOG = r'{escaped_dist_log}'", content,
                         flags=re.MULTILINE)

        # 4. Записуємо зміни
        with open(temp_py, "w", encoding="utf-8") as f:
            f.write(content)

        # 5. Запускаємо інтерпретатор зі всіма шляхами до модулів
        env = os.environ.copy()

        # Знаходимо батьківську папку 1FULL_MINIMIZE_SCRIPTS та папку MV3dLaunch
        parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
        mv3d_launch_dir = os.path.join(parent_dir, "MV3dLaunch")

        paths = [parent_dir, current_dir, mv3d_launch_dir]
        env["PYTHONPATH"] = ";".join(paths) + ((";" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

        try:
            subprocess.run([sys.executable, temp_py], env=env, check=True)
            print(f" ✅ Успішно завершено для: {method}.\n")
        except subprocess.CalledProcessError as e:
            print(f" ❌ Помилка під час обчислень для {method}: {e}\n")
        finally:
            if os.path.exists(temp_py):
                os.remove(temp_py)

print("=" * 70)
print("Вітаємо! Всі розрахунки успішно завершено.")
print("=" * 70)