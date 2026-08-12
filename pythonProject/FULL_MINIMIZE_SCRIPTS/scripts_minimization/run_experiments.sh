#!/bin/bash

# ==============================================================================
# Налаштування параметрів для перебору
# ==============================================================================

# Масиви з назвами папок картинок та відповідними їм суфіксами у назвах файлів .csv
IMAGE_FOLDERS=("AZ31_iA" "AZ31_iB" "WE43-0P")
IMAGE_SUFFIXES=("AZ31_imgA" "AZ31_imgB" "WE43-0P") # Замініть WE43-0P на реальний суфікс, якщо він інший

# Масив з методами оптимізації (як у вашій таблиці)
METHODS=("SLSQP" "L-BFGS-B" "Dual Annealing" "basinhopping" "Differential Evolution")

# Метрика (якщо потрібно перебрати декілька, можна також зробити цикл)
METRIC="SMAPE" 

# ==============================================================================
# Шляхи
# ==============================================================================
BASE_PY="min.py"
TEMP_PY="temp_run_min.py"

# Базова папка для збереження результатів
BASE_OUTPUT_DIR="/home/valeriia.hriskova/MatVizProject/PythonScripts/Result"
TARGET_IMG_DIR="/home/valeriia.hriskova/MatVizProject/PythonScripts/target_img"

# ==============================================================================
# Головний цикл
# ==============================================================================

# Отримуємо довжину масиву зображень
IMG_COUNT=${#IMAGE_FOLDERS[@]}

for (( i=0; i<$IMG_COUNT; i++ )); do
    FOLDER_NAME="${IMAGE_FOLDERS[$i]}"
    FILE_SUFFIX="${IMAGE_SUFFIXES[$i]}"
    
    for METHOD in "${METHODS[@]}"; do
        
        # Формуємо шлях для результатів: Result / Картинка / Метод / Метрика
        OUT_DIR="$BASE_OUTPUT_DIR/$FOLDER_NAME/$(echo $METHOD | tr ' ' '_')/$METRIC"
        
        echo "=========================================================="
        echo "Запуск: Зображення = $FOLDER_NAME | Метод = $METHOD | Метрика = $METRIC"
        echo "Папка виводу: $OUT_DIR"
        echo "=========================================================="
        
        # 1. Створюємо папку, якщо її не існує
        mkdir -p "$OUT_DIR"
        
        # 2. Робимо тимчасову копію оригінального скрипта
        cp "$BASE_PY" "$TEMP_PY"
        
        # 3. Підміняємо змінні у тимчасовому скрипті за допомогою sed
        
        # Підміна методу
        sed -i "s/^selected_method = .*/selected_method = '$METHOD'/g" "$TEMP_PY"
        
        # Підміна метрики
        sed -i "s/^selected_metric_type = .*/selected_metric_type = '$METRIC'/g" "$TEMP_PY"
        
        # Підміна папки для результатів (використовуємо | як роздільник для sed, бо в шляху є /)
        sed -i "s|^output_folder = .*|output_folder = r\"$OUT_DIR\"|g" "$TEMP_PY"
        
        # Формуємо нові шляхи до цільових CSV файлів
        NEW_TARGET_NORMAL="r\"$TARGET_IMG_DIR/$FOLDER_NAME/processed_output/statistics_image_properties_($FILE_SUFFIX).csv\""
        NEW_TARGET_LOG="r\"$TARGET_IMG_DIR/$FOLDER_NAME/processed_output/Arcsinh_statistics_image_properties_($FILE_SUFFIX).csv\""
        NEW_TARGET_DIST="r\"$TARGET_IMG_DIR/$FOLDER_NAME/processed_output/processed_image_properties_($FILE_SUFFIX).csv\""
        NEW_TARGET_DIST_LOG="r\"$TARGET_IMG_DIR/$FOLDER_NAME/processed_output/processed_Arcsinh_image_properties_($FILE_SUFFIX).csv\""
        
        # Підміна шляхів до CSV у скрипті
        sed -i "s|^TARGET_FILE_NORMAL = .*|TARGET_FILE_NORMAL = $NEW_TARGET_NORMAL|g" "$TEMP_PY"
        sed -i "s|^TARGET_FILE_LOG = .*|TARGET_FILE_LOG = $NEW_TARGET_LOG|g" "$TEMP_PY"
        sed -i "s|^TARGET_FILE_DIST = .*|TARGET_FILE_DIST = $NEW_TARGET_DIST|g" "$TEMP_PY"
        sed -i "s|^TARGET_FILE_DIST_LOG = .*|TARGET_FILE_DIST_LOG = $NEW_TARGET_DIST_LOG|g" "$TEMP_PY"
        
        # 4. Запускаємо модифікований скрипт
        python3 "$TEMP_PY"
        
        # 5. Видаляємо тимчасовий скрипт після виконання
        rm -f "$TEMP_PY"
        
        echo "Завершено для: $METHOD. Результати збережено у $OUT_DIR"
        echo ""
    done
done

echo "Всі експерименти успішно завершено! Дані для таблиці зібрано."