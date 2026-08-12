from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import os

from numba.core.cgutils import terminate
from scipy import stats

# Ігнорування FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Оголошення property_names тут
# property_names = ['Norm Area', 'Perimeter', 'Shape Factor', 'ECR', 'Orientation', 'Scale Factor', 'Center of Mass X',
#                   'Center of Mass Y', 'Inertia Tensor XX', 'Inertia Tensor XY', 'Determinant', 'Aspect Ratio',
#                   'Compactness Ratio', 'area-to-ellipse Ratio', 'Inertia Tensor_Area XX', 'Inertia Tensor_Area XY']

# property_names = ['Norm Area', 'Perimeter', 'Shape Factor', 'ECR', 'Orientation', 'Scale Factor', 'Center of Mass X',
#                   'Center of Mass Y', 'Inertia Tensor XX', 'Inertia Tensor XY', 'Determinant']

property_names = ['Inertia Tensor/Area XX', 'Inertia Tensor/Area XY', 'Inertia Tensor/Area YY']


def read_csv(filename):
    # Зчитуємо тільки зазначені стовпці
    try:
        df = pd.read_csv(filename, usecols=property_names)
    except ValueError as e:
        print(f"Помилка: {e}. Перевірте, чи всі зазначені властивості присутні у файлі.")
        return None

    # Замінюємо всі значення inf на NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Видаляємо рядки з NaN значеннями
    df.dropna(inplace=True)

    # Перетворюємо DataFrame у NumPy масив для подальшої обробки
    data = df.values

    return data


def process_data(data):
    df = pd.DataFrame(data, columns=property_names)

    # Створюємо булевий індекс, який показує, чи кожна властивість знаходиться в межах діапазону
    valid_rows = ((df >= df.mean() - 1.2 * df.std()) & (df <= df.mean() + 1.2 * df.std())).all(axis=1)

    # # Додамо умову для видалення зерен з Shape Factor більше за 1
    # valid_rows &= df['Shape Factor'] <= 1
    #
    # # Віддзеркалення значень орієнтації
    # for i, value in enumerate(df['Orientation']):
    #     if value > np.pi / 2:
    #         value -= np.pi
    #     elif value < -np.pi / 2:
    #         value += np.pi
    #     df.at[i, 'Orientation'] = value
    #
    # # Перевірка мінімального та максимального значення орієнтації після відображення
    # min_orientation_after = df['Orientation'].min()
    # max_orientation_after = df['Orientation'].max()
    # print("Мінімальне значення орієнтації після відображення:", min_orientation_after)
    # print("Максимальне значення орієнтації після відображення:", max_orientation_after)

    # Відбираємо лише ті рядки, де всі властивості знаходяться в межах діапазону
    df = df[valid_rows]

    print(df.head())
    print("-----------------------------------------")

    return df.values


def plot_and_save_violinplot(data, property_name):
    filtered_data = data[data['Property'] == property_name]  # Фільтрація даних
    # Додаємо перевірку
    if filtered_data.empty:
        print(f"Немає даних для властивості: {property_name}")
        return  # Якщо даних немає, виходимо з функції

    plt.figure(figsize=(6, 4))
    sns.violinplot(x='Property', y='Value', hue='Category', data=data[data['Property'] == property_name],
                   palette="flare", split=True)
    plt.title(property_name)
    plt.xlabel('Property')
    plt.ylabel('Value')

    for category in ['Generated', 'Real']:
        if category in data['Category'].values:
            category_data = data[(data['Property'] == property_name) & (data['Category'] == category)]['Value']
            Mean = np.mean(category_data)
            std = np.std(category_data)
            Median = np.median(category_data)

            align = 'left' if category == 'Generated' else 'right'

            plt.text(0.01 if align == 'left' else 0.99, 0.98, f'mean: {Mean:.4f}',
                     transform=plt.gca().transAxes, fontsize=7, verticalalignment='top', weight='bold',
                     horizontalalignment=align)
            plt.text(0.01 if align == 'left' else 0.99, 0.93, f'std: {std:.4f}',
                     transform=plt.gca().transAxes, fontsize=7, verticalalignment='top', weight='bold',
                     horizontalalignment=align)
            plt.text(0.01 if align == 'left' else 0.99, 0.88, f'median: {Median:.4f}',
                     transform=plt.gca().transAxes, fontsize=7, verticalalignment='top', weight='bold',
                     horizontalalignment=align)

        # 1. Очищуємо ім'я від косих рисок
        safe_property_name = property_name.replace('/', '_')

        # 2. Формуємо повне ім'я файлу (як у вас у коді)
        output_filename = f'./WE43-0P/WE43-0P_Radial_{safe_property_name}_violinplot.png'

        target_dir = os.path.dirname(output_filename)
        os.makedirs(target_dir, exist_ok=True)
        # -----------------------------------------------

        # 3. Зберігаємо графік
        plt.savefig(output_filename, format='png', bbox_inches='tight', dpi=300)


# Зчитування даних з файлів
generated_image_properties = read_csv(
    r"D:\Project(MatViz3D)\Random\Paper_Optimisation\Zp_paper_result\Result\WE43-0P\Differential_Evolution\SMAPE\iteration_data\0035\properties_output.csv")
real_image_properties = read_csv(
    r'D:\University\MatViz\1FULL_MINIMIZE_SCRIPTS\Target_img\WE43-0P\processed_output\processed_image_properties_(WE43-0P).csv')

generated_image_properties = process_data(generated_image_properties)
real_image_properties = process_data(real_image_properties)

selected_properties = ['Inertia Tensor/Area XX', 'Inertia Tensor/Area XY', 'Inertia Tensor/Area YY']

# Підготовка даних для відображення
generated_data = {
    'Category': ['Generated'] * generated_image_properties.shape[0] * len(selected_properties),
    'Property': np.tile(selected_properties, generated_image_properties.shape[0]),
    'Value': generated_image_properties[:, [property_names.index(prop) for prop in selected_properties]].flatten()
}

real_data = {
    'Category': ['Real'] * real_image_properties.shape[0] * len(selected_properties),
    'Property': np.tile(selected_properties, real_image_properties.shape[0]),
    'Value': real_image_properties[:, [property_names.index(prop) for prop in selected_properties]].flatten()
}

df_combined = pd.concat([pd.DataFrame(generated_data), pd.DataFrame(real_data)], ignore_index=True)

# # Створення фігури та сітки вісей для кожного графіка
# fig = plt.figure(figsize=(17, 8))
# gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.1])
#
# axes = np.empty((2, 3), dtype=object)
#
# for i in range(2):
#     for j in range(3):
#         axes[i, j] = plt.subplot(gs[i, j])
#         # axes[i, j].grid(True)  # Додана сітка
#
# # Побудова графіка для кожної властивості
# for i, prop in enumerate(selected_properties):
#     row, col = divmod(i, 3)
#     sns.violinplot(x='Property', y='Value', hue='Category', data=df_combined[df_combined['Property'] == prop],
#                    palette="viridis", split=True, ax=axes[row, col])
#     axes[row, col].set_title(prop)  # Додавання заголовку для кожного графіка
#
#     # Додавання інформації
#     for category in ['Generated', 'Real']:
#         if category in df_combined['Category'].values:
#             Mean = np.mean(
#                 df_combined[(df_combined['Property'] == prop) & (df_combined['Category'] == category)]['Value'])
#             std = np.std(
#                 df_combined[(df_combined['Property'] == prop) & (df_combined['Category'] == category)]['Value'])
#             Median = np.median(
#                 df_combined[(df_combined['Property'] == prop) & (df_combined['Category'] == category)]['Value'])
#
#             align = 'left' if category == 'Generated' else 'right'
#
#             axes[row, col].text(0.01 if align == 'left' else 0.99, 0.98, f'mean: {Mean:.4f}',
#                                 transform=axes[row, col].transAxes, fontsize=7, verticalalignment='top', weight='bold',
#                                 horizontalalignment=align)
#             axes[row, col].text(0.01 if align == 'left' else 0.99, 0.93, f'std: {std:.4f}',
#                                 transform=axes[row, col].transAxes, fontsize=7, verticalalignment='top', weight='bold',
#                                 horizontalalignment=align)
#             axes[row, col].text(0.01 if align == 'left' else 0.99, 0.88, f'median: {Median:.4f}',
#                                 transform=axes[row, col].transAxes, fontsize=7, verticalalignment='top', weight='bold',
#                                 horizontalalignment=align)
#
# # Прибирання легенди з кожного графіка
# for ax in axes.flat:
#     ax.get_legend().remove()
#
# # Додавання загальної легенди
# ax_legend = plt.subplot(gs[:, -1])
# ax_legend.legend(*axes[0, 0].get_legend_handles_labels(), title='Category')
# ax_legend.axis('off')  # Вимикаємо вісь для легенди
#
# # Встановлення логарифмічного масштабу для осі y
# # plt.yscale('log')
#
# # Зберігаємо графіки у файл
# # file_path_pdf = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
# # plt.savefig(file_path_pdf, format='png', bbox_inches='tight', dpi=300)
#
# plt.tight_layout()  # Розташування графіків

# plot_and_save_violinplot(df_combined, 'Area')
# plot_and_save_violinplot(df_combined, 'Shape Factor')
# plot_and_save_violinplot(df_combined, 'ECR')
# plot_and_save_violinplot(df_combined, 'Orientation')
# plot_and_save_violinplot(df_combined, 'Scale Factor')
# plot_and_save_violinplot(df_combined, 'Inertia Tensor XX')
# plot_and_save_violinplot(df_combined, 'Inertia Tensor XY')
# plot_and_save_violinplot(df_combined, 'Aspect Ratio')
# plot_and_save_violinplot(df_combined, 'Compactness Ratio')
# plot_and_save_violinplot(df_combined, 'area-to-ellipse Ratio')
# plot_and_save_violinplot(df_combined, 'Inertia Tensor_Area XX')
# plot_and_save_violinplot(df_combined, 'Inertia Tensor_Area XY')

plot_and_save_violinplot(df_combined, 'Inertia Tensor/Area XX')
plot_and_save_violinplot(df_combined, 'Inertia Tensor/Area XY')
plot_and_save_violinplot(df_combined, 'Inertia Tensor/Area YY')

plt.show()

# Виведення таблиці з обрізаними даними
print(" " * 35 + "|   Real     |  Generated |")
print("-" * 65)


# Проходження по кожній властивості
for prop in selected_properties:
    real_mean = np.mean(real_image_properties[:, property_names.index(prop)])
    real_std = np.std(real_image_properties[:, property_names.index(prop)])
    real_median = np.median(real_image_properties[:, property_names.index(prop)])
    real_mode = float(stats.mode(real_image_properties[:, property_names.index(prop)])[0])  # Мода
    real_range = np.ptp(real_image_properties[:, property_names.index(prop)])  # Діапазон
    real_iqr = np.percentile(real_image_properties[:, property_names.index(prop)], 75) - \
               np.percentile(real_image_properties[:, property_names.index(prop)], 25)  # Міжквартильний діапазон
    real_q1 = np.percentile(real_image_properties[:, property_names.index(prop)], 25)  # Перший квартиль
    real_q3 = np.percentile(real_image_properties[:, property_names.index(prop)], 75)  # Третій квартиль

    generated_mean = np.mean(generated_image_properties[:, property_names.index(prop)])
    generated_std = np.std(generated_image_properties[:, property_names.index(prop)])
    generated_median = np.median(generated_image_properties[:, property_names.index(prop)])
    generated_mode = float(stats.mode(generated_image_properties[:, property_names.index(prop)])[0])  # Мода
    generated_range = np.ptp(generated_image_properties[:, property_names.index(prop)])  # Діапазон
    generated_iqr = np.percentile(generated_image_properties[:, property_names.index(prop)], 75) - \
                    np.percentile(generated_image_properties[:, property_names.index(prop)],
                                  25)  # Міжквартильний діапазон
    generated_q1 = np.percentile(generated_image_properties[:, property_names.index(prop)], 25)  # Перший квартиль
    generated_q3 = np.percentile(generated_image_properties[:, property_names.index(prop)], 75)  # Третій квартиль

    # Виведення рядка таблиці з відповідними значеннями
    print(f"{prop:<24} | Mean    | {real_mean:^10.4f} | {generated_mean:^10.4f} |")
    print(f"{'':<24} | std     | {real_std:^10.4f} | {generated_std:^10.4f} |")
    print(f"{'':<24} | Median  | {real_median:^10.4f} | {generated_median:^10.4f} |")
    print(f"{'':<24} | Mode    | {real_mode:^10.4f} | {generated_mode:^10.4f} |")
    print(f"{'':<24} | Range   | {real_range:^10.4f} | {generated_range:^10.4f} |")
    print(f"{'':<24} | IQR     | {real_iqr:^10.4f} | {generated_iqr:^10.4f} |")
    print(f"{'':<24} | Q1      | {real_q1:^10.4f} | {generated_q1:^10.4f} |")
    print(f"{'':<24} | Q3      | {real_q3:^10.4f} | {generated_q3:^10.4f} |")
    print("-" * 65)

