import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate


# Функція для зчитування даних з CSV файлу
def read_csv(filename, usecols):
    chunksize = 10000  # Розмір порції
    data_list = []

    for chunk in pd.read_csv(filename, delimiter=',', skiprows=1, chunksize=chunksize, usecols=usecols,
                             encoding='utf-8'):
        # Спроба конвертувати всі значення у числові, нечислові значення стануть NaN
        chunk = chunk.apply(pd.to_numeric, errors='coerce')
        data_list.append(chunk)

    data = pd.concat(data_list, ignore_index=True)
    data = data.dropna()  # Видалення рядків з NaN значеннями
    return data.to_numpy()


# Функція для фільтрації даних за середнім значенням та стандартним відхиленням
def filter_data(data, property_names):
    df = pd.DataFrame(data, columns=property_names[:11])

    mean = df.mean()
    std = df.std()
    valid_rows = ((df >= mean - 1.5 * std) & (df <= mean + 1.5 * std)).all(axis=1)
    valid_rows &= df.iloc[:, property_names.index('Shape Factor')] <= 1

    return df[valid_rows].values


# Функція для відзеркалення значень орієнтації
def reflect_orientation(data, property_names):
    df = pd.DataFrame(data, columns=property_names[:11])

    for i, value in enumerate(df.iloc[:, property_names.index('Orientation')]):
        if value > np.pi / 2:
            value -= np.pi
        elif value < -np.pi / 2:
            value += np.pi
        df.at[i, 'Orientation'] = value

    return df.values


# Функція для обробки даних та побудови гістограм
def process_and_plot(directory_path, property_name, apply_data_processing=True, plot_histograms=False):
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]

    statistics = []

    print(f"Processing {len(file_paths)} files...")

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        data = read_csv(file_path, usecols=list(range(11)))

        if apply_data_processing:
            # data = filter_data(data, property_names)
            data = reflect_orientation(data, property_names)

        property_values = data[:, property_names.index(property_name)]
        print(f"Data shape: {data.shape}, Property values length: {len(property_values)}")

        # Обчислення статистичних даних
        mean = np.mean(property_values)
        std_dev = np.std(property_values)
        median = np.median(property_values)

        if len(property_values) > 0:
            values, counts = np.unique(property_values, return_counts=True)
            mode = values[np.argmax(counts)]
        else:
            mode = np.nan

        data_range = np.ptp(property_values)
        iqr = np.percentile(property_values, 75) - np.percentile(property_values, 25)
        lower_quartile = np.percentile(property_values, 25)
        upper_quartile = np.percentile(property_values, 75)
        q0 = np.percentile(property_values, 0.01)
        q4 = np.percentile(property_values, 99.9)

        statistics.append([
            os.path.basename(file_path), mean, std_dev, median, mode, data_range, iqr, lower_quartile,
            upper_quartile, q0, q4
        ])

        if plot_histograms:
            num_bins = int(len(property_values))
            plt.figure(figsize=(10, 8))
            plt.hist(property_values, bins=num_bins, color='blue', edgecolor='black', linewidth=1.2, alpha=0.7)
            plt.title(f'File: {os.path.basename(file_path)}, Property: {property_name}')
            plt.xlabel(property_name)
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()

            # Збереження зображення з назвою файлу та властивості
            base_name = os.path.basename(file_path).split(".")[0]
            output_file = os.path.join(directory_path, f'img_{base_name}_{property_name}.png')
            plt.savefig(output_file, dpi=400, bbox_inches='tight')
            plt.close()

    # Виведення статистичних даних у консоль
    headers = ["File Name", "Mean", "Standard Deviation", "Median", "Mode", "Range", "IQR", "Lower Quartile",
               "Upper Quartile", "Q0.01", "Q99.9"]
    print(tabulate(statistics, headers=headers, tablefmt="simple", floatfmt=".7f", numalign="right", stralign="center"))

    # Повернення статистики для подальшого використання
    return statistics


def plot_statistics(statistics, x_values, y_property_name, poly_degree, plt_title, output_path, img_name,
                    mean_value=None):
    # Задаємо значення для осі X
    files = x_values

    # Вибір значень для осі Y на основі вибраної статистичної властивості
    y_values = {
        'Mean': [stat[1] for stat in statistics],
        'Standard Deviation': [stat[2] for stat in statistics],
        'Median': [stat[3] for stat in statistics],
        'Mode': [stat[4] for stat in statistics],
        'Range': [stat[5] for stat in statistics],
        'IQR': [stat[6] for stat in statistics],
        'Lower Quartile': [stat[7] for stat in statistics],
        'Upper Quartile': [stat[8] for stat in statistics],
        'Q0.01': [stat[9] for stat in statistics],
        'Q99.9': [stat[10] for stat in statistics]
    }[y_property_name]

    # Задаємо квантилі для побудови на графіку
    lower_quartiles = [stat[7] for stat in statistics]
    upper_quartiles = [stat[8] for stat in statistics]
    q0 = [stat[9] for stat in statistics]
    q4 = [stat[10] for stat in statistics]

    plt.figure(figsize=(12, 8))

    # Побудова лінії для вибраної статистичної властивості
    plt.plot(files, y_values, label=y_property_name, color='blue', marker='o')

    # Побудова лінії для квантилів Q1 та Q3
    plt.fill_between(files, lower_quartiles, upper_quartiles, color='blue', alpha=0.25, label='Q1 - Q3')

    # Побудова лінії для квантилів Q0 та Q4
    # plt.fill_between(files, q0, q4, color='blue', alpha=0.2, label='Q0 - Q4')

    # Апроксимація поліномом
    coefficients = np.polyfit(range(len(files)), y_values, poly_degree)
    polynomial = np.polyval(coefficients, range(len(files)))

    # Обчислення R^2
    ss_res = np.sum((y_values - polynomial) ** 2)
    ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Додавання лініії поліноміальної апроксимації на графік
    plt.plot(files, polynomial, color='red', linestyle='--', label=f'Poly fit (degree {poly_degree})')

    # Формування рівняння полінома для відображення на графіку
    poly_equation = ' + '.join([f'{coeff:.5g}$x^{poly_degree - i}$' for i, coeff in enumerate(coefficients[:-1])])
    poly_equation += f' + {coefficients[-1]:.5g}'

    # Виведення рівняння полінома та R^2 на графік
    plt.text(0.05, 0.95, f'y = {poly_equation}', transform=plt.gca().transAxes, fontsize=13,
             verticalalignment='top')
    plt.text(0.05, 0.90, f'$R^2$ = {r_squared:.4f}', transform=plt.gca().transAxes, fontsize=13,
             verticalalignment='top')

    # Додавання горизонтальної лінії для середнього значення
    if mean_value is not None:
        plt.axhline(y=mean_value, color='darkred', linestyle='--', label=f'Real mean = {mean_value}')

    plt.title(f'{plt_title}')
    plt.xlabel('Concentration')
    # plt.xlabel('Size')
    plt.ylabel(y_property_name)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Збереження зображення з назвою властивості
    output_file = os.path.join(output_path, img_name)
    plt.savefig(output_file, dpi=600, bbox_inches='tight')

    print(f"Plot saved as {output_file}")


# Функція для побудови графіку з кількома середніми значеннями
def plot_combined_means(mean_lists, labels, title, output_path, img_name, mean_value=None):
    x_values = [10, 20, 30, 40, 50, 60]  # Задання значень для осі X
    plt.figure(figsize=(12, 8))

    for mean_values, label in zip(mean_lists, labels):
        plt.plot(x_values, mean_values[:len(x_values)], marker='o',
                 label=label)  # Обрізка mean_values до довжини x_values

    # Додавання горизонтальної лінії для середнього значення
    if mean_value is not None:
        plt.axhline(y=mean_value, color='darkred', linestyle='--', label=f'Real mean = {mean_value}')

    plt.title(title)
    plt.xlabel('Size')  # Тут можна змінити підпис для осі X
    plt.ylabel('Mean Value')
    plt.xticks(x_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Збереження зображення
    output_file = os.path.join(output_path, img_name)
    plt.savefig(output_file, dpi=600, bbox_inches='tight')

    print(f"Combined mean plot saved as {output_file}")


if __name__ == '__main__':
    # Використання функції для побудови графіка зі статистичними характеристиками
    property_names = ['Norm Area', 'Perimeter', 'Shape Factor', 'ECR',
                      'Orientation', 'Scale Factor', 'Aspect Ratio',
                      'Compactness Ratio', 'area-to-ellipse Ratio',
                      'Center of Mass X', 'Center of Mass Y',
                      'Inertia Tensor XX', 'Inertia Tensor XY', 'Inertia Tensor YX', 'Inertia Tensor YY',
                      'Inertia Tensor/Area XX', 'Inertia Tensor/Area XY', 'Inertia Tensor/Area YX',
                      'Inertia Tensor/Area YY']

    mean_normArea_real = 0.003
    mean_ecr_real = 0.03
    mean_Orientation_real = -0.574
    mean_Scale_Factor_real = 1.95

    # print('----------------------------------------------')
    # use_property = 'Norm Area'
    # x_values = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
    # # x_values = ['10', '20', '30', '40', '50', '60', '70', '80']
    # # plt_stat_title = 'Moore (wave), Norm Area, Concentration(const): 20'
    # plt_stat_title = 'Probability Ellipse (wave), Norm Area, Size(const): 35'
    # directory_path = r"C:\Users\user\OneDrive - Нацiональний технiчний унiверситет Харкiвський полiтехнiчний iнститут\MatViz3D\Summer 2024\wave\ellipse\s(const)35\Result"
    # output_img_path = r"C:\Users\user\OneDrive - Нацiональний технiчний унiверситет Харкiвський полiтехнiчний iнститут\MatViz3D\Summer 2024\img_plot"
    # # img_stat_name = f'1moore(wave)_NormArea_Concentration(const)20.png'
    # img_stat_name = f'1ellipse(wave)_NormArea_Size(const)35.png'
    #
    # # Побудова гістограм для вибраної властивості та побудова графіка статистики
    # statistics = process_and_plot(directory_path, use_property)
    # print('------------------------------')
    # plot_statistics(statistics, x_values, 'Mean', poly_degree=3, plt_title=plt_stat_title,
    #                 output_path=output_img_path, img_name=img_stat_name, mean_value=mean_normArea_real)
    #
    # print('----------------------------------------------')
    # use_property = 'ECR'
    # x_values = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
    # # x_values = ['10', '20', '30', '40', '50', '60', '70', '80']
    # # plt_stat_title = 'Moore (wave), ECR, Concentration(const): 20'
    # plt_stat_title = 'Probability Ellipse (wave), ECR, Size(const): 35'
    # directory_path = r"C:\Users\user\OneDrive - Нацiональний технiчний унiверситет Харкiвський полiтехнiчний iнститут\MatViz3D\Summer 2024\wave\ellipse\s(const)35\Result"
    # output_img_path = r"C:\Users\user\OneDrive - Нацiональний технiчний унiверситет Харкiвський полiтехнiчний iнститут\MatViz3D\Summer 2024\img_plot"
    # # img_stat_name = f'1moore(wave)_ECR_Concentration(const)20.png'
    # img_stat_name = f'1ellipse(wave)_ECR_Size(const)35.png'
    #
    # # Побудова гістограм для вибраної властивості та побудова графіка статистики
    # statistics = process_and_plot(directory_path, use_property)
    # print('------------------------------')
    # plot_statistics(statistics, x_values, 'Mean', poly_degree=3, plt_title=plt_stat_title,
    #                 output_path=output_img_path, img_name=img_stat_name, mean_value=mean_ecr_real)
    #
    # print('----------------------------------------------')
    # use_property = 'Orientation'
    # x_values = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
    # # x_values = ['10', '20', '30', '40', '50', '60', '70', '80']
    # plt_stat_title = 'Probability Ellipse (wave), Orientation, Size(const): 35'
    # # plt_stat_title = 'Moore (wave), Orientation, Concentration(const): 20'
    # directory_path = r"C:\Users\user\OneDrive - Нацiональний технiчний унiверситет Харкiвський полiтехнiчний iнститут\MatViz3D\Summer 2024\wave\ellipse\s(const)35\Result"
    # output_img_path = r"C:\Users\user\OneDrive - Нацiональний технiчний унiверситет Харкiвський полiтехнiчний iнститут\MatViz3D\Summer 2024\img_plot"
    # # img_stat_name = f'1moore(wave)_Orientation_Concentration(const)20.png'
    # img_stat_name = f'1ellipse(wave)_Orientation_Size(const)35.png'
    #
    # # Побудова гістограм для вибраної властивості та побудова графіка статистики
    # statistics = process_and_plot(directory_path, use_property)
    # print('------------------------------')
    # plot_statistics(statistics, x_values, 'Mean', poly_degree=3, plt_title=plt_stat_title,
    #                 output_path=output_img_path, img_name=img_stat_name, mean_value=mean_Orientation_real)
    #
    # print('----------------------------------------------')
    # use_property = 'Scale Factor'
    # x_values = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
    # # x_values = ['10', '20', '30', '40', '50', '60', '70', '80']
    # plt_stat_title = 'Probability Ellipse (wave), Scale Factor, Size(const): 35'
    # # plt_stat_title = 'Moore (wave), Scale Factor, Concentration(const): 20'
    # directory_path = r"C:\Users\user\OneDrive - Нацiональний технiчний унiверситет Харкiвський полiтехнiчний iнститут\MatViz3D\Summer 2024\wave\ellipse\s(const)35\Result"
    # output_img_path = r"C:\Users\user\OneDrive - Нацiональний технiчний унiверситет Харкiвський полiтехнiчний iнститут\MatViz3D\Summer 2024\img_plot"
    # # img_stat_name = f'1moore(wave)_ScaleFactor_Concentration(const)20.png'
    # img_stat_name = f'1ellipse(wave)_ScaleFactor_Size(const)35.png'
    #
    # # Побудова гістограм для вибраної властивості та побудова графіка статистики
    # statistics = process_and_plot(directory_path, use_property)
    # print('------------------------------')
    # plot_statistics(statistics, x_values, 'Mean', poly_degree=3, plt_title=plt_stat_title,
    #                 output_path=output_img_path, img_name=img_stat_name, mean_value=mean_Scale_Factor_real)

    directories = [
        r"D:\Project(MatViz3D)\summer2024\wave\moore\c(const)10\Result",
        r"D:\Project(MatViz3D)\summer2024\wave\moore\c(const)20\Result",
        r"D:\Project(MatViz3D)\summer2024\wave\moore\c(const)50\Result"
    ]
    use_property = 'Norm Area'
    output_img_path = r"C:\Users\user\OneDrive - Нацiональний технiчний унiверситет Харкiвський полiтехнiчний iнститут\MatViz3D\Summer 2024\img_plot"
    img_stat_name = f'wave_c(const)_combined_mean_NormArea_Moore.png'
    plt_stat_title = 'Combined Mean Values for Norm Area; Moore (wave)'

    mean_lists = []
    # labels = ['Moore', 'Probability Circle', 'Probability Ellipse']
    labels = ['Concentration: 10', 'Concentration: 20', 'Concentration: 50']

    for directory in directories:
        statistics = process_and_plot(directory, use_property)
        means = [stat[1] for stat in statistics]  # Отримання середніх значень
        mean_lists.append(means)
        labels.append(f"Data from {os.path.basename(directory)}")

    # Побудова графіку з декількома середніми значеннями
    plot_combined_means(mean_lists, labels, plt_stat_title, output_img_path, img_stat_name,
                        mean_value=mean_normArea_real)

    #--------------------------------------------

    directories = [
        r"D:\Project(MatViz3D)\summer2024\wave\moore\c(const)10\Result",
        r"D:\Project(MatViz3D)\summer2024\wave\moore\c(const)20\Result",
        r"D:\Project(MatViz3D)\summer2024\wave\moore\c(const)50\Result"
    ]
    use_property = 'ECR'
    output_img_path = r"C:\Users\user\OneDrive - Нацiональний технiчний унiверситет Харкiвський полiтехнiчний iнститут\MatViz3D\Summer 2024\img_plot"
    img_stat_name = f'wave_c(const)_combined_mean_ECR_Moore.png'
    plt_stat_title = 'Combined Mean Values for ECR; Moore (wave)'

    mean_lists = []
    # labels = ['Moore', 'Probability Circle', 'Probability Ellipse']
    labels = ['Concentration: 10', 'Concentration: 20', 'Concentration: 50']

    for directory in directories:
        statistics = process_and_plot(directory, use_property)
        means = [stat[1] for stat in statistics]  # Отримання середніх значень
        mean_lists.append(means)
        labels.append(f"Data from {os.path.basename(directory)}")

    # Побудова графіку з декількома середніми значеннями
    plot_combined_means(mean_lists, labels, plt_stat_title, output_img_path, img_stat_name,
                        mean_value=mean_ecr_real)

    #-----------------------------------------------

    directories = [
        r"D:\Project(MatViz3D)\summer2024\wave\moore\c(const)10\Result",
        r"D:\Project(MatViz3D)\summer2024\wave\moore\c(const)20\Result",
        r"D:\Project(MatViz3D)\summer2024\wave\moore\c(const)50\Result"
    ]
    use_property = 'Orientation'
    output_img_path = r"C:\Users\user\OneDrive - Нацiональний технiчний унiверситет Харкiвський полiтехнiчний iнститут\MatViz3D\Summer 2024\img_plot"
    img_stat_name = f'wave_c(const)_combined_mean_Orientation_Moore.png'
    plt_stat_title = 'Combined Mean Values for Orientation; Moore (wave)'

    mean_lists = []
    # labels = ['Moore', 'Probability Circle', 'Probability Ellipse']
    labels = ['Concentration: 10', 'Concentration: 20', 'Concentration: 50']

    for directory in directories:
        statistics = process_and_plot(directory, use_property)
        means = [stat[1] for stat in statistics]  # Отримання середніх значень
        mean_lists.append(means)
        labels.append(f"Data from {os.path.basename(directory)}")

    # Побудова графіку з декількома середніми значеннями
    plot_combined_means(mean_lists, labels, plt_stat_title, output_img_path, img_stat_name,
                        mean_value=mean_Orientation_real)

    #------------------------------------------------

    directories = [
        r"D:\Project(MatViz3D)\summer2024\wave\moore\c(const)10\Result",
        r"D:\Project(MatViz3D)\summer2024\wave\moore\c(const)20\Result",
        r"D:\Project(MatViz3D)\summer2024\wave\moore\c(const)50\Result"
    ]
    use_property = 'Scale Factor'
    output_img_path = r"C:\Users\user\OneDrive - Нацiональний технiчний унiверситет Харкiвський полiтехнiчний iнститут\MatViz3D\Summer 2024\img_plot"
    img_stat_name = f'wave_c(const)_combined_mean_ScaleFactor_Moore.png'
    plt_stat_title = 'Combined Mean Values for Scale Factor; Moore (wave)'

    mean_lists = []
    # labels = ['Moore', 'Probability Circle', 'Probability Ellipse']
    labels = ['Concentration: 10', 'Concentration: 20', 'Concentration: 50']

    for directory in directories:
        statistics = process_and_plot(directory, use_property)
        means = [stat[1] for stat in statistics]  # Отримання середніх значень
        mean_lists.append(means)
        labels.append(f"Data from {os.path.basename(directory)}")

    # Побудова графіку з декількома середніми значеннями
    plot_combined_means(mean_lists, labels, plt_stat_title, output_img_path, img_stat_name,
                        mean_value=mean_Scale_Factor_real)

    plt.show()
    plt.close()
