import numpy as np
import skimage
from skimage.measure import regionprops, find_contours
import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd


all_properties = {
        'Area': [], 'Perimeter': [], 'Shape Factor': [], 'ECR': [],
        'Orientation': [], 'Scale Factor': [], 'Aspect Ratio': [],
        'Compactness Ratio': [], 'area-to-ellipse Ratio': [],
        'Center of Mass X': [], 'Center of Mass Y': [],
        'Inertia Tensor XX': [], 'Inertia Tensor XY': [], 'Inertia Tensor YX': [], 'Inertia Tensor YY': [],
        'Inertia Tensor/Area XX': [], 'Inertia Tensor/Area XY': [], 'Inertia Tensor/Area YX': [],
        'Inertia Tensor/Area YY': []
    }


def process_and_plot(file_path):
    data = np.genfromtxt(file_path, delimiter=';', skip_header=1, dtype=int)
    cube_size = np.max(data[:, :3]) + 1
    layers = np.zeros((cube_size, cube_size, cube_size), dtype=int)

    # Fill the layers array
    layers[data[:, 0], data[:, 1], data[:, 2]] = data[:, 3]

    return process_layers(layers)


def calculate_perimeter(grain_mask):
    contours = find_contours(grain_mask, 0.5)
    perimeter = sum(np.sum(np.round(contour).astype(int)) for contour in contours)
    return perimeter


def process_layers(layers):
    cube_size = layers.shape[0]

    for z in range(cube_size):
        layer_area = np.prod(layers[:, :, z].shape)  # Площа шару
        unique_colors = np.unique(layers[:, :, z])
        for grain_color in unique_colors:
            grain_mask = (layers[:, :, z] == grain_color)
            labeled_grains = skimage.measure.label(grain_mask, connectivity=2)
            for region in regionprops(labeled_grains):
                if region.area > 1:
                    # Пропустити зерна на краях
                    if all(coord > 0 and coord < cube_size - 1 for coord in region.coords[:, 0]) and \
                            all(coord > 0 and coord < cube_size - 1 for coord in region.coords[:, 1]):
                        # Calculate properties
                        area = region.area
                        norm_area = area / layer_area  # Нормалізація площі
                        perimeter = calculate_perimeter(grain_mask)
                        shape_factor = 4. * np.pi * area / (perimeter ** 2)
                        bbox_area = region.bbox_area / area if area != 0 else 0
                        convex_area = region.convex_area / area if area != 0 else 0

                        scale_factor = region.major_axis_length / region.minor_axis_length if region.minor_axis_length != 0 else np.inf
                        major_minor_area = area / (
                                    region.major_axis_length * region.minor_axis_length) if region.minor_axis_length != 0 else 0

                        orientation = region.orientation
                        center_of_mass = region.centroid
                        inertia_tensor = region.inertia_tensor
                        inertia_tensor_area = inertia_tensor / area if area != 0 else np.zeros_like(inertia_tensor)
                        ecr = np.sqrt(norm_area / np.pi)

                        # Append properties
                        all_properties['Area'].append(norm_area)
                        all_properties['Perimeter'].append(perimeter)
                        all_properties['Shape Factor'].append(shape_factor)
                        all_properties['ECR'].append(ecr)
                        all_properties['Orientation'].append(orientation)
                        if not np.isinf(scale_factor):
                            all_properties['Scale Factor'].append(scale_factor)
                        all_properties['Center of Mass X'].append(center_of_mass[0])
                        all_properties['Center of Mass Y'].append(center_of_mass[1])
                        all_properties['Inertia Tensor XX'].append(inertia_tensor[0, 0])
                        all_properties['Inertia Tensor XY'].append(inertia_tensor[0, 1])
                        all_properties['Inertia Tensor YX'].append(inertia_tensor[1, 0])
                        all_properties['Inertia Tensor YY'].append(inertia_tensor[1, 1])
                        all_properties['Inertia Tensor/Area XX'].append(inertia_tensor_area[0, 0])
                        all_properties['Inertia Tensor/Area XY'].append(inertia_tensor_area[0, 1])
                        all_properties['Inertia Tensor/Area YX'].append(inertia_tensor_area[1, 0])
                        all_properties['Inertia Tensor/Area YY'].append(inertia_tensor_area[1, 1])
                        all_properties['Aspect Ratio'].append(bbox_area)
                        all_properties['Compactness Ratio'].append(convex_area)
                        all_properties['area-to-ellipse Ratio'].append(major_minor_area)

    return all_properties


def pad_to_max_length(properties):
    max_len = max(len(lst) for lst in properties.values())
    for key, lst in properties.items():
        if len(lst) < max_len:
            properties[key].extend([None] * (max_len - len(lst)))
    return properties


def save_properties_to_csv(output_file_path, all_properties):
    all_properties = pad_to_max_length(all_properties)  # Pad to max length
    df = pd.DataFrame(all_properties)
    df.to_csv(output_file_path, index=False)


def process_all_files_in_directory(directory_path, output_file_path):
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_and_plot, os.path.join(directory_path, filename))
                   for filename in os.listdir(directory_path) if filename.endswith(".csv")]
        for future in futures:
            file_properties = future.result()
            for key in all_properties.keys():
                all_properties[key].extend(file_properties[key])

    save_properties_to_csv(output_file_path, all_properties)


def process_multiple_directories(base_directory, concentrations, suffix, output_directory):
    for concentration in concentrations:
        directory_path = os.path.join(base_directory, f"{concentration}{suffix}")
        output_file_path = os.path.join(output_directory, f"RESULT({concentration}{suffix}).csv")
        process_all_files_in_directory(directory_path, output_file_path)
        print(f"Complete RESULT({concentration}{suffix}).csv")


if __name__ == "__main__":
    base_directory = r"D:\Project(MatViz3D)\c(const)50"
    concentrations = ["10c", "20c", "30c", "40c", "50c", "60c", "70c", "80c", "90c"]
    sizes = ["10s"]
    suffix_c = "_35s"
    suffix_s = "_50c"
    output_directory = os.path.join(base_directory, "Result")

    process_multiple_directories(base_directory, sizes, suffix_s, output_directory)
