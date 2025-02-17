import csv
import logging
import numpy as np
import skimage
from scipy.optimize import minimize, dual_annealing
from skimage.measure import regionprops
from MatViz3DLauncher import MatViz3DLauncher

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Fixed cube size
FIXED_SIZE = 50


def start(launcher, size, concentration, halfaxis_a, halfaxis_b, halfaxis_c, orientation_angle_a, orientation_angle_b,
          orientation_angle_c, wave_coefficient, output_file):
    """
    Runs MatViz3D with given parameters and returns the output file.
    """
    print(f"Starting MatViz3D with parameters: size={size}, concentration={concentration}, "
          f"halfaxis_a={halfaxis_a}, halfaxis_b={halfaxis_b}, halfaxis_c={halfaxis_c}, "
          f"orientation_angle_a={orientation_angle_a}, orientation_angle_b={orientation_angle_b}, "
          f"orientation_angle_c={orientation_angle_c}, wave_coefficient={wave_coefficient}")
    return launcher.start(
        size=size,
        concentration=concentration,
        halfaxis_a=halfaxis_a,
        halfaxis_b=halfaxis_b,
        halfaxis_c=halfaxis_c,
        orientation_angle_a=orientation_angle_a,
        orientation_angle_b=orientation_angle_b,
        orientation_angle_c=orientation_angle_c,
        wave_coefficient=wave_coefficient,
        output_file=output_file
    )


def calculate_mean_ecr(layers):
    """
    Calculates the mean Equivalent Circle Radius (ECR) from the given 3D layers.
    """
    cube_size = layers.shape[0]
    ecr_values = []

    for axis in ['z', 'x', 'y']:
        for index in range(cube_size):
            layer = layers[:, :, index] if axis == 'z' else layers[index, :, :] if axis == 'x' else layers[:, index, :]
            unique_colors = set(layer.flatten())

            for grain_color in unique_colors:
                grain_mask = (layer == grain_color)
                labeled_grains = skimage.measure.label(grain_mask, connectivity=2)

                for region in regionprops(labeled_grains):
                    if region.area > 1:
                        ecr = np.sqrt(region.area / (np.pi * np.prod(layer.shape)))
                        if np.isfinite(ecr):
                            ecr_values.append(ecr)

    return np.mean(ecr_values) if ecr_values else np.inf


def process_file(file_path):
    """
    Reads a CSV file and processes it to calculate the mean ECR.
    """
    try:
        data = np.genfromtxt(file_path, delimiter=';', skip_header=1, dtype=int)
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return None

    cube_size = np.max(data[:, :3]) + 1
    layers = np.zeros((cube_size, cube_size, cube_size), dtype=int)
    layers[data[:, 0], data[:, 1], data[:, 2]] = data[:, 3]

    return calculate_mean_ecr(layers)


def minimize_ecr(params, launcher, output_file, csv_filename, target_value):
    """
    Objective function for optimization: minimizes the absolute error of the mean ECR.
    """
    wave_coefficient, concentration, halfaxis_a, halfaxis_b, halfaxis_c, orientation_angle_a, orientation_angle_b, orientation_angle_c = params

    generated_file = start(launcher, FIXED_SIZE, concentration, halfaxis_a, halfaxis_b, halfaxis_c,
                           orientation_angle_a, orientation_angle_b, orientation_angle_c, wave_coefficient, output_file)

    if generated_file:
        mean_ecr = process_file(generated_file)
        if mean_ecr is None:
            logging.error("Failed to process ECR.")
            return np.inf

        abs_error = np.abs(mean_ecr - target_value)
        rel_error = (abs_error / mean_ecr) * 100 if mean_ecr != 0 else np.inf

        print(f"Mean ECR: {mean_ecr:.4f}")
        print(f"Absolute error: {abs_error}")
        print(f"Relative error: {rel_error:.2f}%")
        print(f"________________________________\n")

        # Append results to CSV file
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                wave_coefficient, concentration, halfaxis_a, halfaxis_b, halfaxis_c,
                orientation_angle_a, orientation_angle_b, orientation_angle_c,
                abs_error, mean_ecr, rel_error
            ])

        return abs_error
    else:
        logging.error("Failed to generate file.")
        return np.inf


def optimize_ecr(bounds, x0, launcher, output_file, csv_filename, target_value, method="minimize"):
    """
    Performs optimization to minimize the error in ECR calculation.
    """
    logging.info("Starting optimization...")

    if method == "minimize":
        result = minimize(
            minimize_ecr,
            x0=x0,
            args=(launcher, output_file, csv_filename, target_value),
            method='Nelder-Mead',
            options={'disp': True, 'maxiter': 50}
        )
    elif method == "dual_annealing":
        result = dual_annealing(
            minimize_ecr,
            bounds=bounds,
            args=(launcher, output_file, csv_filename, target_value),
            maxiter=50
        )
    else:
        raise ValueError("Unsupported optimization method")

    logging.info(f"Best parameters found: {result.x}")
    logging.info(f"Error value: {result.fun}")

    return result


def main():
    """
    Main function to execute the optimization process.
    """

    bounds = [
        (0, 1),  # wave_coefficient range
        (0.1, 95),  # concentration range
        (1, 1.5),  # halfaxis_a range
        (1, 1.5),  # halfaxis_b range
        (1, 1.5),  # halfaxis_c range
        (0, 360),  # orientation_angle_a range
        (0, 360),  # orientation_angle_b range
        (0, 360)  # orientation_angle_c range
    ]

    """
    Initial guess (starting point) for the optimization parameters
    Corresponds to the values for: wave_coefficient, concentration, halfaxis_a, halfaxis_b, halfaxis_c,
    orientation_angle_a, orientation_angle_b, orientation_angle_c
    """
    x0 = [0.2, 2, 1.4, 1.4, 1.4, 80, 80, 80]

    # Target value for ECR
    target_value = 0.0432

    # Path to the executable of MatViz3D that will be launched for each optimization run
    exe_path = r"D:\\Project(MatViz3D)\\GITHUB\\MatViz3D\\build\\Desktop_Qt_6_8_1_MinGW_64_bit-Debug\\debug\\MatViz3D.exe"

    # Path to the output file where the results of each optimization run will be saved
    output_file = "min_output.csv"

    # Path to the CSV file where optimization logs will be written (including parameter values, ECR, etc.)
    csv_filename = "optimization_log.csv"

    # Choose the optimization method (e.g., "minimize" or "dual_annealing")
    optimization_method = "minimize"

    launcher = MatViz3DLauncher(exe_path)
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "wave_coefficient", "concentration", "halfaxis_a", "halfaxis_b", "halfaxis_c",
            "orientation_angle_a", "orientation_angle_b", "orientation_angle_c",
            "fun", "mean_ecr", "relative_error(%)"
        ])

    result = optimize_ecr(bounds, x0, launcher, output_file, csv_filename, target_value, method=optimization_method)
    print(result)


if __name__ == "__main__":
    main()
