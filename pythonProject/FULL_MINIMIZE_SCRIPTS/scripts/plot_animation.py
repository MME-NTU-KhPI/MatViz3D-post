import os
import imageio.v2 as imageio

# ================= SETTINGS =================

# 1. Path to the main directory (where 0001, 0002, etc. are located)
# Note: do not put a trailing slash (\\) at the end of the path
BASE_DIR = r"D:\Project(MatViz3D)\Random\Paper_Optimisation\test\AZ31iA\Rezult\EnergyDistance_DifferentialEvolution_AZ31iA\histograms"

# 2. Speed options (FPS - frames per second). Multiple values can be specified.
# SPEEDS = [2, 5, 10]
SPEEDS = [10, 15]

# 3. Format options. Multiple formats can be specified ('gif', 'mp4', 'avi', etc.)
FORMATS = ['gif']

# 4. Which plots to animate? (Specify file endings)
# If the list is empty [], the script will animate ALL found plot types.
TARGET_PLOTS = [
    'Inertia_Tensor_Area_XX_hist.png',
    'Inertia_Tensor_Area_YY_hist.png',
    'Inertia_Tensor_Area_XY_hist.png'
]


# ================================================

def main():
    # Automatically generate the path for the output directory (next to BASE_DIR)
    parent_dir = os.path.dirname(BASE_DIR)
    base_folder_name = os.path.basename(BASE_DIR)
    output_dir = os.path.join(parent_dir, f"{base_folder_name}_Animations")

    # Create the main output directory and subdirectories for each format
    for fmt in FORMATS:
        os.makedirs(os.path.join(output_dir, fmt.lower()), exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Get a sorted list of all subdirectories (0001, 0002...)
    folders = sorted([f.path for f in os.scandir(BASE_DIR) if f.is_dir()])

    if not folders:
        print("Error: No subdirectories found.")
        return

    # Take the first folder to determine the available plot types
    first_folder = folders[0]
    sample_files = [f.name for f in os.scandir(first_folder) if f.is_file() and f.name.endswith('.png')]

    # Determine all available plot suffixes
    all_plot_suffixes = ["_".join(f.split("_")[1:]) for f in sample_files]

    # Filter plots if the user specified target plots in TARGET_PLOTS
    if TARGET_PLOTS:
        plot_suffixes = [s for s in all_plot_suffixes if any(target in s for target in TARGET_PLOTS)]
    else:
        plot_suffixes = all_plot_suffixes

    print(f"Found plot types for animation: {len(plot_suffixes)}")

    # Iterate over each plot type
    for suffix in plot_suffixes:
        print(f"\nCollecting frames for: {suffix}...")
        images = []

        for folder in folders:
            matching_files = [f for f in os.listdir(folder) if f.endswith(suffix)]

            if matching_files:
                file_path = os.path.join(folder, matching_files[0])
                images.append(imageio.imread(file_path))
            else:
                print(f"  Warning: File for {suffix} is missing in folder {os.path.basename(folder)}")

        if images:
            base_name = suffix.replace('.png', '')

            # Save in all selected formats and speeds
            for fps in SPEEDS:
                for fmt in FORMATS:
                    fmt = fmt.lower()
                    # Format the file name (e.g., Inertia_Tensor_Area_XX_hist_2fps.gif)
                    filename = f"{base_name}_{fps}fps.{fmt}"
                    # Save path in the corresponding format subdirectory
                    filepath = os.path.join(output_dir, fmt, filename)

                    # Save the file
                    if fmt == 'gif':
                        imageio.mimsave(filepath, images, fps=fps, loop=0)
                    else:
                        imageio.mimsave(filepath, images, fps=fps)

                    print(f"  Saved: {fmt}/{filename}")

    print("\nDone! All animations have been successfully created.")


if __name__ == "__main__":
    main()