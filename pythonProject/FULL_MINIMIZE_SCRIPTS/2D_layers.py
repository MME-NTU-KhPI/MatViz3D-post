import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# --- ANALYSIS SETTINGS ---
ANALYSIS_MODE = 'sample'  # 'all', 'sample', 'single'
SAMPLE_SKIP = 30
SAMPLE_OFFSET = 15
SINGLE_LAYER_INDEX = 0

SELECTED_AXES = ['x']  # 'x', 'y', 'z'

# --- 3D SETTINGS ---
ENABLE_3D_STACK_VISUALIZATION = True
SAVE_INDIVIDUAL_2D_SLICES = True
SLICE_THICKNESS = 1

# --- EXPORT SETTINGS ---
FORMATS_3D = ['png']  # Formats for 3D stack (e.g., ['png', 'svg', 'pdf'])
FORMATS_2D = ['png']  # Formats for 2D slices
DPI_SETTING = 600     # Resolution (300 is good, 600 is excellent for print)

# --- CAMERA ANGLE SETTINGS ---
# Dictionary with angles (elev, azim). You can add your own custom views!
CAMERA_ANGLES = {
    'default': (30, -60),        # Default matplotlib view
    'isometric': (35.264, 45),   # Isometric projection (all 3 axes at the same angle)
    'dimetric': (20, 60),        # Dimetric projection
    'trimetric': (25, -50),      # Trimetric projection
    'top_down': (90, -90),       # Top-down view (Plan)
    'front': (0, -90),           # Front view
    'side': (0, 0)               # Side view
}

# Specify which views you want to render and save (you can list multiple)
VIEWS_TO_RENDER = ['isometric', 'dimetric', 'default']
# ----------------------------

def load_volume(file_path):
    print(f"Reading file: {file_path}...")
    try:
        data = np.genfromtxt(file_path, delimiter=';', skip_header=1, dtype=int)
    except Exception as e:
        print(f"Error opening file: {e}")
        return None

    if data.size == 0:
        print("File is empty.")
        return None

    max_coords = np.max(data[:, :3], axis=0)
    cube_size_x, cube_size_y, cube_size_z = max_coords + 1

    print(f"Detected cube dimensions: {cube_size_x}x{cube_size_y}x{cube_size_z}")

    volume = np.zeros((cube_size_x, cube_size_y, cube_size_z), dtype=int)
    volume[data[:, 0], data[:, 1], data[:, 2]] = data[:, 3]

    return volume


def create_random_colormap(n_colors):
    hues = np.random.rand(n_colors)
    sats = np.random.uniform(0.6, 1.0, n_colors)
    vals = np.random.uniform(0.85, 1.0, n_colors)

    hsv_colors = np.column_stack((hues, sats, vals))
    rgb_colors = mcolors.hsv_to_rgb(hsv_colors)

    np.random.shuffle(rgb_colors)
    return ListedColormap(rgb_colors)


def visualize_3d_stack(volume, indices, axis, cmap, norm, output_dir):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    cube_size_x, cube_size_y, cube_size_z = volume.shape

    print(f"  -> Fast generation of 3D surfaces with shadows for the '{axis}' axis...")

    SHADE_ON = True
    AA_ON = False

    for i in indices:
        if axis == 'x':
            t_max = min(SLICE_THICKNESS, cube_size_x - i)
            if t_max <= 0: continue

            layer2d = volume[i, :, :]
            color_layer = cmap(norm(layer2d))
            NY, NZ = cube_size_y, cube_size_z

            yy, zz = np.meshgrid(np.arange(NY + 1), np.arange(NZ + 1), indexing='ij')

            # Front face
            xx = np.full_like(yy, i)
            ax.plot_surface(xx, yy, zz, facecolors=color_layer, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)
            # Back face
            xx = np.full_like(yy, i + t_max)
            ax.plot_surface(xx, yy, zz, facecolors=color_layer, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)

            # Bottom face
            xx, yy = np.meshgrid(np.array([i, i + t_max]), np.arange(NY + 1), indexing='ij')
            zz = np.full_like(xx, 0)
            colors = np.zeros((1, NY, 4))
            colors[0, :, :] = color_layer[:, 0, :]
            ax.plot_surface(xx, yy, zz, facecolors=colors, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)
            # Top face
            zz = np.full_like(xx, NZ)
            colors[0, :, :] = color_layer[:, -1, :]
            ax.plot_surface(xx, yy, zz, facecolors=colors, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)

            # Left face
            xx, zz = np.meshgrid(np.array([i, i + t_max]), np.arange(NZ + 1), indexing='ij')
            yy = np.full_like(xx, 0)
            colors = np.zeros((1, NZ, 4))
            colors[0, :, :] = color_layer[0, :, :]
            ax.plot_surface(xx, yy, zz, facecolors=colors, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)
            # Right face
            yy = np.full_like(xx, NY)
            colors[0, :, :] = color_layer[-1, :, :]
            ax.plot_surface(xx, yy, zz, facecolors=colors, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)

        elif axis == 'y':
            t_max = min(SLICE_THICKNESS, cube_size_y - i)
            if t_max <= 0: continue

            layer2d = volume[:, i, :]
            color_layer = cmap(norm(layer2d))
            NX, NZ = cube_size_x, cube_size_z

            xx, zz = np.meshgrid(np.arange(NX + 1), np.arange(NZ + 1), indexing='ij')

            yy = np.full_like(xx, i)
            ax.plot_surface(xx, yy, zz, facecolors=color_layer, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)
            yy = np.full_like(xx, i + t_max)
            ax.plot_surface(xx, yy, zz, facecolors=color_layer, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)

            yy, xx = np.meshgrid(np.array([i, i + t_max]), np.arange(NX + 1), indexing='ij')
            zz = np.full_like(yy, 0)
            colors = np.zeros((1, NX, 4))
            colors[0, :, :] = color_layer[:, 0, :]
            ax.plot_surface(xx, yy, zz, facecolors=colors, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)
            zz = np.full_like(yy, NZ)
            colors[0, :, :] = color_layer[:, -1, :]
            ax.plot_surface(xx, yy, zz, facecolors=colors, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)

            yy, zz = np.meshgrid(np.array([i, i + t_max]), np.arange(NZ + 1), indexing='ij')
            xx = np.full_like(yy, 0)
            colors = np.zeros((1, NZ, 4))
            colors[0, :, :] = color_layer[0, :, :]
            ax.plot_surface(xx, yy, zz, facecolors=colors, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)
            xx = np.full_like(yy, NX)
            colors[0, :, :] = color_layer[-1, :, :]
            ax.plot_surface(xx, yy, zz, facecolors=colors, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)

        elif axis == 'z':
            t_max = min(SLICE_THICKNESS, cube_size_z - i)
            if t_max <= 0: continue

            layer2d = volume[:, :, i]
            color_layer = cmap(norm(layer2d))
            NX, NY = cube_size_x, cube_size_y

            xx, yy = np.meshgrid(np.arange(NX + 1), np.arange(NY + 1), indexing='ij')

            zz = np.full_like(xx, i)
            ax.plot_surface(xx, yy, zz, facecolors=color_layer, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)
            zz = np.full_like(xx, i + t_max)
            ax.plot_surface(xx, yy, zz, facecolors=color_layer, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)

            zz, xx = np.meshgrid(np.array([i, i + t_max]), np.arange(NX + 1), indexing='ij')
            yy = np.full_like(zz, 0)
            colors = np.zeros((1, NX, 4))
            colors[0, :, :] = color_layer[:, 0, :]
            ax.plot_surface(xx, yy, zz, facecolors=colors, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)
            yy = np.full_like(zz, NY)
            colors[0, :, :] = color_layer[:, -1, :]
            ax.plot_surface(xx, yy, zz, facecolors=colors, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)

            zz, yy = np.meshgrid(np.array([i, i + t_max]), np.arange(NY + 1), indexing='ij')
            xx = np.full_like(zz, 0)
            colors = np.zeros((1, NY, 4))
            colors[0, :, :] = color_layer[0, :, :]
            ax.plot_surface(xx, yy, zz, facecolors=colors, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)
            xx = np.full_like(zz, NX)
            colors[0, :, :] = color_layer[-1, :, :]
            ax.plot_surface(xx, yy, zz, facecolors=colors, shade=SHADE_ON, antialiased=AA_ON, rstride=1, cstride=1)

    ax.set_xlim([0, cube_size_x])
    ax.set_ylim([0, cube_size_y])
    ax.set_zlim([0, cube_size_z])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Loop through selected camera angles
    for view_name in VIEWS_TO_RENDER:
        if view_name in CAMERA_ANGLES:
            elev, azim = CAMERA_ANGLES[view_name]
            ax.view_init(elev=elev, azim=azim)
        else:
            print(f"  WARNING: View '{view_name}' not found. Using default.")
            ax.view_init(elev=30, azim=-60)

        # Save the image for the current angle in all specified formats
        for fmt in FORMATS_3D:
            filename = f"3D_stack_{axis}_{view_name}.{fmt}"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=DPI_SETTING, bbox_inches='tight', format=fmt)
            print(f"  -> Saved ({view_name}): {filename}")

    plt.close()


def visualize_slices(volume, mode=ANALYSIS_MODE, sample_skip=SAMPLE_SKIP, sample_offset=SAMPLE_OFFSET,
                     single_index=SINGLE_LAYER_INDEX, selected_axes=SELECTED_AXES, output_dir=None):
    os.makedirs(output_dir, exist_ok=True)

    cube_size = volume.shape[0]

    if mode == 'all':
        indices = list(range(cube_size))
    elif mode == 'sample':
        indices = list(range(sample_offset, cube_size, sample_skip + 1))
    elif mode == 'single':
        indices = [single_index]

    max_id = np.max(volume)
    norm = mcolors.Normalize(vmin=0, vmax=max_id)
    cmap = create_random_colormap(max_id + 1)

    print(f"Starting visualization: Mode={mode}, Axes={selected_axes}, Indices count={len(indices)}")
    count = 0

    for axis in selected_axes:
        if axis == 'x': axis_limit = volume.shape[0]
        elif axis == 'y': axis_limit = volume.shape[1]
        elif axis == 'z': axis_limit = volume.shape[2]
        else: continue

        if ENABLE_3D_STACK_VISUALIZATION:
            visualize_3d_stack(volume, indices, axis, cmap, norm, output_dir)

        if SAVE_INDIVIDUAL_2D_SLICES:
            for i in indices:
                if i >= axis_limit: continue

                if axis == 'x':
                    layer = volume[i, :, :]
                    xlabel, ylabel = 'Y', 'Z'
                elif axis == 'y':
                    layer = volume[:, i, :]
                    xlabel, ylabel = 'X', 'Z'
                elif axis == 'z':
                    layer = volume[:, :, i]
                    xlabel, ylabel = 'X', 'Y'

                plt.figure(figsize=(10, 10))
                plt.imshow(layer.T, cmap=cmap, norm=norm, interpolation='nearest', origin='lower')

                plt.title(f"Axis: {axis.upper()}, Layer: {i}")
                plt.xlabel(f"{xlabel} axis")
                plt.ylabel(f"{ylabel} axis")
                plt.tight_layout()

                for fmt in FORMATS_2D:
                    filename = f"slice_{axis}_{i:04d}.{fmt}"
                    save_path = os.path.join(output_dir, filename)
                    plt.savefig(save_path, dpi=DPI_SETTING, bbox_inches='tight', format=fmt)

                plt.close()
                count += 1

    if SAVE_INDIVIDUAL_2D_SLICES:
        print(f"Done! Saved {count} individual 2D images (in formats {FORMATS_2D}) to the folder '{output_dir}'.")


def process_all_csv_in_folder(root_folder):
    print(f"Starting folder traversal: {root_folder}")
    for dirpath, dirnames, filenames in os.walk(root_folder):
        csv_files = [f for f in filenames if f.lower().endswith(".csv")]
        for csv_file in csv_files:
            csv_path = os.path.join(dirpath, csv_file)
            print(f"\nFound CSV: {csv_path}")

            volume = load_volume(csv_path)
            if volume is None: continue

            csv_name = os.path.splitext(csv_file)[0]
            output_dir = os.path.join(dirpath, f"{csv_name}_slices")

            visualize_slices(
                volume,
                mode=ANALYSIS_MODE,
                sample_skip=SAMPLE_SKIP,
                sample_offset=SAMPLE_OFFSET,
                single_index=SINGLE_LAYER_INDEX,
                selected_axes=SELECTED_AXES,
                output_dir=output_dir
            )

if __name__ == "__main__":
    ROOT_FOLDER = r"D:\Project(MatViz3D)\Random\Test"
    if not os.path.exists(ROOT_FOLDER):
        print(f"WARNING: Folder '{ROOT_FOLDER}' not found.")
    else:
        process_all_csv_in_folder(ROOT_FOLDER)
