import os
import pandas as pd
import matplotlib.pyplot as plt


# Function to normalize a column of the DataFrame to the range [0, 1]
def normalize_column(df, col):
    min_val, max_val = df[col].min(), df[col].max()
    # If column has variation, normalize it
    if min_val != max_val:
        # If values are already within [0, 1], do not normalize
        if min_val >= 0 and max_val <= 1:
            return df[col]
        return (df[col] - min_val) / (max_val - min_val)
    return 0.5  # Return a neutral value if no variation in the column


# Function to normalize x0 values based on the data in the DataFrame
def normalize_x0(x0, df):
    normalized_x0 = []
    for idx, (col, val) in enumerate(zip(df.columns, x0)):
        min_val, max_val = df[col].min(), df[col].max()
        if min_val != max_val:
            # If values are already within [0, 1], do not normalize
            if min_val >= 0 and max_val <= 1:
                normalized_val = val
            else:
                normalized_val = (val - min_val) / (max_val - min_val)
        else:
            normalized_val = 0.5  # Return neutral value if no variation

        # Ensure the value stays in the [0, 1] range
        normalized_val = max(0, min(1, normalized_val))

        normalized_x0.append(normalized_val)

    # Ensure repeated values stay the same after normalization
    for i in range(1, len(normalized_x0)):
        if x0[i] == x0[i - 1]:
            normalized_x0[i] = normalized_x0[i - 1]

    return normalized_x0


# Main function to plot optimization progress
def plot_optimization(csv_path, output_dir, x0=None):
    """
    Plots the parameter changes during optimization and saves the plot as a PNG file.

    :param csv_path: Path to the CSV file containing optimization log data.
    :param output_dir: Directory where the plot will be saved.
    :param x0: List of initial values for the optimization parameters (optional).
    """
    # Extract file name for the plot output
    file_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_filename = os.path.join(output_dir, f"plot_line({file_name}).png")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Exclude unnecessary columns from the DataFrame
    columns_to_exclude = ["fun", "mean_ecr", "relative_error(%)"]
    df = df[[col for col in df.columns if col not in columns_to_exclude]]

    # Normalize the data (except for x0 values)
    df_normalized = df.copy()
    for col in df.columns:
        df_normalized[col] = normalize_column(df, col)

    # Plot the normalized values for each column
    plt.figure(figsize=(13, 8))
    for column in df_normalized.columns:
        plt.plot(df.index + 1, df_normalized[column], label=column)

    # If x0 values are provided, normalize and plot them
    if x0 is not None:
        normalized_x0 = normalize_x0(x0, df)
        last_iteration = df.index.max() + 1

        shift_x0 = []
        seen_values = {}
        for idx, val in enumerate(normalized_x0):
            if val in seen_values:
                shift_x0.append(last_iteration + 0.5 * (seen_values[val] + 1))
                seen_values[val] += 1
            else:
                shift_x0.append(last_iteration)
                seen_values[val] = 0

        # Plot the shifted x0 values as points
        for idx, (col, val) in enumerate(zip(df.columns, shift_x0)):
            plt.scatter(val, normalized_x0[idx], marker='o', s=50, label=f"{col} (x0)")

    # Customize the plot labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Parameter Value')
    plt.title('Parameter Changes During Optimization')
    plt.legend(loc='upper left', fontsize=7, markerscale=0.3)
    plt.grid(True)
    plt.tight_layout()

    # Set plot limits
    plt.ylim(0, 1.05)
    plt.xlim(df.index.min() + 1, df.index.max() + 5)

    # Save the plot as a PNG file
    plt.savefig(output_filename)
    plt.show()

    print(f"Plot saved at: {output_filename}")
    plt.close()


# Main entry point for the script
def main():
    """
    Main function to run the script. Modify the 'csv_path' and 'x0' values as needed.
    """
    csv_path = "optimization_log_Nelder-Mead.csv"  # Path to the CSV file with optimization log
    output_dir = "."  # Directory where the plot will be saved (current directory by default)
    x0 = [0.1, 1, 1.5, 1.5, 1.5, 90, 90, 90]  # Example initial parameter values (x0)
    plot_optimization(csv_path, output_dir, x0)


# Ensure the script runs only when executed directly (not when imported)
if __name__ == "__main__":
    main()
