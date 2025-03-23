import numpy as np
import matplotlib.pylab as plt
from matplotlib import colors
import pandas as pd
import HDF5Operator
import mv3d_post as mv3d
from scipy.stats import norm
import scipy.stats as stats
import argparse
import os

def process_dataset(hdf5, data_path):
    S_matrix = hdf5.read_data(data_path + "/S_matrix")
    #print(S_matrix)
    S_diag = np.diag(S_matrix)
    modulus = 1 / S_diag
    #print(modulus)
    P_matrix = np.copy(S_matrix)
    for i in range(0, 6):
        P_matrix[i, :] *= -modulus

    #print(np.ravel(P_matrix))
    poissions = np.ravel(P_matrix)
    return modulus, poissions
    pass



def plot_hist_set(df, nrows=1, ncol=1, units="", fontsize = 0, hspace=0.2, wspace=0.2, filename=""):
    # Plot histogram for each column
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=(ncol*1.45*1.5 , nrows*1.45 ))
    # creating a dictionary
    
    for i, column in enumerate(df.columns):
        #print(df[column])
        ax = np.ravel(axes)[i]

        # Set tick font size
        if fontsize:
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(fontsize)
        
        latex_column = column.replace('_', '_{') + '}'
        s = r'\left\langle{{ {0} }}\right\rangle = {1:.3f}\:\mathrm{{{2}}}'.format(latex_column, np.mean(df[column]), units)
        s += r';\quad'
        s += r'\sqrt{{\mathrm{{var}}\left[{0}\right]}}={1:.3f}\:\mathrm{{{2}}}'.format(latex_column, np.std(df[column]), units)
        s = '$' + s + '$'

        coef_of_var =  np.std(df[column]) / df[column].mean()
        if (np.abs(coef_of_var) < 1e-4):
            ax.set_axis_off() # Hide all axis
            continue

        # Plot histogram for the current column
        n_bins = int(np.sqrt(len(df[column]))) - 1
        
        N, bins, patches = ax.hist(df[column], bins=n_bins, density=True, alpha=0.8)

        # We'll color code by height, but you could use any scalar
        fracs = N / N.max()

        # we need to normalize the data to 0..1 for the full range of the colormap
        normalized = colors.Normalize(fracs.min(), fracs.max())

        # Now, we'll loop through our objects and set the color of each accordingly
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(normalized(thisfrac))
            thispatch.set_facecolor(color)

        
        # Fit a normal distribution to the data
        mu, std = norm.fit(df[column])  # Fit a normal distribution to the data
        
        # Plot the PDF (fitted distribution) on top of the histogram
        xmin, xmax = ax.get_xlim()  # Get the x-axis limits
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)  # Calculate the PDF
        ax.plot(x, p, 'k', linewidth=1)  # Plot the PDF
        
        if fontsize == 0:
            fontsize = plt.rcParams['axes.titlesize']
        # Set titles and labels
        ax.set_title(s, fontsize=fontsize)
        ax.set_xlabel("$"+latex_column+"$", fontsize=fontsize)
        ax.set_ylabel('Density', fontsize=fontsize)

    #plt.tight_layout()
    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    if filename:
        fig.savefig(filename + ".png", dpi=300)
        fig.savefig(filename)

    #plt.show()
    pass

def save_descriptive_statistics(df, filename):
    """
    Save descriptive statistics for each column of a pandas DataFrame to a file.
    Statistics include mean, std, coefficient of variation, 3rd moment, and 4th moment.
    """
    stats_df = pd.DataFrame(columns=['mean', 'std', 'coef_of_var', 'moment_3', 'moment_4'])
    
    for column in df.columns:
        mean = df[column].mean()
        std = df[column].std()
        coef_of_var = std / mean if mean != 0 else float('inf')
        if np.abs(coef_of_var) < 1e-4:
            moment_3 = moment_4 = 0
        else:
            moment_3 = stats.moment(df[column], moment=3)
            moment_4 = stats.moment(df[column], moment=4)
        
        stats_df.loc[column] = [mean, std, coef_of_var, moment_3, moment_4]
    
    stats_df.to_csv(filename, index_label='column')

def calculate_column_means(df):
    """
    Calculate the mean of each column in a pandas DataFrame.
    Returns a numpy array of means.
    """
    return df.mean().to_numpy()

def main():

    # Print the current working directory
    print(f"Current working directory: {os.getcwd()}")
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process HDF5 dataset.")
    parser.add_argument('filename', type=str, help='Path to the HDF5 file')
    args = parser.parse_args()

    filename = args.filename

    

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' does not exist.")
        exit(1)

    # Create an instance of the HDF5Operator class
    hdf5 = HDF5Operator.HDF5Operator(filename)
    dataset_list = hdf5.list_datasets('/')

    print("Open file ", filename)

    print("Loading data....")

    df_mudulus = pd.DataFrame(columns=['E_xx', 'E_yy', 'E_zz', 'G_xy', 'G_yz', 'G_xz'])
    df_poissons = pd.DataFrame(columns=[
        'nu_xx', 'nu_xy', 'nu_xz', 'nu_xyxx', 'nu_yzxx', 'nu_xzxx',
        'nu_yx', 'nu_yy', 'nu_yz', 'nu_xyyy', 'nu_yzyy', 'nu_xzyy',
        'nu_zx', 'nu_zy', 'nu_zz', 'nu_xyzz', 'nu_yzzz', 'nu_xzzz',
        'nu_xxxy', 'nu_yyxy', 'nu_zzxy', 'nu_xyxy', 'nu_yzxy', 'nu_xzxy',
        'nu_xxyz', 'nu_yyyz', 'nu_zzyz', 'nu_xyyz', 'nu_yzyz', 'nu_xzyz',
        'nu_xxxz', 'nu_yyxz', 'nu_zzxz', 'nu_xyxz', 'nu_yzxz', 'nu_xzxz',
        ])

    df_mudulus.index.name = df_poissons.index.name = 'id'

    for ds in dataset_list:
        if ds.endswith('last_set'):
            continue
        print("Loading dataset ", ds)
        modulus, poissions = process_dataset(hdf5, "/" + ds)
        df_mudulus.loc[len(df_mudulus)] = modulus
        df_poissons.loc[len(df_poissons)] = poissions

    df_mudulus.to_csv(filename+'_modulus.csv')
    df_poissons.to_csv(filename+'_poissons.csv')

    save_descriptive_statistics(df_mudulus / 1e9, filename + '_modulus_stats.csv')
    save_descriptive_statistics(df_poissons, filename + '_poissons_stats.csv')

    column_means_modulus = calculate_column_means(df_mudulus / 1e9)
    column_means_poissons = calculate_column_means(df_poissons)
    
    mv3d.plt_matrix(column_means_poissons.reshape((6,6)), format_str=': .3f', filename=filename + '_poissons_mean.eps')

    # plot_hist_set(df_mudulus / 1e9, nrows=2, ncol=3, units="GPa",
    #                wspace=0.5, hspace=0.6, fontsize=6, filename=filename+'modlulus.eps')
    # plot_hist_set(df_poissons, nrows=6, ncol=6, fontsize=6, 
    #               hspace=0.7, wspace=0.8, filename=filename+'poissons.eps')





    exit()


if __name__ == '__main__':
    main()
