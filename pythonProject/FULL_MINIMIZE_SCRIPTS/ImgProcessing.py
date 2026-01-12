import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
from skimage import exposure
from skimage.io import imread
from skimage.morphology import opening, closing, square


def multiple_formatter(denominator=4, number=np.pi, latex='\\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$' % latex
            elif num == -1:
                return r'$-%s$' % latex
            else:
                return r'$%s%s$' % (num, latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$' % (latex, den)
            elif num == -1:
                return r'$\frac{-%s}{%s}$' % (latex, den)
            else:
                return r'$\frac{%s%s}{%s}$' % (num, latex, den)

    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=4, number=np.pi, latex='\\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


def rnd_cmap(n):
    if n < 2:
        n = 2
    n = int(n)
    colors = np.empty([n, 4])
    colors[1] = [1., 1., 1., 1.]  # white background
    colors[0] = [0., 0., 0., 1.]  # black background
    for i in range(2, n):
        colors[i] = [np.random.random_sample(), np.random.random_sample(), np.random.random_sample(), 1]

    cm = ListedColormap(colors, name='my_list')
    return cm


def plot_hist(data, bins, ax, i, yticks):
    if len(data) == 0:
        print(f"warning: not enough data points to fit lognormal distribution for {yticks[i]}")
        return

    na_lognparam = scipy.stats.lognorm.fit(data)
    data_mean, data_std = np.mean(data), np.std(data)

    num_bins = int(np.sqrt(len(data)))
    ax.hist(data, bins=num_bins, density=True, label=' ')
    na_x = np.linspace(np.min(data), np.max(data), 1000)
    ax.plot(na_x, scipy.stats.lognorm.pdf(na_x, *na_lognparam))
    s = r'$<E>={0:.3e}$'.format(data_mean)
    s += '\n'
    s += r'$\sqrt{\mathrm{var}[K_\sigma]}=' + '{0:.3e}$'.format(data_std)
    # plt.text(0.5, 0.8, "{0} {1}".format(*na_lognparam), transform=ax.transAxes, horizontalalignment='center',
    #          verticalalignment='center', fontsize=6)
    # plt.text(0.3, 0.88, s, transform=ax.transAxes, horizontalalignment='left', verticalalignment='center', fontsize=6)


def plot_hist_na(data, ax, i, yticks):
    if len(data) < 2:
        print(f"warning: not enough data points to plot KDE for {yticks[i]}")
        return

    hist, bin_edges = np.histogram(data)
    edg = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w = bin_edges[1] - bin_edges[0]

    kde = gaussian_kde(data)

    xdata = np.linspace(data.min(), data.max(), 300)
    kdeval = kde.evaluate(xdata)

    ax.plot(xdata, kdeval, label=yticks[i])
    ax.grid()


def vdk_perimeter(image):
    (w, h) = image.shape
    image = (image * 255).astype(np.uint8)
    data = np.zeros((w + 2, h + 2), dtype=image.dtype)
    data[1:-1, 1:-1] = image
    dilat = skimage.morphology.binary_dilation(data)
    newdata = dilat - data

    kernel = np.array([[10, 2, 10],
                       [2, 1, 2],
                       [10, 2, 10]])

    T = skimage.filters.edges.convolve(newdata, kernel, mode='constant', cval=0)

    cat_a = np.array([5, 15, 7, 25, 27, 17])
    cat_b = np.array([21, 33])
    cat_c = np.array([13, 23])
    cat_a_num = np.count_nonzero(np.isin(T, cat_a))
    cat_b_num = np.count_nonzero(np.isin(T, cat_b))
    cat_c_num = np.count_nonzero(np.isin(T, cat_c))

    perim = cat_a_num + cat_b_num * np.sqrt(2.) + cat_c_num * (1. + np.sqrt(2.)) / 2.

    return perim


def save_image_properties_to_csv(file_path, properties):
    directory = os.path.dirname(file_path)
    filename = os.path.splitext(os.path.basename(file_path))[0]
    csv_file_path = os.path.join(directory, f'image_properties_({filename}).csv')

    with open(csv_file_path, 'w') as csv_file:
        csv_file.write(
            'Area,Perimeter,Shape Factor,ECR,Orientation,Scale Factor,X Coordinate,Y Coordinate,'
            'Inertia Tensor XX,Inertia Tensor XY,Inertia Tensor YY,Determinant,Aspect Ratio,'
            'Compactness Ratio,area-to-ellipse Ratio,'
            'Inertia Tensor/Area XX,Inertia Tensor/Area XY,Inertia Tensor/Area YY\n')
        for i in range(len(properties['Area'])):
            inertia_tensor_xx = properties["Inertia Tensor"][i][0, 0] if i < len(properties["Inertia Tensor"]) else ''
            inertia_tensor_xy = properties["Inertia Tensor"][i][0, 1] if i < len(properties["Inertia Tensor"]) else ''
            inertia_tensor_yy = properties["Inertia Tensor"][i][1, 1] if i < len(properties["Inertia Tensor"]) else ''
            # inertia_tensor_xx = properties["Inertia Tensor"][i][0] if i < len(properties["Inertia Tensor"]) else ''
            # inertia_tensor_xy = properties["Inertia Tensor"][i][1] if i < len(properties["Inertia Tensor"]) else ''
            # determinant = np.linalg.det(properties["Inertia Tensor"][i]) if i < len(
            #     properties["Inertia Tensor"]) else ''
            determinant = 0
            inertia_tensor_area_xx = properties["Inertia Tensor/Area"][i][0, 0] if i < len(
                properties["Inertia Tensor/Area"]) else ''
            inertia_tensor_area_xy = properties["Inertia Tensor/Area"][i][0, 1] if i < len(
                properties["Inertia Tensor/Area"]) else ''
            inertia_tensor_area_yy = properties["Inertia Tensor/Area"][i][1, 1] if i < len(
                properties["Inertia Tensor/Area"]) else ''

            csv_file.write(f'{properties["Area"][i]},{properties["Perimeter"][i]},{properties["Shape Factor"][i]},'
                           f'{properties["ECR"][i]},{properties["Orientation"][i]},{properties["Scale Factor"][i]},'
                           f'{properties["X Coordinate"][i]},{properties["Y Coordinate"][i]},'
                           f'{inertia_tensor_xx},{inertia_tensor_xy},{inertia_tensor_yy},'
                           f'{determinant},{properties["Aspect Ratio"][i]},'
                           f'{properties["Compactness Ratio"][i]},{properties["area-to-ellipse Ratio"][i]},'
                           f'{inertia_tensor_area_xx},{inertia_tensor_area_xy},{inertia_tensor_area_yy}\n')


def main():
    path = r'D:\University\MatViz\1FULL_MINIMIZE_SCRIPTS\AZ31_iA'

    files = os.listdir(path)
    numfiles = len(files)
    # norm_area = np.zeros((numfiles, 700))
    # cs = np.zeros((numfiles, 700))
    # scale_factor = np.zeros((numfiles, 700))
    # orient = np.zeros((numfiles, 700))
    yticks = np.empty(numfiles, dtype='U25')
    i = 0

    fig1_na = plt.figure(dpi=100)
    fig1_cs = plt.figure(dpi=100)
    fig1_scf = plt.figure(dpi=100)
    fig1_ori = plt.figure(dpi=100)

    ax1_na = fig1_na.add_subplot(111, title="$N_a$")
    ax1_cs = fig1_cs.add_subplot(111, title="$C_s$")
    ax1_scf = fig1_scf.add_subplot(111, title="$S_c$")
    ax1_ori = fig1_ori.add_subplot(111, title=r"$\psi$")

    ax1_na.set_title('Histogram of Grain Areas')
    ax1_cs.set_title('Histogram of Grain Perimeters')
    ax1_scf.set_title('Histogram of Grain Scale Factors')
    ax1_ori.set_title('Histogram of Grain Orientations')

    ax1_na.grid(True)
    ax1_cs.grid(True)
    ax1_scf.grid(True)
    ax1_ori.grid(True)

    ax1_ori.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    ax1_ori.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax1_ori.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax1_ori.set_xlim(left=0, right=np.pi)

    ax1_scf.set_xlim(left=1, right=5)
    ax1_cs.set_xlim(left=0.1, right=0.8)
    ax1_na.set_xlim(left=0, right=0.02)

    for name in files:
        if name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(path, name)

            yticks[i] = name
            _norm_area, _per, _cs, _esr, _scf, _ori, x1, y1, _inertia_tensor, _determinant, bbox_area, convex_area, major_minor_area, inertia_tensors_area = process(
                image_path,
                plot_hist=True,
                plot_orientation=True)

            plot_hist_na(_norm_area, ax1_na, i, yticks)
            plot_hist(_per, None, ax1_cs, i, yticks)
            plot_hist(_scf, None, ax1_scf, i, yticks)
            plot_hist(_ori, None, ax1_ori, i, yticks)

            i += 1

            properties = {
                'Area': _norm_area,
                'Perimeter': _per,
                'Shape Factor': _cs,
                'ECR': _esr,
                'Orientation': _ori,
                'Scale Factor': _scf,
                'X Coordinate': x1,
                'Y Coordinate': y1,
                'Inertia Tensor': _inertia_tensor,
                'Aspect Ratio': bbox_area,
                'Compactness Ratio': convex_area,
                'area-to-ellipse Ratio': major_minor_area,
                'Inertia Tensor/Area': inertia_tensors_area
            }

            # Збереження властивостей у CSV файл
            save_image_properties_to_csv(image_path, properties)

    fig1_na.tight_layout()
    # fig1_na.legend()
    # fig1_na.savefig('2s_hist_kde_na.png', dpi=300)
    # fig1_na.savefig('2s_hist_kde_na.eps', dpi=300)
    fig1_na.show()

    fig1_cs.tight_layout()
    # fig1_cs.legend()
    # fig1_cs.savefig('2s_hist_kde_cs.png', dpi=300)
    # fig1_cs.savefig('2s_hist_kde_cs.eps', dpi=300)
    fig1_cs.show()

    fig1_scf.tight_layout()
    # fig1_scf.legend()
    # fig1_scf.savefig('2s_hist_kde_scf.png', dpi=300)
    # fig1_scf.savefig('2s_hist_kde_scf.eps', dpi=300)
    fig1_scf.show()

    fig1_ori.tight_layout()
    # fig1_ori.legend()
    # fig1_ori.savefig('2s_hist_kde_ori.png', dpi=300)
    # fig1_ori.savefig('2s_hist_kde_ori.eps', dpi=300)
    fig1_ori.show()

    plt.show()

    pass


def label2bright_rgb(label_img, seed=42):
    np.random.seed(seed)
    n_labels = np.max(label_img) + 1
    # Створюємо яскраві кольори: кожен компонент у межах [0.4, 1.0]
    colors = np.vstack([[0, 0, 0], 0.4 + 0.6 * np.random.rand(n_labels - 1, 3)])
    out = np.zeros((*label_img.shape, 3))
    for label in range(n_labels):
        mask = label_img == label
        out[mask] = colors[label]
    return out


def process(filename, plot_orientation=False, plot_hist=False):
    data = imread(filename, as_gray=True)
    # data = skimage.filters.sobel(data)
    # data[data < 1e-1] = 0
    # print(data, np.nonzero(data))
    print(filename)
    # print(data, data.shape)

    plt.figure()
    plt.title("data file")
    plt.imshow(data, cmap='gray')
    plt.show()

    # data = exposure.adjust_gamma(data, gamma=1.5)
    data = exposure.adjust_gamma(data, gamma=0.7)

    plt.figure()
    plt.title("data gamma")
    plt.imshow(data, cmap='gray')
    plt.show()

    data_l = skimage.filters.laplace(data)

    plt.figure()
    plt.title("data_l")
    plt.imshow(data_l, cmap='gray')
    plt.show()

    # data = data + data_l

    plt.figure()
    plt.title("data + data_l")
    plt.imshow(data + data_l, cmap='gray')
    plt.show()

    data = skimage.filters.sobel(data)

    plt.figure()
    plt.title("data")
    plt.imshow(data, cmap='gray')
    plt.show()

    data = ((data - np.min(data)) / (np.max(data) - np.min(data))) * 255
    trr = 20
    # trr = skimage.filters.threshold_otsu(data)
    print(trr)

    data[data < trr] = 0
    data[data >= trr] = 1

    plt.figure()
    plt.title("data trr")
    plt.imshow(data, cmap='gray')
    plt.show()

    data = data.astype('int')
    base = os.path.basename(filename)
    file_title = os.path.splitext(base)[0]

    data = data == 0
    # print(data)

    # Морфологічне відкриття та закриття для усунення шуму
    selem1 = square(3)
    data = opening(data, selem1)
    plt.figure()
    plt.title("After morphological opening")
    plt.imshow(data, cmap='gray')
    plt.show()

    selem2 = square(2)
    data = closing(data, selem2)
    plt.figure()
    plt.title("After closing")
    plt.imshow(data, cmap='gray')
    plt.show()

    label_img = skimage.measure.label(data)
    # print(label_img)
    # plt.close()
    # plt.close()
    # plt.close()
    # plt.close()
    # plt.close()

    colored = label2bright_rgb(label_img)
    plt.figure()
    plt.title("Processed image")
    plt.imshow(colored)
    plt.show()

    regions = skimage.measure.regionprops(label_img)
    print(label_img)
    # print(regions)
    # exit()
    # print(regions)
    # exit(0)

    area = np.zeros(len(regions))
    bbox = np.zeros(len(regions))
    convex = np.zeros(len(regions))
    major = np.zeros(len(regions))
    minor = np.zeros(len(regions))
    perimeter = np.zeros(len(regions))
    orient = np.zeros(len(regions))
    scale_factor = np.zeros(len(regions))
    x = np.zeros(len(regions))
    y = np.zeros(len(regions))
    inertia_tensor = np.zeros((len(regions), 2, 2))
    inertia_tensor_area = np.zeros((len(regions), 2, 2))
    # inertia_tensor = np.zeros((len(regions), 1, 2))
    determinant = np.zeros(len(regions))

    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()

    h, w = data.shape
    bwimage = data

    i = 0
    for props in regions:
        if props.area < 20 or props.coords.min(axis=0)[0] <= 1 or props.coords.max(axis=0)[0] >= h - 2 or \
                props.coords.min(axis=0)[1] <= 1 or props.coords.max(axis=0)[1] >= w - 2:
            # Пропустити об'єкти, які дотикаються країв (видалити зерна, розташовані на краях)
            continue

        if props.area < 20:
            continue

        # if props.centroid is None or len(props.centroid) != 2:
        #     print("warning: in {} area of cell {} centroid is not defined or has unexpected values".format(filename, i))
        #     continue

        y0, x0 = props.centroid
        x[i], y[i] = x0, y0

        area[i] = props.area
        bbox[i] = props.area_bbox
        convex[i] = props.area_convex
        major[i] = props.major_axis_length
        minor[i] = props.minor_axis_length
        perimeter[i] = np.max([props.perimeter, vdk_perimeter(props.convex_image)])
        orient[i] = props.orientation
        if props.minor_axis_length:
            scale_factor[i] = props.major_axis_length / props.minor_axis_length
        else:
            scale_factor[i] = np.inf
            print("warning: scale factor for {} cell is infinity".format(i))

        # Оновлення обчислення тензора інерції та визначника
        coords = props.coords - np.array([[y0, x0]])
        # inertia_tensor[i] = np.cov(coords, rowvar=False)
        inertia_tensor[i] = props.inertia_tensor
        # inertia_tensor[i] = props.inertia_tensor_eigvals
        # determinant[i] = np.linalg.det(inertia_tensor[i])
        determinant[i] = 0

        if plot_orientation:
            x1 = x0 + np.cos(orient[i]) * 0.5 * props.major_axis_length
            y1 = y0 - np.sin(orient[i]) * 0.5 * props.major_axis_length
            x2 = x0 - np.sin(orient[i]) * 0.5 * props.minor_axis_length
            y2 = y0 - np.cos(orient[i]) * 0.5 * props.minor_axis_length

            ax.plot((x0, x1), (y0, y1), '-r', linewidth=0.5)
            ax.plot((x0, x2), (y0, y2), '-g', linewidth=0.5)
            ax.plot(x0, y0, '.r', markersize=1)

        cs_i = 4. * np.pi * area[i] / (perimeter[i] ** 2)

        if cs_i > 10:
            print("warning: Cs for {} cell > 1".format(i))
            plt.close()
            print(i, cs_i, props._slice, area[i], perimeter[i], vdk_perimeter(props.convex_image))
            plt.imshow(props.convex_image.astype(np.uint8))
            plt.show()

        i += 1

    ax.axis((0, h, w, 0))
    ax.imshow(bwimage, interpolation='none', cmap=rnd_cmap(np.max(bwimage)), origin="lower")

    if plot_hist:
        divider = make_axes_locatable(ax)
        axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
        axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

        axHistx.xaxis.set_tick_params(labelbottom=False)
        axHisty.yaxis.set_tick_params(labelleft=False)

        bins = int(np.sqrt(i))

        if bins <= 0:
            bins = 10  # Set a default positive value if bins is not positive

        axHistx.hist(x, edgecolor="black", bins=bins)
        axHisty.hist(y, orientation='horizontal', edgecolor="black", bins=bins)

    fig.tight_layout()
    # fig.savefig("./models2/" + file_title + '_micro.png', dpi=300)

    cond = np.logical_and(area > 10, scale_factor < np.inf)
    area = area[cond]
    perimeter = perimeter[cond]
    orient = orient[cond]
    scale_factor = scale_factor[cond]
    inertia_tensor = inertia_tensor[cond]
    determinant = determinant[cond]
    bbox = bbox[cond]
    convex = convex[cond]
    major = major[cond]
    minor = minor[cond]

    print('Image size =', data.shape)
    print('Number of pixels =', w * h)
    print('Num of detected grains = ', area.size)
    print('Grains per square pixel =', float(i) / w / h)

    norm_area = area / w / h
    shape_factor = 4. * np.pi * area / (perimeter ** 2)  # параметр відносного обсягу (коефіцієнт форми)
    esr = np.sqrt(norm_area / np.pi)
    bbox_area = bbox / area
    convex_area = convex / area
    major_minor_area = area / (major * minor)
    inertia_tensors_area = np.zeros_like(inertia_tensor)
    for i in range(len(area)):
        inertia_tensors_area[i] = inertia_tensor[i] / area[i]

    return norm_area, perimeter, shape_factor, esr, scale_factor, orient, x, y, inertia_tensor, determinant, bbox_area, convex_area, major_minor_area, inertia_tensors_area


if __name__ == "__main__":
    main()
