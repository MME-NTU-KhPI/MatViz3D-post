import h5py
import numpy as np
import HDF5Operator
import matplotlib.pyplot as plt

ID = 0
X = 1
Y = 2
Z = 3
UX = 4
UY = 5
UZ = 6
SX = 7
SY = 8
SZ = 9
SXY = 10
SYZ = 11
SXZ = 12
EpsX = 13
EpsY = 14
EpsZ = 15
EpsXY = 16
EpsYZ = 17
EpsXZ = 18


def vector_to_matrix(vector):
    c_matrix = np.zeros([6, 6])
    k = 0
    for i in range(0, 6):
        for j in range(i, 6):
            c_matrix[j, i] = c_matrix[i, j] = vector[k]
            k += 1
    return c_matrix


def print_matrix(matrix, format_str=' .3e'):
    for row in matrix:
        print(' '.join([('{:' + format_str + '}').format(item) for item in row]))


def plt_matrix(matrix, format_str=': .3f', scale=1, filename=None):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.matshow(matrix, cmap=plt.get_cmap('coolwarm'))
    max_i, max_j = matrix.shape
    for i in range(max_i):
        for j in range(max_j):
            c = ("{" + format_str + "}").format(matrix[j, i] * scale)
            ax.text(i, j, c, va='center', ha='center')

    labs = ["XX", "YY", "ZZ", "XY", "YZ", "XZ"]
    ax.set_xticks(np.arange(max_i), labels=labs)
    ax.set_yticks(np.arange(max_j), labels=labs)

    if filename:
        fig.savefig(filename + ".png", dpi=300)
        fig.savefig(filename)


def get_A_b_from_data(data, inverse=False):
    stress = data[SX:SXZ + 1]
    strain = data[EpsX:EpsXZ + 1]

    if inverse:
        stress, strain = strain, stress

    C = np.zeros([6, 21], dtype=float)

    indexes = [
        [0, 1, 2, 3, 4, 5],
        [1, 6, 7, 8, 9, 10],
        [2, 7, 11, 12, 13, 14],
        [3, 8, 12, 15, 16, 17],
        [4, 9, 13, 16, 18, 19],
        [5, 10, 14, 17, 19, 20]]

    for i, idx in enumerate(indexes):
        C[i, idx] = strain

    return stress, C


# Solve system of linear equations using minimizing the residual algorithm (least squares)
def solve_linear_equations(A, b):
    x, residuals, rank, s = np.linalg.lstsq(A, b)
    print("Solving lstsq info:")
    print("\tResiduals: ", residuals)
    print("\tRank: ", rank)
    print("\tSingular values: ", s)
    print()
    return x


def process_dataset(hdf5, data_path):
    # Initial data
    A = np.empty([0, 21], dtype=float)
    b = np.array([], dtype=float)

    data_list = hdf5.list_datasets(data_path)

    for list_item in data_list:
        if list_item.startswith('ls_'):
            data = hdf5.read_data("/" + data_path + "/" + list_item + "/results_avg")
            lb, lA = get_A_b_from_data(data, inverse=True)
            b = np.append(lb, b, axis=0)
            A = np.append(lA, A, axis=0)

    # Solve linear equations
    x = solve_linear_equations(A, b)

    S_matrix = vector_to_matrix(x)

    print("S matrix, 1/TPa = ")
    print_matrix(S_matrix * 1e12, format_str=' >7.2f')
    print()

    C_matrix = np.linalg.inv(S_matrix)
    print("C matrix, GPa = ")
    print_matrix(C_matrix / 1e9, format_str=' >7.2f')

    print("C Eigen values, GPa =", np.linalg.eigvals(C_matrix) / 1e9)
    print()

    S_inv_diag = 1 / np.diag(S_matrix)
    print('Ex = {: .3e}'.format(S_inv_diag[0]))
    print('Ey = {: .3e}'.format(S_inv_diag[1]))
    print('Ez = {: .3e}'.format(S_inv_diag[2]))
    print()
    print('Gxy = {: .3e}'.format(S_inv_diag[3]))
    print('Gyz = {: .3e}'.format(S_inv_diag[4]))
    print('Gxz = {: .3e}'.format(S_inv_diag[5]))
    print()

    P_matrix = np.copy(S_matrix)
    for i in range(0, 6):
        P_matrix[i, :] *= -S_inv_diag

    print("Poisson's matrix")
    print_matrix(P_matrix, format_str=' .3f')

    print()

    hdf5.rewrite_data("/" + data_path + '/C_matrix', C_matrix)
    hdf5.rewrite_data("/" + data_path + '/S_matrix', S_matrix)
    hdf5.rewrite_data("/" + data_path + '/P_matrix', P_matrix)


def main():
    #path = R'd:\temp\1\build-MatViz3D-Desktop_Qt_6_3_1_MinGW_64_bit-Debug\\'
    path = "z:\\ans_proj\\matviz\\"
    filename = './result-25-5.hdf5'

    # Create an instance of the HDF5Operator class
    hdf5 = HDF5Operator.HDF5Operator(path + filename)
    dataset_list = hdf5.list_datasets('/')

    print("Open file ", path + filename)

    print("Loading data....")

    for ds in dataset_list:
        if ds.endswith('last_set'):
            continue
        print("Loading dataset ", ds)
        process_dataset(hdf5, "/" + ds)
    exit()


if __name__ == '__main__':
    main()
