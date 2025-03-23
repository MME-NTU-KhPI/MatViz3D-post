import h5py

class HDF5Operator:
    def __init__(self, file_path):
        self.file_path = file_path

    def write_data(self, dataset_name, data):
        with h5py.File(self.file_path, 'a') as f:
            f.create_dataset(dataset_name, data=data)

    def read_data(self, dataset_name):
        with h5py.File(self.file_path, 'r') as f:
            return f[dataset_name][:]

    def list_datasets(self, data_path):
        with h5py.File(self.file_path, 'r') as f:
            return list(f[data_path].keys())

    def is_exist(self, data_path):
        with h5py.File(self.file_path, 'r') as f:
            res = data_path in f
            return res

    def rewrite_data(self, data_path, data):
        if self.is_exist(data_path):
            with h5py.File(self.file_path, 'r+') as f:
                f[data_path][...] = data
        else:
            self.write_data(data_path, data)
