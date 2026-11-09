
import numpy as np
import pickle
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.dataset_name = dName

    def load(self):
        print('loading data...')

        with open(self.dataset_source_folder_path +
                  self.dataset_source_file_name, 'rb') as f:
            data = pickle.load(f)

        result = {}

        for split in ['train', 'test']:
            X = []
            y = []

            for instance in data[split]:
                X.append(instance['image'])
                y.append(instance['label'])

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.int64)

            if self.dataset_name == "MNIST":
                X = X[:, np.newaxis, :, :]

            elif self.dataset_name == "ORL":
                X = X[:, :, :, 0]
                X = X[:, np.newaxis, :, :]
                y = y - 1

            elif self.dataset_name == "CIFAR":
                X = X.transpose(0, 3, 1, 2)

            X = X / 255.0

            result[split] = {'X': X, 'y': y}

            print(split, X.shape, y.shape)

        return result
