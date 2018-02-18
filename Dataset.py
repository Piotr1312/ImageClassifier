import numpy as np
import matplotlib.pyplot as plt
import pickle

class Dataset:
    def __init__(self, dataset_path = 'dataset/'):
        self.dataset_path = dataset_path

    def _load_from_file(self, name):
        with open(name, 'rb') as file:
            data = pickle.load(file, encoding = 'bytes')
        return data

    def _get_data_from_dict(self, name):
        return np.array(name[b'data'] / 255)

    def _get_labels_from_dict(self, name):
        return np.array(name[b'labels'])

    def get_arrays_from_training_dataset(self):
        data_array = []
        labels_array = []

        for i in range(1,6):
            dataset = self._load_from_file('%s/data_batch_%s' % (self.dataset_path, i))
            data = self._get_data_from_dict(dataset)
            labels = self._get_labels_from_dict(dataset)
            data_array.append(data)
            labels_array.append(labels)

        data_array = np.concatenate(data_array)
        labels_array = np.concatenate(labels_array)

        return data_array, labels_array

    def get_arrays_from_test_dataset(self):
        dataset = self._load_from_file('%s/test_batch' % self.dataset_path)
        data_array = self._get_data_from_dict(dataset)
        labels_array = self._get_labels_from_dict(dataset)

        return data_array, labels_array

    def split_dataset_into_train_valid(self, dataset, training_percent):
        split_number = int(training_percent / 100 * 50000)
        training_dataset = dataset[:split_number]
        validation_dataset = dataset[split_number:]

        return training_dataset, validation_dataset

    def get_labels_names(self):
        return np.array(['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

    def show_sample_images(self, rows_number, columns_number):
        labels_names = self.get_labels_names()
        data, labels = self.get_arrays_from_training_dataset()

        figure, axes = plt.subplots(rows_number, columns_number, figsize=(columns_number, rows_number))
        axes = axes.flatten()

        for i in range(rows_number*columns_number):
            image = data[i].reshape([3, 32, 32])
            image = image.transpose([1, 2, 0])
            label_name = labels_names[labels[i]]
            axes[i].imshow(image)
            axes[i].set_title(label_name)
            axes[i].set_yticks([])
            axes[i].set_xticks([])

        figure.canvas.set_window_title('Sample images')
        plt.tight_layout()
        plt.show()
