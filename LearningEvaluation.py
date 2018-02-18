import matplotlib.pyplot as plt
import numpy as np

def get_test_set_efficiency(dataset, neural_network):
    test_data, test_labels = dataset.get_arrays_from_test_dataset()
    predicted_labels = neural_network.predict_labels(test_data)
    test_efficiency = (np.sum(test_labels == predicted_labels) / test_data.shape[0])

    return int(test_efficiency * 100)

def get_training_efficiency(dataset, neural_network, dataset_split):
    data, labels = dataset.get_arrays_from_training_dataset()
    training_data, validation_data = dataset.split_dataset_into_train_valid(data, 100 - dataset_split)
    training_labels, validation_labels = dataset.split_dataset_into_train_valid(labels, 100 - dataset_split)
    predicted_training_labels = neural_network.predict_labels(training_data)
    predicted_validation_labels = neural_network.predict_labels(validation_data)
    training_set_efficiency = (np.sum(training_labels == predicted_training_labels) / training_data.shape[0])
    validation_set_efficiency = (np.sum(validation_labels == predicted_validation_labels) / validation_data.shape[0])

    return int(training_set_efficiency * 100), int(validation_set_efficiency * 100)

def plot_convergence(neural_network):
    iterations = len(neural_network.cost_history)
    plt.plot(range(iterations), neural_network.cost_history)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    figure = plt.gcf()
    figure.canvas.set_window_title('Gradient convergence')
    plt.tight_layout()
    plt.show()

def show_sample_misclassified_images(dataset, neural_network, rows_number, columns_number):
    labels_names = dataset.get_labels_names()
    test_data, test_labels = dataset.get_arrays_from_test_dataset()

    predicted_labels = neural_network.predict_labels(test_data)
    misclassified_images = test_data[test_labels != predicted_labels][:(rows_number * columns_number)]
    misclassified_labels = predicted_labels[test_labels != predicted_labels][:(rows_number * columns_number)]
    correct_labels = test_labels[test_labels != predicted_labels][:(rows_number * columns_number)]

    figure, axes = plt.subplots(rows_number, columns_number, figsize=(columns_number + 2, rows_number + 2))
    axes = axes.flatten()

    for i in range(rows_number * columns_number):
        image = misclassified_images[i].reshape([3, 32, 32])
        image = image.transpose([1, 2, 0])
        correct_label = labels_names[correct_labels[i]]
        misclassified_label = labels_names[misclassified_labels[i]]
        axes[i].imshow(image)
        axes[i].set_title('True: %s\nPredicted: %s' % (correct_label, misclassified_label))
        axes[i].set_yticks([])
        axes[i].set_xticks([])

    figure.canvas.set_window_title('Sample misclassified images')
    plt.tight_layout()
    plt.show()
