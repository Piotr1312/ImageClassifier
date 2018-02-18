from Dataset import Dataset
from NeuralNetwork import NeuralNetwork

dataset = Dataset(dataset_path = 'dataset/')
data, labels = dataset.get_arrays_from_training_dataset()
training_data, validation_data = dataset.split_dataset_into_train_valid(data, 80)
training_labels, validation_labels = dataset.split_dataset_into_train_valid(labels, 80)

neural_network = NeuralNetwork(input_units = 3072,
                               hidden_units = 500,
                               output_units = 10,
                               learning_rate = 0.0002,
                               regularization_parameter = 0.1,
                               iterations = 1500,
                               minibatches = 400,
                               )

neural_network.learn(training_data,
                     training_labels,
                     validation_data,
                     validation_labels)

neural_network.save_weights_to_file('appdata/Weights')
