import numpy as np
import sys
import time

class NeuralNetwork:
    def __init__(self,
                 input_units = 3072,
                 hidden_units = 500,
                 output_units = 10,
                 learning_rate = 0.0002,
                 regularization_parameter = 0.1,
                 iterations = 1500,
                 minibatches = 400,
                 ):

        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.learning_rate = learning_rate
        self.regularization_parameter = regularization_parameter
        self.iterations = iterations
        self.minibatches = minibatches

    def _encode_labels_onehot(self, labels):
        onehot_encoded_labels = np.zeros((labels.shape[0], self.output_units))

        for index in range(labels.shape[0]):
            value = labels[index]
            onehot_encoded_labels[index, value] = 1

        return onehot_encoded_labels

    def _calculate_sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _calculate_cost(self, onehot_encoded_labels, network_output):
        regularization_factor = self.regularization_parameter * (np.sum(self.weights_input_hidden ** 2) + np.sum(self.weights_hidden_output ** 2))
        cost = -np.sum(onehot_encoded_labels * (np.log(network_output)) + (1 - onehot_encoded_labels) * np.log(1 - network_output)) + regularization_factor

        return cost

    def predict_labels(self, features):
        network_output = self._propagate_forward(features)[1]
        predicted_labels = np.argmax(network_output, axis = 1)

        return predicted_labels

    def _initialize_weights(self):
        self.bias_input_hidden = np.zeros(self.hidden_units)
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_units, self.hidden_units))
        self.bias_hidden_output = np.zeros(self.output_units)
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_units, self.output_units))

    def _propagate_forward(self, features):
        input_for_hidden_layer = features.dot(self.weights_input_hidden) + self.bias_input_hidden
        hidden_layer_output = self._calculate_sigmoid(input_for_hidden_layer)

        input_for_output_layer = hidden_layer_output.dot(self.weights_hidden_output) + self.bias_hidden_output
        network_output = self._calculate_sigmoid(input_for_output_layer)

        return hidden_layer_output, network_output

    def _propagate_back(self, network_output, hidden_layer_output, features, onehot_encoded_labels):
        error_output = network_output - onehot_encoded_labels
        error_hidden = error_output.dot(self.weights_hidden_output.T) * (hidden_layer_output * (1 - hidden_layer_output))

        delta_weights_input_hidden = (features.T).dot(error_hidden)
        delta_bias_input_hidden = (np.ones(features.shape[0])).dot(error_hidden)

        delta_weights_hidden_output = (hidden_layer_output.T).dot(error_output)
        delta_bias_hidden_output = (np.ones(hidden_layer_output.shape[0])).dot(error_output)

        return delta_weights_input_hidden, delta_bias_input_hidden, delta_weights_hidden_output, delta_bias_hidden_output

    def _regularize(self, delta_weights_input_hidden, delta_weights_hidden_output):
        delta_weights_input_hidden = delta_weights_input_hidden + self.regularization_parameter * self.weights_input_hidden
        delta_weights_hidden_output = delta_weights_hidden_output + self.regularization_parameter * self.weights_hidden_output

        return delta_weights_input_hidden, delta_weights_hidden_output

    def _update_weights(self, delta_weights_input_hidden, delta_bias_input_hidden, delta_weights_hidden_output, delta_bias_hidden_output):
        self.weights_input_hidden -= self.learning_rate * delta_weights_input_hidden
        self.bias_input_hidden -= self.learning_rate * delta_bias_input_hidden

        self.weights_hidden_output -= self.learning_rate * delta_weights_hidden_output
        self.bias_hidden_output -= self.learning_rate * delta_bias_hidden_output

    def _shuffle_data(self, training_data, training_labels):
        shuffled_indexes = np.random.permutation(training_data.shape[0])
        training_data = training_data[shuffled_indexes]
        training_labels = training_labels[shuffled_indexes]

        return training_data, training_labels

    def learn(self, training_data, training_labels, validation_data, validation_labels):
        self._initialize_weights()
        self.cost_history = []
        start_time = time.time()

        for iteration in range(self.iterations):
            training_data, training_labels = self._shuffle_data(training_data, training_labels)
            onehot_encoded_labels = self._encode_labels_onehot(training_labels)

            minibatches = np.array_split(range(training_data.shape[0]), self.minibatches)

            for minibatch in minibatches:
                hidden_layer_output, network_output = self._propagate_forward(training_data[minibatch])

                delta_weights_input_hidden,\
                delta_bias_input_hidden, \
                delta_weights_hidden_output, \
                delta_bias_hidden_output = \
                    self._propagate_back(network_output,
                                         hidden_layer_output,
                                         training_data[minibatch],
                                         onehot_encoded_labels[minibatch]
                                         )

                delta_weights_input_hidden,\
                delta_weights_hidden_output = \
                    self._regularize(delta_weights_input_hidden,
                                     delta_weights_hidden_output
                                     )

                self._update_weights(delta_weights_input_hidden,
                                     delta_bias_input_hidden,
                                     delta_weights_hidden_output,
                                     delta_bias_hidden_output
                                     )

            end_time = time.time()
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

            network_output = self._propagate_forward(training_data)[1]
            cost = self._calculate_cost(onehot_encoded_labels, network_output)

            training_predicted_labels = self.predict_labels(training_data)
            validation_predicted_labels = self.predict_labels(validation_data)

            traininig_set_accuracy = ((np.sum(training_labels == training_predicted_labels)) / training_data.shape[0])
            validation_set_accuracy = ((np.sum(validation_labels == validation_predicted_labels)) / validation_data.shape[0])

            sys.stdout.write('Iteration: %d/%d, '
                             'Cost: %d, '
                             'Training set accuracy: %d%%, '
                             'Validation set accuracy: %d%%, '
                             'Learning time: %s\n' %
                             (iteration + 1,
                              self.iterations,
                              cost,
                              traininig_set_accuracy * 100,
                              validation_set_accuracy * 100,
                              elapsed_time
                              )
                             )

            self.cost_history.append(cost)

    def save_weights_to_file(self, name):
        np.savez(name,
                 bias_input_hidden = self.bias_input_hidden,
                 weights_input_hidden = self.weights_input_hidden,
                 bias_hidden_output = self.bias_hidden_output,
                 weights_hidden_output = self.weights_hidden_output
                 )

    def load_weights_from_file(self, name):
        weights = np.load(name)
        self.weights_input_hidden = weights['weights_input_hidden']
        self.bias_input_hidden = weights['bias_input_hidden']
        self.weights_hidden_output = weights['weights_hidden_output']
        self.bias_hidden_output = weights['bias_hidden_output']
