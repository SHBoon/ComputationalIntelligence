import numpy as np
import matplotlib.pyplot as plt
import scipy.special

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        # Initialize weights with a normal distribution
        self.wih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        self.lr = learning_rate

        # Sigmoid activation function
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs_array = np.array(inputs_list, ndmin=2).T
        targets_array = np.array(targets_list, ndmin=2).T

        # Forward pass
        hidden_inputs = np.dot(self.wih, inputs_array)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # Calculate errors
        output_errors = targets_array - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update weights
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs_array))

    def query(self, inputs_list):
        inputs_array = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs_array)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


def test_fashion_mnist(input_nodes=784, hidden_nodes=100, output_nodes=10, learning_rate=0.3, training_iterations=5, training_length=1000):
    # Open the Fashion MNIST training dataset
    data_file = open('/Users/sol/GitHub/ComputationalIntelligence/Coursework/MNIST/Fashion/fashion_MNIST/fashion_mnist_train.csv', 'r')  # Update with the correct path
    data_list = data_file.readlines()
    data_file.close()

    # Use only a subset of the training data
    data_list = data_list[:training_length]

    # Initialize the neural network
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Train the network
    for _ in range(training_iterations):
        for record in data_list:
            all_values = record.split(',')

            # Normalize and shift inputs
            inputs = (np.asarray(all_values[1:], dtype=np.float64) / 255.0 * 0.99) + 0.01

            # Create target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99

            # Train the network
            n.train(inputs, targets)

    # Open the Fashion MNIST test dataset
    test_data_file = open('Coursework/MNIST/Fashion/fashion_MNIST/fashion_mnist_test.csv', 'r')  # Update with the correct path
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # Scorecard for performance evaluation
    scorecard = []

    for record in test_data_list:
        all_values = record.split(',')

        # Normalize and shift inputs
        inputs = (np.asarray(all_values[1:], dtype=np.float64) / 255.0 * 0.99) + 0.01

        # Query the network
        outputs = n.query(inputs)

        # The index of the highest value corresponds to the predicted label
        label = np.argmax(outputs)

        # Append 1 for correct prediction, 0 otherwise
        correct_label = int(all_values[0])
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    # Calculate and print the accuracy
    scorecard_array = np.asarray(scorecard)
    accuracy = scorecard_array.sum() / scorecard_array.size
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return scorecard