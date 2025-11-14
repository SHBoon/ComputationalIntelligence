import numpy as np
import matplotlib.pyplot as plt
import scipy.special

class NeuralNetwork:
   def __init__ (self, input_nodes, hidden_nodes, output_nodes, learning_rate):
      self.i_nodes = input_nodes

      self.h_nodes = hidden_nodes

      self.o_nodes = output_nodes

      self.wih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))

      self.who = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

      self.lr = learning_rate

      self.activation_function = lambda x: scipy.special.expit(x)

   def train(self, inputs_list, targets_list):
      inputs_array = np.array(inputs_list, ndmin=2).T

      targets_array = np.array(targets_list, ndmin=2).T

      hidden_inputs = np.dot(self.wih, inputs_array)

      hidden_outputs = self.activation_function(hidden_inputs)

      final_inputs = np.dot(self.who, hidden_outputs)

      final_outputs = self.activation_function(final_inputs)

      output_errors = targets_array - final_outputs

      hidden_errors = np.dot(self.who.T, output_errors)

      self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
      
      self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs_array))
      
   def query(self, inputs_list):
      inputs_array = np.array(inputs_list, ndmin=2).T

      hidden_inputs = np.dot(self.wih, inputs_array)

      hidden_outputs = self.activation_function(hidden_inputs)

      final_inputs = np.dot(self.who, hidden_outputs)

      final_outputs = self.activation_function(final_inputs)

      return final_outputs

def test_mnist_hw(input_nodes = 784, hidden_nodes = 100, output_nodes = 10, learning_rate = 0.3, training_iterations = 1, training_length=1000):
   # Open the training samples in read mode
   data_file = open('/Users/sol/GitHub/ComputationalIntelligence/Coursework/MNIST/Handwriting/MNIST/mnist_train.csv', 'r')
                  
   # Read all of the lines from the file into memory
   data_list = data_file.readlines()

   # Close the file
   data_file.close()

   data_list = data_list[:training_length]

   n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

   for _ in range(training_iterations):
      for digit in data_list:
         # Split the record by the commas
         all_values = digit.split(',')

         # Normalise and shift
         inputs = (np.asarray(all_values[1:], dtype=np.float64) / 255.0 * 0.99) + 0.01

         # Create the target output values (all 0.01, except the desired label which is 0.99)
         targets = np.zeros(output_nodes) + 0.01

         # Use the label to identify target ouput
         targets[int(all_values[0])] = 0.99

         # Train the network

         n.train(inputs, targets)

         pass

   # Load the MNIST test samples CSV file into a list
   test_data_file = open('/Users/sol/GitHub/ComputationalIntelligence/Coursework/MNIST/Handwriting/MNIST/mnist_test.csv', 'r')
   test_data_list = test_data_file.readlines()
   test_data_file.close()

   # Scorecard list for how well the network performs, initially empty
   scorecard = []

   # Loop through all of the records in the test data set
   for record in test_data_list:
      # Split the record by the commas
      all_values = record.split(',')

      # The correct label is the first value
      correct_label = int(all_values[0])

      # Scale and shift the inputs
      inputs = (np.asarray(all_values[1:], dtype=np.float64) / 255.0 * 0.99) + 0.01

      # Query the network
      outputs = n.query(inputs)

      # The index of the highest value output corresponds to the label
      label = np.argmax(outputs)

      # Append either a 1 or a 0 to the scorecard list
      if (label == correct_label):
         scorecard.append(1)
      else:
         scorecard.append(0)
         pass
      pass
      
   return scorecard