#  Lab code for week 3 of Computational Intelligence

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
   
input_nodes = 2

hidden_nodes = 2

output_nodes = 1

learning_rate = 0.5

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

target_list = [[0], [0], [0], [2]]

print("Untrained Neural Network Results:")
for i in test_inputs:
   print(f"Input: {i} -> Output: {n.query(i)}")

for i in range(100000):
   for input, target in zip(test_inputs, target_list):
      n.train(input, target)

print("Trained Neural Network Results:")
for i in test_inputs:
   print(f"Input: {i} -> Output: {n.query(i)}")