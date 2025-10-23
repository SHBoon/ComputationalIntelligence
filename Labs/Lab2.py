# Lab code for week 2 of Computational Intelligence

import numpy as np
import matplotlib.pyplot as plt

def perceptron(inputs_list, weights_list, bias):

   inputs = np.asarray(inputs_list)

   weights = np.asarray(weights_list)

   summation = np.dot(inputs, weights) + bias

   output = 1 if summation > 0 else 0

   return output

inputs = [[0, 0], [0, 1], [1, 0] ,[1, 1]]
weights = [-1, -1]
bias = 1.5

fig = plt.figure()

for input_vector in inputs:
   result = perceptron(input_vector, weights, bias)

   if result == 1:
      plt.scatter(input_vector[0], input_vector[1], s=50, color="green", zorder=3)
   elif result == 0:
      plt.scatter(input_vector[0], input_vector[1], s=50, color="red", zorder=3)

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

x = np.array([-10, 10])
y = (-weights[0]/weights[1]) * x - (bias/weights[1])

plt.plot(x, y, color="blue", zorder=2)

plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("State Space of Gate")

plt.grid(True, linewidth=1, linestyle=':')

plt.tight_layout()

plt.show()