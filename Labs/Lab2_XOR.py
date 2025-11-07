import numpy as np
import matplotlib.pyplot as plt

def perceptron(inputs_list, weights_list, bias):
    inputs = np.asarray(inputs_list)
    weights = np.asarray(weights_list)
    summation = np.dot(inputs, weights) + bias
    output = 1 if summation > 0 else 0
    return output

# Define the XOR gate using two layers of perceptrons
def xor_gate(input_vector):
    # Layer 1: AND gate and OR gate
    and_gate = perceptron(input_vector, [1, 1], -1.5)  # AND gate
    or_gate = perceptron(input_vector, [1, 1], -0.5)   # OR gate

    # Debugging: Print intermediate outputs
    print(f"Input: {input_vector}, AND Gate: {and_gate}, OR Gate: {or_gate}")

    # Layer 2: Combine AND and OR gates to compute XOR
    xor_output = perceptron([and_gate, or_gate], [1, -1], -0.5)  # XOR gate

    # Debugging: Print XOR output
    print(f"XOR Output: {xor_output}")

    return xor_output

# Inputs for the XOR gate
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Plot the XOR gate
fig = plt.figure()

for input_vector in inputs:
    result = xor_gate(input_vector)
    if input_vector[0] != input_vector[1]:
        plt.scatter(input_vector[0], input_vector[1], s=50, color="green", zorder=3)
    elif result == 0:
        plt.scatter(input_vector[0], input_vector[1], s=50, color="red", zorder=3)

# Add linear separator lines for the AND gate and OR gate
x = np.linspace(-0.5, 1.5, 100)

# AND gate decision boundary: weights = [1, 1], bias = -1.5
y_and = (-1 / 1) * x + 1.5
plt.plot(x, y_and, color="blue", linestyle="-", label="AND Gate Separator")

# OR gate decision boundary: weights = [1, 1], bias = -0.5
y_or = (-1 / 1) * x + 0.5
plt.plot(x, y_or, color="blue", linestyle="-", label="OR Gate Separator")

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("XOR Gate State Space with Linear Separators")

plt.grid(True, linewidth=1, linestyle=':')

plt.tight_layout()

plt.show()