import numpy as np
from itertools import product
from Coursework1_MNIST_Fashion import test_fashion_mnist

# Define the hyperparameter search space
hidden_nodes_list = [50, 100, 200]
learning_rate_list = [0.1, 0.3, 0.7]
training_iterations_list = [1, 2, 5]
training_length_list = [1000, 5000, 10000]

# Initialize variables to store the best parameters and accuracy
best_accuracy = 0
best_params = {}

# Iterate over all combinations of hyperparameters
for hidden_nodes, learning_rate, training_iterations, training_length in product(
    hidden_nodes_list, learning_rate_list, training_iterations_list, training_length_list
):
    print(f"Testing: hidden_nodes={hidden_nodes}, learning_rate={learning_rate}, "
          f"training_iterations={training_iterations}, training_length={training_length}")
    
    # Run the test_mnist_hw function with the current parameters
    scorecard = test_fashion_mnist(
        input_nodes=784,
        hidden_nodes=hidden_nodes,
        output_nodes=10,
        learning_rate=learning_rate,
        training_iterations=training_iterations,
        training_length=training_length
    )
    
    # Calculate accuracy
    scorecard_array = np.asarray(scorecard)
    accuracy = scorecard_array.sum() / scorecard_array.size
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Update the best parameters if the current accuracy is higher
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = {
            "hidden_nodes": hidden_nodes,
            "learning_rate": learning_rate,
            "training_iterations": training_iterations,
            "training_length": training_length
        }

# Print the best parameters and accuracy
print("\nBest Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")