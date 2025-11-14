import matplotlib.pyplot as plt
import numpy as np
import time
from Coursework1_MNIST_Fashion import test_fashion_mnist

# List of training dataset sizes to test
training_sizes = [100, 500, 1000, 5000, 10000, 30000, 60000]

# Lists to store the accuracy and time for each training size
accuracies = []
times = []

for training_size in training_sizes:
    # Measure the time taken for training and testing
    start_time = time.time()
    scorecard = test_fashion_mnist(training_length=training_size)
    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    times.append(elapsed_time)
    
    # Calculate the accuracy
    scorecard_array = np.asarray(scorecard)
    accuracy = scorecard_array.sum() / scorecard_array.size
    accuracies.append(accuracy)
    
    print(f"Training Size: {training_size}, Accuracy: {accuracy * 100:.2f}%, Time: {elapsed_time:.2f} seconds")

# Plot training dataset size vs accuracy and time
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot accuracy on the left y-axis
ax1.set_xlabel('Training Dataset Size')
ax1.set_ylabel('Accuracy (%)', color='b')
ax1.plot(training_sizes, [a * 100 for a in accuracies], marker='o', linestyle='-', color='b', label='Accuracy')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# Plot time on the right y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Time (seconds)', color='r')
ax2.plot(training_sizes, times, marker='o', linestyle='--', color='r', label='Time')
ax2.tick_params(axis='y', labelcolor='r')

# Add a title and legends
plt.title('Training Dataset Size vs Accuracy and Time')
fig.tight_layout()
plt.show()