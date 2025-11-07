# Numpy for useful maths
import numpy
# Sklearn contains some useful CI tools
# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
# k Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
# Matplotlib for plotting
import matplotlib.pyplot as plt

# Load the train and test MNIST data
train = numpy.loadtxt('/Users/sol/GitHub/ComputationalIntelligence/Coursework/MNIST/Fashion/fashion_MNIST/fashion_mnist_train_1000.csv', delimiter=',')
test = numpy.loadtxt('/Users/sol/GitHub/ComputationalIntelligence/Coursework/MNIST/Fashion/fashion_MNIST/fashion_mnist_test_10.csv', delimiter=',')

# Separate labels from training data
train_data = train[:, 1:]
train_labels = train[:, 0]
test_data = test[:, 1:]
test_labels = test[:, 0]

# Select number of components to extract
pca = PCA(n_components = 10)

# Fit to the training data
pca.fit(train_data)

# Determine amount of variance explained by components
print("Total Variance Explained: ", numpy.sum(pca.explained_variance_ratio_))

# Plot the explained variance
plt.plot(pca.explained_variance_ratio_)
plt.title('Variance Explained by Extracted Componenents')
plt.ylabel('Variance')
plt.xlabel('Principal Components')
plt.show()

# Extract the principal components from the training data
train_ext = pca.fit_transform(train_data)

# Transform the test data using the same components
test_ext = pca.transform(test_data)

# Normalise the data sets
min_max_scaler = MinMaxScaler()
train_norm = min_max_scaler.fit_transform(train_ext)
test_norm = min_max_scaler.fit_transform(test_ext)

# Create a KNN classification system with k = 5
# Uses the p2 (Euclidean) norm
knn = KNeighborsClassifier(n_neighbors=5, p=2)
knn.fit(train_norm, train_labels)

# Feed the test data in the classifier to get the predictions
pred = knn.predict(test_norm)

# Check how many were correct
scorecard = []

for i, sample in enumerate(test_data):
# Check if the KNN classification was correct
   if round(pred[i]) == test_labels[i]:
      scorecard.append(1)
   else:
      scorecard.append(0)
   pass
# Calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print("Performance = ", (scorecard_array.sum() / scorecard_array.size) * 100, ' % ')