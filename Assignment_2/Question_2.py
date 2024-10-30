import pandas as pd
from sklearn.model_selection import train_test_split
import math
from collections import Counter
from google.colab import drive

# 1. Load dataset
def load_dataset(filename):
    df = pd.read_csv(filename)
    return df

data = load_dataset('lab2_dataset/q2/Iris.csv')
print(data)
X = data.drop(columns=['Species']).values
y = data['Species'].values


# 2. Split dataset into training and test sets using scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# 3. Define the kNN algorithm
def euclidean_distance(instance1, instance2):
    """Calculate the Euclidean distance between two instances."""
    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(instance1, instance2)))
    return distance

def get_neighbors(X_train, test_instance, k):
    """Get the k nearest neighbors for a test instance."""
    distances = []
    for idx, train_instance in enumerate(X_train):
        dist = euclidean_distance(train_instance, test_instance)
        distances.append((dist, idx))

    # Sort by distance and return k neighbors
    distances.sort(key=lambda x: x[0])
    neighbors = [idx for _, idx in distances[:k]]
    return neighbors

def get_response(neighbors, y_train):
    """Determine the class label for a current instance based on the majority
    class label of its k neighbors."""
    neighbor_labels = [y_train[idx] for idx in neighbors]
    most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
    return most_common_label

# 4. Use the kNN algorithm to predict the class labels of the test set
k = 3
predictions = []
for current_instance in X_test:
    neighbors = get_neighbors(X_train, current_instance, k)
    prediction = get_response(neighbors, y_train)
    predictions.append(prediction)

# 5. Calculate the accuracy of the predictions
correct = sum([y_true == y_pred for y_true, y_pred in zip(y_test, predictions)])
accuracy = (correct / len(y_test)) * 100.0
print(f"Accuracy: {accuracy:.2f}%")
