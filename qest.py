import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Example input (flattened image vectors)
X = np.array([[0.1, 0.2, 0.3, 0.4],  # Dog image example (simplified)
              [0.5, 0.6, 0.7, 0.8]])  # Cat image example (simplified)

# Example output (1 for dog, 0 for cat)
y = np.array([[1], [0]])

# Network architecture
input_layer_size = 4  # Number of features
hidden_layer1_size = 5  # First hidden layer neurons
hidden_layer2_size = 4  # Second hidden layer neurons
output_layer_size = 1  # Binary classification

# Fixed weights and biases
W1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
               [0.6, 0.7, 0.8, 0.9, 1.0],
               [1.1, 1.2, 1.3, 1.4, 1.5],
               [1.6, 1.7, 1.8, 1.9, 2.0]])
b1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

W2 = np.array([[0.2, 0.3, 0.4, 0.5],
               [0.6, 0.7, 0.8, 0.9],
               [1.0, 1.1, 1.2, 1.3],
               [1.4, 1.5, 1.6, 1.7],
               [1.8, 1.9, 2.0, 2.1]])
b2 = np.array([[0.1, 0.2, 0.3, 0.4]])

W3 = np.array([[0.2], [0.3], [0.4], [0.5]])
b3 = np.array([[0.1]])

# Training parameters
epochs = 100000
learning_rate = 0.01  # Added learning rate for weight updates

# Training loop
for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = sigmoid(z3)

    # Compute error
    error = y - a3

    # Backpropagation
    d_a3 = error * sigmoid_derivative(a3)
    d_W3 = np.dot(a2.T, d_a3)
    d_b3 = np.sum(d_a3, axis=0, keepdims=True)

    d_z2 = np.dot(d_a3, W3.T)
    d_a2 = d_z2 * sigmoid_derivative(a2)
    d_W2 = np.dot(a1.T, d_a2)
    d_b2 = np.sum(d_a2, axis=0, keepdims=True)

    d_z1 = np.dot(d_a2, W2.T)
    d_a1 = d_z1 * sigmoid_derivative(a1)
    d_W1 = np.dot(X.T, d_a1)
    d_b1 = np.sum(d_a1, axis=0, keepdims=True)

    # Update weights and biases
    W3 += learning_rate * d_W3
    b3 += learning_rate * d_b3
    W2 += learning_rate * d_W2
    b2 += learning_rate * d_b2
    W1 += learning_rate * d_W1
    b1 += learning_rate * d_b1

    # Print loss every 10000 epochs
    if epoch % 10000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Epoch {epoch}, Loss: {loss}")

# Final predictions
y_pred = a3
print("Final Predictions:", y_pred)

# Answering the given questions
a3_values = y_pred.flatten().tolist()
a2_min = a2.min()
d_W1_min = d_W1.min()

print(f"a3 = [{', '.join(map(str, a3_values))}]")
print(f"a2.min() = {a2_min}")
print(f"d_W1.min() = {d_W1_min}")

# General Conclusion after 100000 epochs
if y_pred[0][0] > 0.5 and y_pred[1][0] < 0.5:
    conclusion = "NN predicts image of dog"
elif y_pred[0][0] < 0.5 and y_pred[1][0] > 0.5:
    conclusion = "NN predicts image of cat"
else:
    conclusion = "NN can't define correct image class"

print("General Conclusion after 100000 epochs:", conclusion)