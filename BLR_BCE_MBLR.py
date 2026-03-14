"""
Manual implementation of a Binary Linear Regression using NumPy.

Model:          Y = sigmoid(W @ X + b)
Cost Function:  Binary Cross-Entropy (BCE)
Optimisation:   Mini-Batch Gradient Descent (MBGD)
"""

import numpy as np

def main():
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]) 
    y_train = np.array([0, 0, 0, 1, 1, 1]) 

    X_train = standardise(X_train)

    W, b = mini_batch_gradient_descent(W=np.array([0.0,0.0]), b=0, learn_rate=0.01, 
                                iterations=10000, y_train=y_train, X_train=X_train, batch_size = 1)
    print(f"Final Results:\nw: {W.flatten()}\nb: {b:.2f}")


def standardise(variable):
    """
    Standardise data to have: mean = 0, and standard deviation = 1.
    Used to make gradient descent converge faster and more reliably.
    
    Returns the X matrix with each column standardised. 
    """
    mean = np.mean(variable, axis=0)
    std = np.std(variable, axis=0)
    return (variable - mean) / std


def mini_batch_gradient_descent(W, b, learn_rate, iterations, y_train, X_train, batch_size):
    """
    Train a binary logistic regression model using Mini-Batch Gradient Descent.

    The training data is shuffled at the start of each iteration and
    divided into smaller batches. For each batch, the gradient of the
    Binary Cross-Entropy (BCE) cost function is computed and the model
    parameters (weights and bias) are updated.

    Returns the final weight matrix (W) and the final bias (b).
    """
    m = X_train.shape[0]
    W = W.reshape(-1,1)
    y_train = y_train.reshape(-1,1)

    for i in range(iterations):
        # shuffle data
        permutation = np.random.permutation(m)
        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            # Formula: y_pred = 1 / [1 + e^-(X_train @ W + b)]
            # Shapes: (6,1) = 1 / [1 + e^-{(6,2) @ (2,1) + scalar}]
            y_pred = 1 / (1 + np.exp(-((X_batch @ W) + b)))

            # BCE, and partial derivatives for W and b
            cost = -np.mean(y_batch * np.log(y_pred) + (1 - y_batch) * np.log(1 - y_pred))
            dw = (1/batch_size) * (X_batch.T @ (y_pred - y_batch))
            db = (1/batch_size) * np.sum(y_pred - y_batch)

            # update gradients
            W -= learn_rate * dw
            b -= learn_rate * db

            if i%100 == 0:
                print(f"Iteration: {i} Batch {j}\tCost: {cost:.4f}\tW: {W.flatten()}\tb: {b:.2f}")
    return W, b


if __name__ == '__main__':
    main()
