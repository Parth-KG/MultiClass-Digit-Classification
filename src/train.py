import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_mnist
from model import MultiClassNN
from preprocessing import preprocess_data


def train():
    print("Loading Data...")
    X, y = load_mnist()

    print("Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print(f"Shape for training: X={X_train.shape}, y={y_train.shape}")

    model = MultiClassNN(784, 64, 10)

    # Hyperparameters
    epochs = 30
    batch_size = 64
    learning_rate = 0.1

    m = X_train.shape[1]   # number of training examples
    losses = []

    print(f"Starting training for {epochs} epochs (batch size {batch_size})...")

    for epoch in range(epochs):
        # Shuffle the training data each epoch (critical for SGD)
        permutation = np.random.permutation(m)
        X_shuffled = X_train[:, permutation]
        y_shuffled = y_train[:, permutation]

        epoch_loss = 0.0
        num_batches = 0

        # Iterate through mini-batches
        for start in range(0, m, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[:, start:end]
            y_batch = y_shuffled[:, start:end]

            # Forward
            A2, cache = model.forward(X_batch)

            # Loss
            loss = model.compute_loss(A2, y_batch)
            epoch_loss += loss
            num_batches += 1

            # Backward
            grads = model.backward(X_batch, y_batch, cache)

            # Update
            model.W1 -= learning_rate * grads["dW1"]
            model.b1 -= learning_rate * grads["db1"]
            model.W2 -= learning_rate * grads["dW2"]
            model.b2 -= learning_rate * grads["db2"]

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        # Track train accuracy each epoch so you can see progress
        A2_train, _ = model.forward(X_train)
        train_acc = np.mean(np.argmax(A2_train, axis=0) == np.argmax(y_train, axis=0)) * 100

        print(f"Epoch {epoch+1}/{epochs}: Loss {avg_loss:.4f} | Train Acc {train_acc:.2f}%")

    print("\nTraining Complete. Testing...")
    A2_test, _ = model.forward(X_test)
    predictions = np.argmax(A2_test, axis=0)
    labels = np.argmax(y_test, axis=0)
    accuracy = np.mean(predictions == labels) * 100

    np.save("weights.npy", {
        "W1": model.W1, "b1": model.b1,
        "W2": model.W2, "b2": model.b2
    })

    print(f"Test Accuracy: {accuracy:.2f}%")

    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    train()