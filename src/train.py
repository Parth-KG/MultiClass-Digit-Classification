import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_mnist          
from model import MultiClassNN             
from preprocessing import preprocess_data


def train():
    # 1. Load and Preprocess
    print("Loading Data...")
    X, y = load_mnist()
    
    print("Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    print(f"Shape for training: X={X_train.shape}, y={y_train.shape}")

    # 2. Initialize Model
    # Input: 784 (pixels), Hidden: 64 (neurons), Output: 1 (binary)
    model = MultiClassNN(784, 64, 10)

    # 3. Training Loop
    epochs = 2000        # How many times to see the whole dataset
    learning_rate = 0.005  # How big of a step to take
    losses = []

    print(f"Starting training for {epochs} epochs...")
    
    for i in range(epochs+1):
        # A. Forward Pass (Make a guess)
        A2, cache = model.forward(X_train)

        # B. Compute Loss (How bad was the guess?)
        loss = model.compute_loss(A2, y_train)
        
        # C. Backward Pass (Calculate gradients)
        grads = model.backward(X_train, y_train, cache)

        # D. Update Parameters (Improve the model)
        # Note: We need to manually update the weights in the dictionary
        model.W1 = model.W1 - learning_rate * grads["dW1"]
        model.b1 = model.b1 - learning_rate * grads["db1"]
        model.W2 = model.W2 - learning_rate * grads["dW2"]
        model.b2 = model.b2 - learning_rate * grads["db2"]

        # Record loss
        if i % 100 == 0:
            losses.append(loss)
            print(f"Epoch {i}: Loss {loss:.4f}")

    # 4. Evaluate on Test Set
    print("\nTraining Complete. Testing...")
    A2_test, _ = model.forward(X_test)
    predictions = np.argmax(A2_test, axis=0)   
    labels = np.argmax(y_test, axis=0)         
    accuracy = np.mean(predictions == labels) * 100
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Optional: Plot the loss curve
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Iterations (x100)")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    train()