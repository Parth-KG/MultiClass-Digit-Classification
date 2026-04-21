import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist():
    print("Downloading MNIST... this may take a while.")
    # Fetch data from OpenML
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X_all = mnist.data  # The images (pixels)
    y_all = mnist.target # The labels (0-9)

    # Convert labels to integers
    y_all = y_all.astype(np.uint8)

    print(f"Dataset shape: {X_all.shape}") # Should be approx 14k images
    return X_all, y_all

if __name__ == "__main__":
    X, y = load_mnist()
    print("Success. Data is ready.")