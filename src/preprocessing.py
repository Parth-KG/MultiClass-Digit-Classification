import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(X, y):
    X_normalized = X / 255.0
    num_classes = 10
    y_onehot = np.eye(num_classes)[y]

    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_onehot, test_size=0.2, random_state=42
    )


    print(f"Training Data: {X_train.shape}")
    print(f"Test Data: {X_test.shape}")
    
    return X_train.T, X_test.T, y_train.T, y_test.T

if __name__ == "__main__":
    from data_loader import load_mnist
    
    X_raw, y_raw = load_mnist()
    X_train, X_test, y_train, y_test = preprocess_data(X_raw, y_raw)