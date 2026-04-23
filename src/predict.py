import numpy as np
from PIL import Image
from model import MultiClassNN

def predict(image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert("L")  # grayscale
    img = img.resize((28, 28))
    X = np.array(img) / 255.0
    X = X.reshape(784, 1)

    # Load weights
    weights = np.load("weights.npy", allow_pickle=True).item()
    model = MultiClassNN(784, 64, 10)
    model.W1, model.b1 = weights["W1"], weights["b1"]
    model.W2, model.b2 = weights["W2"], weights["b2"]

    # Predict
    A2, _ = model.forward(X)
    prediction = np.argmax(A2)
    print(f"Predicted Digit: {prediction}")

if __name__ == "__main__":
    import sys
    predict(sys.argv[1])