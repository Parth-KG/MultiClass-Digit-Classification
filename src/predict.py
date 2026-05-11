import numpy as np
import os   
from PIL import Image
from model import MultiClassNN
import sys

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    X = np.array(img) / 255.0
    return X.reshape(784, 1)

def predict(image_path):
    X = preprocess_image(image_path)

    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npy")
    weights = np.load(weights_path, allow_pickle=True).item()
    model = MultiClassNN(784, 64, 10)
    model.W1, model.b1 = weights["W1"], weights["b1"]
    model.W2, model.b2 = weights["W2"], weights["b2"]

    A2, _ = model.forward(X)
    prediction = np.argmax(A2)
    print(f"Predicted Digit: {prediction}")

if __name__ == "__main__":
    predict(sys.argv[1])