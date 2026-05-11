import numpy as np
from PIL import Image, ImageOps
from model import MultiClassNN
import sys

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = ImageOps.invert(img)
    img = img.point(lambda x: 0 if x < 200 else 255)  # was 128, now 200
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img = img.resize((20, 20), Image.LANCZOS)
    padded = Image.new("L", (28, 28), 0)
    padded.paste(img, (4, 4))
    X = np.array(padded) / 255.0
    return X.reshape(784, 1)

def predict(image_path):
    X = preprocess_image(image_path)

    weights = np.load("weights.npy", allow_pickle=True).item()
    model = MultiClassNN(784, 64, 10)
    model.W1, model.b1 = weights["W1"], weights["b1"]
    model.W2, model.b2 = weights["W2"], weights["b2"]

    A2, _ = model.forward(X)
    prediction = np.argmax(A2)
    print(f"Predicted Digit: {prediction}")

if __name__ == "__main__":
    predict(sys.argv[1])