# Handwritten Digit Recognizer

A multi-class neural network built **from scratch using NumPy** to classify handwritten digits (0–9) from the MNIST dataset. No PyTorch or TensorFlow — just raw math.

## How It Works

- 2-layer neural network (784 → 64 → 10)
- ReLU activation in the hidden layer
- Softmax activation in the output layer
- Categorical cross-entropy loss
- Gradient descent with manual backpropagation

## Project Structure

```
src/
├── data_loader.py      # Fetches MNIST from OpenML
├── preprocessing.py    # Normalizes and one-hot encodes labels
├── model.py            # Neural network (forward, backward, loss)
└── train.py            # Training loop and evaluation
```

## Dataset

[MNIST 784](https://www.openml.org/d/554) via `sklearn.datasets.fetch_openml` — 70,000 grayscale images of handwritten digits, each 28×28 pixels.

## Setup

```bash
pip install numpy scikit-learn matplotlib
python train.py
```

## Results

| Metric | Value |
|--------|-------|
| Architecture | 784 → 64 → 10 |
| Epochs | 2000 |
| Learning Rate | 0.005 |
| Test Accuracy | ~XX% |

> Update the accuracy after training.

## Tech Stack

- Python
- NumPy
- scikit-learn (dataset only)
- Matplotlib (loss curve)

## Author

**Parth Krishan Goswami**  
B.Tech Information Technology, GGSIPU  
[GitHub](https://github.com/Parth-KG) · parthkrishangoswami@gmail.com
