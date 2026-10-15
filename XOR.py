```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 0])

# Single-layer network: no hidden layer
single = MLPClassifier(
    hidden_layer_sizes=(),
    solver="lbfgs",
    max_iter=2000,
    random_state=42
)
single.fit(X, y)

print("Single-layer XOR predictions:")
print(single.predict(X))
print("Accuracy:", accuracy_score(y, single.predict(X)))

# Multi-layer network: hidden layer
multi = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="tanh",
    solver="lbfgs",
    max_iter=5000,
    random_state=42
)
multi.fit(X, y)

print("\nMulti-layer XOR predictions:")
print(multi.predict(X))
print("Actual output:", y)
print("Accuracy:", accuracy_score(y, multi.predict(X)))
```
