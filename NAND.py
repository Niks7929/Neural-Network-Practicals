```python
import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([1, 1, 1, 0])

model = MLPClassifier(
    hidden_layer_sizes=(),
    activation="logistic",
    solver="lbfgs",
    max_iter=2000,
    random_state=42
)

model.fit(X, y)

print("NAND Classifier")
for x in X:
    print(x, "=>", model.predict([x])[0])
```
