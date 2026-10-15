```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def build_model(lr, architecture):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))

    for units in architecture:
        model.add(Dense(units, activation="relu"))

    model.add(Dense(3, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

results = []

for lr in [0.01, 0.001]:
    for architecture in [(8,), (16, 8)]:
        for epochs in [50, 100]:
            tf.keras.backend.clear_session()
            model = build_model(lr, architecture)

            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=16,
                verbose=0
            )

            predictions = np.argmax(
                model.predict(X_test, verbose=0), axis=1
            )
            acc = accuracy_score(y_test, predictions)

            results.append((lr, architecture, epochs, acc))
            print(
                f"LR={lr}, Architecture={architecture}, "
                f"Epochs={epochs}, Test Accuracy={acc:.4f}"
            )

print("\nBest configuration:")
print(max(results, key=lambda item: item[3]))
```
