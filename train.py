import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import Normalizer,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import json
import joblib

# Charger les donnees
iris = load_iris()
X, y = iris.data, iris.target


le = LabelEncoder()
y = le.fit_transform(y)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalizing data

scaler= Normalizer().fit(X_train) # the scaler is fitted to the training set
normalized_x_train= scaler.transform(X_train) # the scaler is applied to the training set
normalized_x_test= scaler.transform(X_test) # the scaler is applied to the test set

# Fitting clasifier to the Training set
# Loading libraries

# Instantiate learning model (k = 3)
model = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
model.fit(normalized_x_train, y_train)

# Predicting the Test set results
y_pred = model.predict(normalized_x_test)



accuracy = accuracy_score(y_test, y_pred)

joblib.dump(model, 'models/iris_model.pkl')

# Sauvegarder les metriques
metrics = {
    "accuracy": accuracy,
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Accuracy: {accuracy:.4f}")