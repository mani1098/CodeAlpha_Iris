"""Iris species classification with scikit-learn."""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load data
df = pd.read_csv("Iris.csv")
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X, y = df[features], df["Species"]
print("Class counts:\n", y.value_counts(), "\n")

# 2. Stratified train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")

# 3. Candidate models
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
}

results = {}
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = acc
    print(f"{name:20s} CV acc (train): {cv.mean():.3f} +/- {cv.std():.3f} | Test acc: {acc:.3f}")

# 4. Detailed evaluation of the best model
best = max(results, key=results.get)
model = models[best]
pred = model.predict(X_test)
print(f"\n=== Detailed evaluation: {best} ===")
print("Confusion matrix (rows=true, cols=predicted):")
labels = sorted(y.unique())
print(pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels))
print("\nClassification report:\n", classification_report(y_test, pred))
