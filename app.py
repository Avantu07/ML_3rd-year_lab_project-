
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def load_and_train_models():

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal", "target"
    ]

    df = pd.read_csv(url, names=columns, na_values="?")
    df = df.dropna()

    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC()
        )
    }

    scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        scores[name] = round(
            accuracy_score(y_test, prediction) * 100, 2
        )

    return df, X, models, scores


def predict_heart_disease(patient, model):
    result = model.predict(patient)[0]
    return result
