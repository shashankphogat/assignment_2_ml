import pandas as pd, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

df = pd.read_csv("data.csv").drop(columns=["id"])
X = df.drop(columns=["diagnosis"]); y = df["diagnosis"]
assert "diagnosis" in df.columns and df["diagnosis"].isin([0,1]).all()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes":         GaussianNB(),
    "SVM":                 SVC(probability=True, random_state=42),
}

os.makedirs("model", exist_ok=True)
rows = []
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    p  = pipe.predict(X_test)
    pr = pipe.predict_proba(X_test)[:, 1]
    rows.append({"Model": name,
                 "Accuracy":  round(accuracy_score(y_test,p), 4),
                 "AUC":       round(roc_auc_score(y_test,pr), 4),
                 "Precision": round(precision_score(y_test,p), 4),
                 "Recall":    round(recall_score(y_test,p), 4),
                 "F1":        round(f1_score(y_test,p), 4),
                 "MCC":       round(matthews_corrcoef(y_test,p), 4)})
    joblib.dump(pipe, f"model/{name.lower().replace(' ','_')}.joblib")

import pandas as _pd
_pd.DataFrame(rows).to_csv("metrics_summary.csv", index=False)
X_test.assign(diagnosis=y_test).to_csv("test_data.csv", index=False)
print(_pd.DataFrame(rows).to_string(index=False))
