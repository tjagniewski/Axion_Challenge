# Breast Cancer Wisconsin, ladowany z sklearn 
# hipoteza: dataset jest niezbalansowany (63% benign / 37% malignant), wiec sama accuracy
# moze nie pokazac ze model slabo wylapuje malignant - sprawdzam porownujac zwykla
# regresje logistyczna z wersja z class_weight='balanced'

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 0 = malignant, 1 = benign

print(y.value_counts().rename({0: "malignant", 1: "benign"}))
print("malignant share:", round((y == 0).mean(), 3))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# baseline - zawsze zgaduje benign, zeby zobaczyc co by dala sama accuracy
baseline = np.ones_like(y_test)
print("\nbaseline accuracy:", accuracy_score(y_test, baseline))
print("baseline malignant recall:", recall_score(y_test, baseline, pos_label=0))


def run(name, model):
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)[:, 1]
    print(f"\n{name}")
    print("accuracy:", accuracy_score(y_test, pred))
    print("precision (benign):", precision_score(y_test, pred, pos_label=1))
    print("recall (malignant):", recall_score(y_test, pred, pos_label=0))
    print("f1 (malignant):", f1_score(y_test, pred, pos_label=0))
    print("roc_auc:", roc_auc_score(y_test, proba))
    print(confusion_matrix(y_test, pred))
    return pred


pred_plain = run("logreg plain", LogisticRegression(max_iter=1000))
pred_bal = run("logreg class_weight=balanced", LogisticRegression(max_iter=1000, class_weight="balanced"))

r1 = recall_score(y_test, pred_plain, pos_label=0)
r2 = recall_score(y_test, pred_bal, pos_label=0)
a1 = accuracy_score(y_test, pred_plain)
a2 = accuracy_score(y_test, pred_bal)

print(f"\nrecall malignant: {r1:.3f} -> {r2:.3f}")
print(f"accuracy: {a1:.3f} -> {a2:.3f}")
if r2 > r1 and a2 >= a1 - 0.02:
    print("hipoteza potwierdzona - balanced lapie wiecej malignant bez realnej straty accuracy")
else:
    print("hipoteza niepotwierdzona na tym splicie")

# wyniki
# benign       357
# malignant    212
# Name: count, dtype: int64
# malignant share: 0.373

# baseline accuracy: 0.6293706293706294
# baseline malignant recall: 0.0

# logreg plain
# accuracy: 0.986013986013986
# precision (benign): 0.9888888888888889
# recall (malignant): 0.9811320754716981
# f1 (malignant): 0.9811320754716981
# roc_auc: 0.9976939203354298
# [[52  1]
#  [ 1 89]]

# logreg class_weight=balanced
# accuracy: 0.965034965034965
# precision (benign): 0.9885057471264368
# recall (malignant): 0.9811320754716981
# f1 (malignant): 0.9541284403669725
# roc_auc: 0.9976939203354298
# [[52  1]
#  [ 4 86]]

# recall malignant: 0.981 -> 0.981
# accuracy: 0.986 -> 0.965
# hipoteza niepotwierdzona na tym splicie