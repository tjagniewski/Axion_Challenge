# load_digits z sklearn (cyfry 8x8, 10 klas) - sprawdzam co sie stanie jak wywalic
# funkcje aktywacji z sieci (MLP). Bez nieliniowosci kolejne warstwy matematycznie
# skladaja sie w jedna liniowa transformacje, wiec MLP z activation='identity' powinien
# wypasc podobnie do zwyklej regresji logistycznej i wyraznie gorzej od MLP z relu,
# bo cyfry nie sa liniowo separowalne w przestrzeni pikseli

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_digits()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)


def run(name, model):
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    print(f"{name}: accuracy = {acc:.3f}")
    return acc


acc_logreg = run("logreg (liniowy baseline)", LogisticRegression(max_iter=2000))
acc_relu = run("MLP relu", MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=2000, random_state=42))
acc_identity = run("MLP identity (bez aktywacji)", MLPClassifier(hidden_layer_sizes=(64,), activation="identity", max_iter=2000, random_state=42))

print()
print("relu vs identity:", round(acc_relu - acc_identity, 3))
print("identity vs logreg:", round(acc_identity - acc_logreg, 3))

if acc_relu > acc_identity and abs(acc_identity - acc_logreg) < 0.03:
    print("hipoteza potwierdzona - bez aktywacji MLP zachowuje sie jak model liniowy i traci do relu")
else:
    print("hipoteza niepotwierdzona na tym splicie")

# wyniki
# logreg (liniowy baseline): accuracy = 0.978
# MLP relu: accuracy = 0.980
# MLP identity (bez aktywacji): accuracy = 0.960

# relu vs identity: 0.02
# identity vs logreg: -0.018
# hipoteza potwierdzona - bez aktywacji MLP zachowuje sie jak model liniowy i traci do relu
