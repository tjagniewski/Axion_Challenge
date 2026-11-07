import numpy as np
from mini_bracket_simulator import X, Y, Z, I, H, Rx, Ry, Rz, CNOT

CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]])

SINGLE_GATES = {"X": X, "Y": Y, "Z": Z, "H": H, "I": I}
ROTATIONS = {"RX": Rx, "RY": Ry, "RZ": Rz}
TWO_QUBIT_GATES = {"CNOT": CNOT, "CZ": CZ}


def pick_single_gate(label=""):
    name = input(f"Wybierz bramkę {label}(X/Y/Z/H/RX/RY/RZ): ").strip().upper()
    if name in SINGLE_GATES:
        return SINGLE_GATES[name]
    if name in ROTATIONS:
        theta = float(input("Podaj kąt theta (w radianach): "))
        return ROTATIONS[name](theta)
    raise ValueError(f"Nieznana bramka: {name}")


def read_initial_state(n_qubits):
    dim = 2 ** n_qubits
    custom = input(f"Podaj WŁASNY stan wejściowy ({dim} amplitud) lub użyj DOMYŚLNEGO "
                    f"|{'0' * n_qubits}>? (W/D): ").strip().lower()
    if custom != "w":
        state = np.zeros(dim, dtype=complex)
        state[0] = 1
        return state

    raw = input(f"Podaj {dim} amplitud oddzielonych przecinkiem (np. 1,0 albo 0.707,0.707): ")
    state = np.array([complex(x) for x in raw.split(",")], dtype=complex)
    if state.shape != (dim,) or not np.isclose(sum(abs(state) ** 2), 1):
        print("Nieprawidłowy stan wejściowy")
        return None
    return state


def universal_simulator():
    n_qubits = int(input("Liczba kubitów (1 lub 2): "))
    if n_qubits not in (1, 2):
        print("Obsługiwane tylko 1 lub 2 kubity")
        return

    state = read_initial_state(n_qubits)
    if state is None:
        return

    n_gates = int(input("Ile bramek po kolei chcesz zastosować? "))

    for i in range(n_gates):
        print(f"\nBramka {i + 1}/{n_gates}")
        if n_qubits == 1:
            gate = pick_single_gate()
        else:
            kind = input(f"Bramka 2-kubitowa ({'/'.join(TWO_QUBIT_GATES)}) "
                          f"czy iloczyn tensorowy dwóch bramek 1-kubitowych? (2/T): ").strip().upper()
            if kind == "2":
                name = input(f"Wybierz bramkę ({'/'.join(TWO_QUBIT_GATES)}): ").strip().upper()
                if name not in TWO_QUBIT_GATES:
                    raise ValueError(f"Nieznana bramka: {name}")
                gate = TWO_QUBIT_GATES[name]
            else:
                g1 = pick_single_gate("dla 1. kubitu ")
                g2 = pick_single_gate("dla 2. kubitu ")
                gate = np.kron(g1, g2)

        state = gate @ state

    print("\nStan wynikowy:", np.round(state, 5))
    return state


if __name__ == "__main__":
    universal_simulator()
