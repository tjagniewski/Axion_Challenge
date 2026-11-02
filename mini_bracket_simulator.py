import numpy as np

# v odpowiada za wprowadzany stan, a b za bramkę
# funkcja dostosowany dla 1-kubitowych stanów
def mini_simulator(v, b):
    if abs(sum(v)) != 1 or v.shape != (2,):
        return(print("Nieprawidłowy stan"))
    if b.shape != (2, 2):
        return(print("Nieprawidłowa bramka"))
    else:
        return v.T @ b

ket_0 = np.array([1, 0])
X = np.array([[0, 1],
             [1, 0]])
Y = np.array([[0, -1j],
             [1j, 0]])
Z = np.array([[1, 0],
             [0, -1]])
H = (1/(2**(1/2)))*np.array([[1, 1],
                            [1, -1]])
def Ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s],
                     [s,  c]])

def Rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s],
                     [-1j * s, c]])

def Rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]])


print(mini_simulator(ket_0, Ry(np.pi / 2)))

print(mini_simulator(ket_0, X))

# funkcja dostosowana dla 2-kubitowych stanów 
def mini_simulator2(v, b):
    if abs(sum(v)) != 1 or v.shape != (4,):
        return(print("Nieprawidłowy stan"))
    if b.shape != (4, 4):
        return(print("Nieprawidłowa bramka"))
    else:
        return v.T @ b

ket_00 = np.array([1, 0, 0, 0])

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

XZ = np.kron(X, Z)  

print(mini_simulator2(ket_00, CNOT))
print(mini_simulator2(ket_00, XZ))