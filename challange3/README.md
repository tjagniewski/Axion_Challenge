## Hipoteza

Liczba warstw ansatzu L wpływa na accuracy klasyfikatora tak samo jak głębokość sieci w klasycznym ML: istnieje optymalne L, a za mało albo za dużo warstw obniża test accuracy.

### Dlaczego

- **Za mało warstw (L = 1):** model ma mało parametrów (P = 4) i tylko jeden CNOT. Na wykładzie entanglement był odpowiednikiem hidden layer, bo pozwala modelowi widzieć interakcje między features. Przy jednej warstwie tych interakcji jest mało, więc model nie dopasuje się do zakrzywionej granicy make_moons. Dojdzie do underfittingu.
- **Za dużo warstw (L = 8):** liczba layers to odpowiednik depth sieci. Więcej parametrów przy zaszumionym zbiorze 200 punktów oznacza, że model dopasuje się do szumu. Skutkować to będzie overfittingiem.

### Przewidywanie

Wszystkie liczby to średnia z 5 seedów.

- Najwyższa test accuracy wypadnie dla **L = 2 lub L = 3**.
- Test accuracy dla **L = 1** i dla **L = 8** będzie co najmniej **3 pp niższa** od maksimum.


## Plan eksperymentu

- **Dane:** `make_moons`, noise = 0.2. 200 punktów treningowych, 200 testowych. Features skalowane do [0, $\pi$].
- **Obwód:** 2 kubity, angle encoding RY($x_i$). Warstwa = RY($\theta$) na każdym kubicie + CNOT(0->1), na końcu dodatkowa warstwa RY.
- **L:** {1, 2, 3, 4, 5, 6, 7, 8}.
- **Measurement:** ⟨$Z_0$⟩, predykcja = sign(f).
- **Trening:** MSE, Adam (lr = 0.1), 100 epok, init $\theta$ ~ U(−0.1, 0.1), parameter-shift.
- **Symulator:** `StatevectorEstimator`, bez shotów.
- **Seedy:** 5.
- **Mierzę:** train i test accuracy. Wykres accuracy vs L.

