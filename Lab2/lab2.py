"""Lab2: badanie zbieżności ES(mu+lambda)"""

import numpy as np
import matplotlib.pyplot as plt

from funkcje import FUNKCJE, WYMIARY, BUDZET_NA_WYMIAR, LICZBA_POWTORZEN
from es import es_mu_plus_lambda

# Badane parametry: (mu, lambda, sigma0) dla ES oraz kroki dla gradientu.
KONFIGURACJE_ES = [(3, 6, 30.0), (10, 20, 30.0), (5, 10, 1.0), (5, 10, 10.0)]


def usrednij(uruchom, budzet):
    """Uruchamia solver LICZBA_POWTORZEN razy i uśrednia krzywe zbieżności."""
    krzywe, koncowe = [], []
    for i in range(LICZBA_POWTORZEN):
        rng = np.random.default_rng(1000 + i)
        najlepsza, historia, _ = uruchom(rng)
        # Po wcześniejszym stopie przedłużamy ostatnią wartość do końca budżetu.
        historia += [historia[-1]] * (budzet - len(historia))
        krzywe.append(historia)
        koncowe.append(najlepsza)
    return np.mean(krzywe, axis=0), np.array(koncowe)


def main():
    for wymiar in WYMIARY:
        budzet = BUDZET_NA_WYMIAR * wymiar

        for nazwa, funkcja, optimum in FUNKCJE:
            print(f"\n {nazwa}, n={wymiar}, FE_max={budzet} ")
            fig, ax = plt.subplots(figsize=(8, 5.5))

            for mu, lam, sigma0 in KONFIGURACJE_ES:
                srednia, koncowe = usrednij(
                    lambda rng: es_mu_plus_lambda(
                        funkcja, wymiar, budzet, mu, lam, sigma0, rng
                    ),
                    budzet,
                )
                ax.plot(
                    np.maximum(srednia - optimum, 1e-12),
                    label=f"ES({mu}+{lam}), σ0={sigma0:g}",
                )
                print(
                    f"ES({mu}+{lam}), σ0={sigma0:g}: "
                    f"f_best = {koncowe.mean():.4e} ± {koncowe.std():.2e}"
                )

            

if __name__ == "__main__":
    main()
