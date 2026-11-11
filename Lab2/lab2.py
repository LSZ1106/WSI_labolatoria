"""Lab2: badanie zbieżności ES(mu+lambda) i porównanie z gradientem prostym."""

import numpy as np
import matplotlib.pyplot as plt

from funkcje import FUNKCJE, WYMIARY, BUDZET_NA_WYMIAR, LICZBA_POWTORZEN
from es import solver as solver_es
from gradient import solver as solver_gd
from pathlib import Path

KATALOG = Path(__file__).parent

# Badane parametry: (mu, lambda, sigma0) dla ES oraz kroki dla gradientu.
KONFIGURACJE_ES = [(3, 6, 30.0), (10, 20, 30.0), (5, 10, 1.0), (5, 10, 10.0)]
KROKI_GD = [0.001, 0.01, 0.1]


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
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "figure.titlesize": 22,
        }
    )

    for nazwa, funkcja, optimum in FUNKCJE:
        # Jeden wykres na funkcję: po jednym panelu dla każdej wymiarowości,
        # na każdym panelu krzywe ES i gradientu.
        fig, osie = plt.subplots(
            1,
            len(WYMIARY),
            figsize=(14, 5.5),
            squeeze=False,
            sharey=True,
        )

        for ax, wymiar in zip(osie[0], WYMIARY):
            budzet = BUDZET_NA_WYMIAR * wymiar
            print(f"\n {nazwa}, n={wymiar}, FE_max={budzet}")

            for mu, lam, sigma0 in KONFIGURACJE_ES:
                srednia, koncowe = usrednij(
                    lambda rng: solver_es(
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

            for krok in KROKI_GD:
                srednia, koncowe = usrednij(
                    lambda rng: solver_gd(funkcja, wymiar, budzet, krok, rng),
                    budzet,
                )
                ax.plot(
                    np.maximum(srednia - optimum, 1e-12),
                    linestyle="--",
                    label=f"GD, α={krok:g}",
                )
                print(
                    f"GD, α={krok:g}: "
                    f"f_best = {koncowe.mean():.4e} ± {koncowe.std():.2e}"
                )

            ax.set_yscale("log")
            ax.tick_params(labelleft=True)
            ax.set_xlabel("Liczba ewaluacji funkcji celu")
            ax.set_ylabel("Różnica względem optimum")
            ax.set_title(f"n={wymiar}")
            ax.grid(True, which="both", alpha=0.35)
            ax.legend(loc="upper right")
            # ax.legend()

        # fig.suptitle(f"Zbieżność ES i gradientu, {nazwa}")
        fig.tight_layout()
        fig.savefig(KATALOG / f"zbieznosc_{nazwa}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
