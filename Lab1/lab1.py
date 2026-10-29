from cecpy.functions.unimodal import sphere
from cecpy.functions.multimodal import ackley, rosenbrock
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

alpha = [0.001, 0.01, 0.1, 1, 10, 100]
LOWER_BOUND = -10.0
UPPER_BOUND = 10.0

f_sphere = sphere
f_rosen = rosenbrock
f_ackley = ackley


def gradient(f, x, h=1e-6, bounds=None):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()

        step = h * max(1.0, abs(x[i]))
        x_plus[i] += step
        x_minus[i] -= step

        if bounds is not None:
            lower, upper = bounds
            x_plus[i] = min(x_plus[i], upper[i])
            x_minus[i] = max(x_minus[i], lower[i])

        interval = x_plus[i] - x_minus[i]
        if interval == 0:
            raise ValueError("nie można wyznaczyć gradientu na podanym przedziale")
        g[i] = (f(x_plus) - f(x_minus)) / interval

    return g


def solver(eval_func, x0, krok, max_iter=5000, tol=1e-6):
    x = np.array(x0, dtype=float).copy()
    lower = np.full_like(x, LOWER_BOUND)
    upper = np.full_like(x, UPPER_BOUND)
    x = np.clip(x, lower, upper)

    fx = float(eval_func(x))
    historia = [fx]

    powod = "limit iteracji"

    for _ in range(max_iter):
        g = gradient(eval_func, x, bounds=(lower, upper))

        # warunek 1: mały gradient
        if np.linalg.norm(g) < tol:
            powod = "mały gradient"
            break

        # wykonaj krok gradientowy
        x_nowy = np.clip(x - krok * g, lower, upper)
        f_nowy = float(eval_func(x_nowy))

        # warunek 2: mała zmiana wartości funkcji
        if abs(f_nowy - fx) < 1e-12:
            powod = "mała zmiana f(x)"
            x = x_nowy
            historia.append(f_nowy)
            break

        x = x_nowy
        fx = f_nowy
        historia.append(fx)

    return x, historia, powod


if __name__ == "__main__":
    x = np.random.uniform(-10, 10, size=10)
    print(x)
    for f in (f_sphere, f_rosen, f_ackley):
        print(f.__name__)
        print("f(x):", f(x))
        print("gradient:", gradient(f, x))

    for f in (f_sphere, f_rosen, f_ackley):
        plt.figure(figsize=(9, 5))
        print(f"\nWyniki dla {f.__name__}:")

        for krok in alpha:
            start = perf_counter()
            wynik, historia, powod = solver(f, x, krok)
            czas = perf_counter() - start

            print(
                f"alpha={krok:<6g} f(x)={historia[-1]:.6e} | "
                f"iteracje={len(historia)-1} | czas={czas:.4f} s | stop: {powod}"
            )

            plt.plot(range(len(historia)), historia, label=f"alpha={krok}")

        plt.title(f"Zbieżność — {f.__name__}")
        plt.xlabel("Numer iteracji t")
        plt.ylabel("Wartość funkcji f(x_t)")
        plt.yscale("symlog", linthresh=1e-8)

        plt.legend()
        plt.grid(True, which="both")
        plt.tight_layout()
        plt.savefig(f"{f.__name__}_zbieznosc.png")
        plt.close()
