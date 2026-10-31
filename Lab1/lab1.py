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


def solver(eval_func, x0, krok, max_iter=200, tol=1e-6):
    x = np.array(x0, dtype=float).copy()
    lower = np.full_like(x, LOWER_BOUND)
    upper = np.full_like(x, UPPER_BOUND)
    x = np.clip(x, lower, upper)

    fx = float(eval_func(x))
    historia = [fx]

    powod = "limit iteracji"

    for _ in range(max_iter):
        g = gradient(eval_func, x, bounds=(lower, upper))

        # Rzutowanie gradientu na granice dziedziny
        g[(x == lower) & (g > 0)] = 0
        g[(x == upper) & (g < 0)] = 0

        if np.linalg.norm(g) < tol:
            powod = "mały gradient"
            break

        x_nowy = np.clip(x - krok * g, lower, upper)

        if np.allclose(x, x_nowy, atol=1e-8, rtol=1e-8):
            powod = "uderzenie w granice dziedziny"
            break

        #  Obliczenie wartości funkcji dla nowego punktu
        f_nowy = float(eval_func(x_nowy))

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
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        print(f"\nWyniki dla {f.__name__}:")

        for ax, krok in zip(axes.flat, alpha):
            start = perf_counter()
            wynik, historia, powod = solver(f, x, krok)
            czas = perf_counter() - start

            print(
                f"alpha={krok:<6g} f(x)={historia[-1]:.6e} | "
                f"iteracje={len(historia)-1} | czas={czas:.4f} s | stop: {powod}"
            )

            ax.plot(range(len(historia)), historia, color="tab:blue", linewidth=1.5)
            ax.set_title(f"α = {krok:g}")
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.set_xlabel("Numer iteracji t")
            ax.set_ylabel("f(x_t)")
            ax.grid(True, which="both", alpha=0.35)

        fig.suptitle(f"Zbieżność {f.__name__} dla różnych kroków")
        fig.tight_layout()
        fig.savefig(f"{f.__name__}_zbieznosc.png", dpi=150)
        plt.close(fig)
