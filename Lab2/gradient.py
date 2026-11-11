import numpy as np

LOWER_BOUND = -100.0
UPPER_BOUND = 100.0


def gradient(f, x, h=1e-6, bounds=None):

    x = np.asarray(x, dtype=float)
    n = len(x)
    idx = np.arange(n)

    X_plus = np.tile(x[:, None], n)
    X_minus = X_plus.copy()
    krok = h * np.maximum(1.0, np.abs(x))
    X_plus[idx, idx] += krok
    X_minus[idx, idx] -= krok

    if bounds is not None:
        lower, upper = bounds
        X_plus[idx, idx] = np.minimum(X_plus[idx, idx], upper)
        X_minus[idx, idx] = np.maximum(X_minus[idx, idx], lower)

    przedzial = X_plus[idx, idx] - X_minus[idx, idx]
    if np.any(przedzial == 0):
        raise ValueError("nie można wyznaczyć gradientu na podanym przedziale")

    wartosci = np.asarray(f(np.concatenate([X_plus, X_minus], axis=1)), dtype=float)
    return (wartosci[:n] - wartosci[n:]) / przedzial


def solver(
    eval_func,
    wymiar,
    budzet,
    krok,
    rng,
    lower=LOWER_BOUND,
    upper=UPPER_BOUND,
    tol=1e-6,
):
    def f1(punkt):  # wartość funkcji celu w pojedynczym punkcie (1 ewaluacja)
        return float(np.asarray(eval_func(punkt.reshape(-1, 1)))[0])

    x = rng.uniform(lower, upper, size=wymiar)
    fx = f1(x)

    # Historia najlepszej wartości po każdej ewaluacji funkcji celu.
    najlepsza = fx
    historia = [najlepsza]
    powod = "budżet"

    koszt_iteracji = 2 * wymiar + 1

    while len(historia) + koszt_iteracji <= budzet:
        g = gradient(eval_func, x, bounds=(lower, upper))
        historia += [najlepsza] * (2 * wymiar)

        # Rzutowanie gradientu na granice dziedziny
        g[(x <= lower) & (g > 0)] = 0
        g[(x >= upper) & (g < 0)] = 0

        if np.linalg.norm(g) < tol:
            powod = "mały gradient"
            break

        x_nowy = np.clip(x - krok * g, lower, upper)

        if np.all(x_nowy == x):
            powod = "uderzenie w granice dziedziny"
            break

        fx = f1(x_nowy)
        x = x_nowy
        if fx < najlepsza:
            najlepsza = fx
        historia.append(najlepsza)

    return najlepsza, historia, powod
