"""Strategia ewolucyjna (mu + lambda) z samoadaptacją zasięgu mutacji."""

import numpy as np


def solver(
    eval_func, wymiar, budzet, mu, lam, sigma0, rng,
    lower=-100.0, upper=100.0, max_stagnacja=25,
):
    tau = 1.0 / np.sqrt(2.0 * wymiar)

    # Populacja startowa.
    X = rng.uniform(lower, upper, size=(wymiar, mu))
    sigma = np.full(mu, float(sigma0))
    fx = np.asarray(eval_func(X), dtype=float)

    # Historia najlepszej wartości po każdej ewaluacji funkcji celu.
    najlepsza = float(fx.min())
    historia = [najlepsza] * mu
    stagnacja = 0
    powod = "budżet"

    while len(historia) < budzet:
        lam_eff = min(lam, budzet - len(historia))  # nie przekraczamy budżetu

        # Reprodukcja + mutacja (log-normalna samoadaptacja sigmy).
        rodzice = rng.integers(0, mu, size=lam_eff)
        sigma_nowe = sigma[rodzice] * np.exp(tau * rng.standard_normal(lam_eff))
        X_nowe = X[:, rodzice] + sigma_nowe * rng.standard_normal((wymiar, lam_eff))
        X_nowe = np.clip(X_nowe, lower, upper)
        f_nowe = np.asarray(eval_func(X_nowe), dtype=float)

        # Sukcesja (mu + lambda): mu najlepszych z rodziców i dzieci.
        X = np.concatenate([X, X_nowe], axis=1)
        sigma = np.concatenate([sigma, sigma_nowe])
        fx = np.concatenate([fx, f_nowe])
        najlepsi = np.argsort(fx)[:mu]
        X, sigma, fx = X[:, najlepsi], sigma[najlepsi], fx[najlepsi]

        if fx[0] < najlepsza:
            najlepsza = float(fx[0])
            stagnacja = 0
        else:
            stagnacja += 1
        historia += [najlepsza] * lam_eff

        if stagnacja >= max_stagnacja:
            powod = "stagnacja"
            break

    return najlepsza, historia, powod
