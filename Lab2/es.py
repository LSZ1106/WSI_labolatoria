"""Strategia ewolucyjna (mu + lambda) z samoadaptacją zasięgu mutacji."""

import numpy as np


def es_mu_plus_lambda(
    funkcja, wymiar, budzet, mu, lam, sigma0, rng,
    lower=-100.0, upper=100.0, max_stagnacja=25,
):
    tau = 1.0 / np.sqrt(2.0 * wymiar)

    # Populacja startowa.
    X = rng.uniform(lower, upper, size=(wymiar, mu))
    sigma = np.full(mu, float(sigma0))
    wart = np.asarray(funkcja(X), dtype=float)

    najlepsza = float(wart.min())
    historia = [najlepsza] * mu
    stagnacja = 0
    powod = "budżet"

    while len(historia) < budzet:
        lam_eff = min(lam, budzet - len(historia))  # nie przekraczamy budżetu

        # Reprodukcja + mutacja (log-normalna samoadaptacja sigmy).
        rodzice = rng.integers(0, mu, size=lam_eff)
        sigma_d = sigma[rodzice] * np.exp(tau * rng.standard_normal(lam_eff))
        X_d = X[:, rodzice] + sigma_d * rng.standard_normal((wymiar, lam_eff))
        X_d = np.clip(X_d, lower, upper)
        wart_d = np.asarray(funkcja(X_d), dtype=float)

        # Sukcesja (mu + lambda): mu najlepszych z rodziców i dzieci.
        X = np.concatenate([X, X_d], axis=1)
        sigma = np.concatenate([sigma, sigma_d])
        wart = np.concatenate([wart, wart_d])
        najlepsi = np.argsort(wart)[:mu]
        X, sigma, wart = X[:, najlepsi], sigma[najlepsi], wart[najlepsi]

        if wart[0] < najlepsza:
            najlepsza = float(wart[0])
            stagnacja = 0
        else:
            stagnacja += 1
        historia += [najlepsza] * lam_eff

        if stagnacja >= max_stagnacja:
            powod = "stagnacja"
            break

    return najlepsza, historia, powod
