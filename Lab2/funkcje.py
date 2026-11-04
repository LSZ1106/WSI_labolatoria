#Funkcje celu: F3 i F18 z benchmarku CEC 2017 oraz funkcja kwadratowa.

import numpy as np

from cecpy.benchmark import CECEdition, CECEvaluator
from cecpy.benchmark.evaluator import problem_optimum_value

LOWER_BOUND = -100.0
UPPER_BOUND = 100.0
WYMIARY = (10, 30)
BUDZET_NA_WYMIAR = 100  # FE_max = 100 * n
LICZBA_POWTORZEN = 10

EDYCJA = CECEdition.CEC2017
cec17 = CECEvaluator(EDYCJA, list(WYMIARY))


# Każda funkcja przyjmuje macierz (wymiar, k) - punkty w kolumnach.
def f3(X):
    return np.asarray(cec17(3, np.asarray(X, dtype=np.float64)))


def f18(X):
    return np.asarray(cec17(18, np.asarray(X, dtype=np.float64)))


def f_kwadratowa(X):
    return np.sum(np.asarray(X, dtype=np.float64) ** 2, axis=0)


# (nazwa, funkcja, wartość optimum)
FUNKCJE = [
    ("CEC2017_F3", f3, problem_optimum_value(EDYCJA, 3)),
    ("CEC2017_F18", f18, problem_optimum_value(EDYCJA, 18)),
    ("kwadratowa", f_kwadratowa, 0.0),
]
