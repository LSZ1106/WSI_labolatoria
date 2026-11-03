import numpy as np
import matplotlib.pyplot as plt

from cecpy.benchmark.evaluator import CECEvaluator, CECEdition, problem_optimum_value

LOWER_BOUND = -100.0
UPPER_BOUND = 100.0
WYMIARY = (10, 30)
BUDZET_NA_WYMIAR = 10_000  
LICZBA_POWTORZEN = 10    

cec17 = CECEvaluator(CECEdition.CEC2017, list(WYMIARY))


def f_rosen(X):  # CEC2017 F3
    return np.asarray(cec17(3, np.asarray(X, dtype=np.float64)))


def f_ackley(X):  # CEC2017 F18
    return np.asarray(cec17(18, np.asarray(X, dtype=np.float64)))


def f_kwadratowa(X):
    X = np.asarray(X, dtype=np.float64)
    return np.sum(X**2, axis=0)


FUNKCJE = [
    ("rosenbrock_F3", f_rosen, problem_optimum_value(CECEdition.CEC2017, 3)),
    ("ackley_F18", f_ackley, problem_optimum_value(CECEdition.CEC2017, 18)),
    ("kwadratowa", f_kwadratowa, 0.0),
]

