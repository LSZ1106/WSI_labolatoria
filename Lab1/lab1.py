from cecpy.functions.unimodal import sphere
from cecpy.functions.multimodal import ackley, rosenbrock
import numpy as np

alpha = [0.001, 0.01, 0.1, 1, 10, 100]


f_sphere = sphere
f_rosen = rosenbrock
f_ackley = ackley


def gradient(f, x, h=1e-6):
    g = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += h
        x_minus[i] -= h

        g[i] = (f(x_plus) - f(x_minus)) / (2 * h)

    return g


if __name__ == "__main__":
    x = np.random.uniform(-10, 10, size=10)
    print(x)
    for f in (f_sphere, f_rosen, f_ackley):
        print(f.__name__)
        print("f(x):", f(x))
        print("gradient:", gradient(f, x))
