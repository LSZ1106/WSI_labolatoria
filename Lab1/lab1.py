from cecpy.functions.unimodal import sphere
from cecpy.functions.multimodal import ackley, rosenbrock
import numpy as np
from autograd import grad

alpha = [0.001, 0.01, 0.1, 1, 10, 100]

f_sphere = sphere
f_rosen = rosenbrock
f_ackley = ackley

x = np.random.uniform(-10, 10, size=10)
print(x)
print(f_sphere(x))
print(f_rosen(x))
print(f_ackley(x))

