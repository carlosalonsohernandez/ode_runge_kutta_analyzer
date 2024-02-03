import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dx
def model(y,x):
    return y / (pow(math.e, x) - 1)

y0 = 5

# time points
x0 = 1
h = 0.02
num = 500
x = x0 + np.arange(0, num) * h

# solve ODE
y = odeint(model,y0,x)

# plot results
print(x, y)
plt.plot(x,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()
