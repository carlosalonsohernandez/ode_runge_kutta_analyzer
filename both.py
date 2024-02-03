import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# start values
x0 = 1
h = 0.02
num = 500
x = x0 + np.arange(0, num) * h # x vals from 0 to 2 with 0.1 increments
y0 = 5
y = []
pre_y = y0


# function to differentiate
def f(x, y):
    return y / (pow(math.e, x) - 1)

# function that returns t4 shown in padlet
def t4(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h / 2 * k1)
    k3 = f(x + h / 2, y + h / 2 * k2)
    k4 = f(x + h, y + h * k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6

# recursive runge_kutta function
def runge_kutta(y, x, h, n):
    # recursively use runge_kutta to get the y value
    if n == 0:
        return y0
    else:
        # return yn + h * t4(xn,yn,h), yn = recursive runge_kutta call
        yn = runge_kutta(y, x - h, h, n - 1)
        return yn + h * t4(x - h, yn, h)


# loop through x values and calculate y values
n = 0
for xval in x:
    pre_y = runge_kutta(pre_y, xval, h, n)
    y.append(pre_y)
    n += 1

# Your existing x and y data


# New set of data to compare
def model(y,x):
    return y / (pow(math.e, x) - 1)


# solve ODE
y2 = odeint(model,y0,x)

# Plotting the first set of data
plt.plot(x, y, label='Runge-Kutta Line')

# Plotting the second set of data
plt.plot(x, y2, label='ODEINT Line')

# Adding labels and title
plt.xlabel('time')
plt.ylabel('y(x)')
plt.title('Comparison of two lines')

# Adding legend to distinguish between the two lines
plt.legend()

# Display the plot
plt.show()