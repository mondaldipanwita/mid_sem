##Mid_sem
##Name: Dipanwita Mondal
##Roll No: 23124006
import numpy as np
import my_library as lib
def rk4(x, y, u, h):
    k1 = h * f(x, y, u)
    k2 = h * f(x + h/2, y + k1[0]/2, u + k1[1]/2)
    k3 = h * f(x + h/2, y + k2[0]/2, u + k2[1]/2)
    k4 = h * f(x + h, y + k3[0], u + k3[1])
    return y + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6, u + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6

def solve_bvp(a, b, ya, yb, u_guess, h=0.1, tol=1e-6, max_iter=1000):
    u = u_guess
    for _ in range(max_iter):
        x_values = np.arange(a, b + h, h)
        y_values = np.zeros_like(x_values)
        u_values = np.zeros_like(x_values)
        y_values[0] = ya
        u_values[0] = u
        for i in range(len(x_values) - 1):
            y_values[i + 1], u_values[i + 1] = rk4(x_values[i], y_values[i], u_values[i], h)
        residual = y_values[-1] - yb
        if abs(residual) < tol:
            return x_values, y_values
        u -= residual * 0.1  # Adjust u_guess based on the residual
    raise ValueError("Shooting method did not converge")

def f(x, y, u):
    return np.array([u, 0.01*(y-20)]) 

# Define boundary conditions
a = 0
b = 10
ya = 40                         # at x=0,T=40C
yb = 200                        # at x=10m, T=200C

# Initial guess for u(0)
u_guess = 0.5

# Solve the boundary value problem
x, y = solve_bvp(a, b, ya, yb, u_guess)

# Print the solution
print("x values:", x)
print("y values:", y)


#-------------------output-----------------
#from the output x=4.5m for T=100C
