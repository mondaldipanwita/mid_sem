#import numpy as np
import matplotlib as plt
from math import exp, cos, pi
from scipy.special import roots_laguerre
from scipy.special import roots_legendre

#___________________________________________ INTEGRATIONS_______________________________________


#---------------------Midpoint method--------------------------

def midpoint_integration(f, a, b, n):
    """
    Compute the definite integral of a function using the midpoint method.

    Parameters:
    func (function): The function to integrate.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): The number of subintervals to use.

    Returns:
    float: The approximate value of the definite integral.
    """
    h = (b - a) / n  # Width of each subinterval
    integral = 0
    
    # Compute the midpoint of each subinterval and evaluate the function
    # at that point, then multiply by the width of the interval and sum.
    for i in range(n):
        x_midpoint = a + (i + 0.5) * h
        integral += f(x_midpoint)
    
    integral *= h  # Multiply by the width of each subinterval
    
    return integral

#--------------------------------------------------------------


#--------------------Trapezoidal------------------------------
def trapezoidal_integration(f, a, b, n):
    """
    Compute the definite integral of a function using the trapezoidal rule.

    Parameters:
    func (function): The function to integrate.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): The number of subintervals to use.

    Returns:
    float: The approximate value of the definite integral.
    """
    h = (b - a) / n  # Width of each subinterval
    
    integral = (f(a) + f(b)) / 2
    
    # Compute the integral using the trapezoidal rule
    for i in range(1, n):
        integral += func(a + i * h)
    
    integral *= h
    
    return integral


#---------------------------------------------------------------



#----------------------Simpson1/3--------------------------------
def simpson13(f,x0,xn,n):
    # calculating step size
    h = (xn - x0) / n
    
    # Finding sum 
    integral_sum = f(x0) + f(xn)
    
    for i in range(1, n, 2):
        integral_sum += 4 * f(x0 + i * h)
    for i in range(2, n-1, 2):
        integral_sum += 2 * f(x0 + i * h)
    integral = h / 3 * integral_sum
    return integral

#-----------------------------------------------------------------



#----------------------Gaussian_quadrature(laguerre)-------------

def laguerre_integration(f, n):
    """
    Compute the definite integral of a function using Gaussian quadrature
    with Laguerre polynomials.

    Parameters:
    func (function): The function to integrate.
    n (int): The number of sample points and weights to use.

    Returns:
    float: The approximate value of the definite integral.
    """
    # Compute sample points (nodes) and weights using Laguerre polynomials
    nodes, weights = roots_laguerre(n)

    # Compute the integral using Gaussian quadrature formula
    integral = 0
    for i in range(n):
        integral += weights[i] * f(nodes[i])

    return integral

#--------------------------------------------------------------


#_____________________________________ROOT_FINDINGS_______________________________________________


#------------------------Bisection method----------------------
def bisection_method(func, a, b, tol=1e-6):
    """
    Find a root of a continuous function using the bisection method.

    Parameters:
    func (function): The continuous function.
    a (float): The left endpoint of the interval.
    b (float): The right endpoint of the interval.
    tol (float): Tolerance for the root.
    max_iter (int): Maximum number of iterations.

    Returns:
    float or None: Approximation of the root if found within tolerance, None otherwise.
    """
    if func(a) * func(b) >= 0:
        print("The function has the same sign at both endpoints. Bisection method cannot be applied.")
        return None

    # Initialize iteration count and the initial midpoint
    i = 0
    c = (a + b) / 2

    # Perform bisection iterations
    while  (b - a)/2 > tol:
        if func(c) == 0:
            print(f"Root found exactly at iteration {i+1}: {c}")
            return c  # Found the root exactly
        elif func(c) * func(a) < 0:
            b = c
        else:
            a = c
        c = (a + b) / 2
        i += 1
        print(f"Iteration {i}: Current approximation = {c}")

    if (b - a)/2 <= tol:
        print(f"Root found within tolerance at iteration {i+1}: {c}")
        return c  # Found the root within tolerance

    print("Maximum number of iterations reached. No root found within tolerance.")
    return None

          
#------------------------------------------------------------------


#---------------------Regula_Falsi------------------------------------

# Regula Falsi method
def regula_falsi(a, b, tol=1e-4, max_iter=1000):
    iterations = 0
    if f(a) * f(b) >= 0:
        raise ValueError("The function does not change sign over the interval [a, b]")
    
    while abs(b - a) > tol :
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if f(c) == 0:
            return c, iterations
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iterations += 1
        
    return (a * f(b) - b * f(a)) / (f(b) - f(a)), iterations  # Return the midpoint if the tolerance is not met


#---------------------------------------------------------------



#-----------------------Newton_raphson_single_variable-------------
def newton_raphson(f, derivative, x0, tol=1e-6):
    """
    Find a root of a function using the Newton-Raphson method.

    Parameters:
    func (function): The function for which to find the root.
    derivative (function): The derivative of the function.
    x0 (float): Initial guess for the root.
    tol (float): Tolerance for the root.
    max_iter (int): Maximum number of iterations.

    Returns:
    float or None: Approximation of the root if found within tolerance, None otherwise.
    """
    i=0
    x = x0
    x_new= x - f(x) / derivative(x)
    while abs(x-x_new)>tol:
        x = x_new
        x_new = x - f(x) / derivative(x)
        i+=1
        if abs(x_new - x) < tol:
            print(f"Root found within tolerance at iteration {i+1}: {x_new}")
       
    return x_new  # Found the root within tolerance
        
    print(f"Iteration {i+1}: Current approximation = {x}")

    print("Maximum number of iterations reached. No root found within tolerance.")
    return None


 #-----------------------------------------------------------------


#-------------------------Netwon_raphson_multivaiable-----------------

def jacobian_inv(df1_dx, df1_dy, df2_dx, df2_dy, x, y):
    # Jacobian matrix
    J = df1_dx(x, y), df1_dy(x, y), df2_dx(x, y), df2_dy(x, y)
    determinant = J[0] * J[3] - J[1] * J[2]
    
    # Check for singularity
    if determinant == 0:
        raise ValueError("Jacobian determinant is zero. Newton-Raphson method cannot proceed.")
    
    # Inverse of Jacobian matrix
    J_inv = (J[3] / determinant, -J[1] / determinant, -J[2] / determinant, J[0] / determinant)
    return J_inv

def newton_raphson_system(f1, f2, df1_dx, df1_dy, df2_dx, df2_dy, x0, y0, tol=1e-6, max_iter=100):
    iteration = 0
    while True:
        fx1 = f1(x0, y0)
        fx2 = f2(x0, y0)
        J_inv = jacobian_inv(df1_dx, df1_dy, df2_dx, df2_dy, x0, y0)
        
        dx = J_inv[0] * fx1 + J_inv[1] * fx2
        dy = J_inv[2] * fx1 + J_inv[3] * fx2
        
        x0 += dx
        y0 += dy
        
        iteration += 1
        
        if abs(f1(x0, y0)) < tol and abs(f2(x0, y0)) < tol or iteration >= max_iter:
            break
    
    return x0, y0
#--------------------------------------------------------------------


#---------------------Secant_method------------------------------

def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    iteration = 0
    while True:
        fx0 = f(x0)
        fx1 = f(x1)
        
        if abs(fx1) < tol:
            return x1
        
        if iteration >= max_iter:
            print("Maximum iterations reached.")
            return None
        
        if fx1 - fx0 == 0:
            print("Division by zero error.")
            return None
        
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        
        x0 = x1
        x1 = x_new
        
        iteration += 1

#-----------------------------------------------------------------


#---------------------Fixed_point-------------------------------
def fixed_point(g, x0, error=1e-4):
    i=0
    x = x0
    x_i = g(x)
    while (abs((x_i - x)/x) > error):
        x = x_i
        x_i = g(x)
        i+=1
        print(x_i)
    # Return the root and the number of iterations    
    return x_i,i
#--------------------------------------------------------------

#---------------multivariables:Fixed_point--------------------
def fixed_point_iteration(g, x0, tol=1e-6):
    """
    Solve a system of equations using the fixed-point iteration method.

    Parameters:
    g (function): The function defining the fixed-point iteration.
    x0 (list or tuple): Initial guess for the solution vector.
    tol (float): Tolerance for convergence. Defaults to 1e-6.
    max_iter (int): Maximum number of iterations. Defaults to 1000.

    Returns:
    list or None: Solution vector if convergence is achieved within tolerance, None otherwise.
    """
    x = list(x0)
    x_new=g(x)
    i=0
    diff= [p-q for p, q in zip(x, x_new)]
    error=np.sqrt(np.sum(np.square(diff)))/(np.sqrt(np.sum(np.square(x))))
    # print(error)
    # print(x,x_new,error)
    while error > tol:
        x = x_new

        x_new = g(x)
        
        if all(abs(x_new[i] - x[i]) < tol for i in range(len(x))):
            return x_new  # Convergence achieved
        error=np.sqrt(np.sum(np.square(diff)))/(np.sqrt(np.sum(np.square(x))))
        i+=1
    print("Maximum number of iterations reached. Convergence not achieved.")
    return x_new


#-------------------------------------------------------------




#_________________________DIFFERENTIAL_EQUATIONS_______________________________________


#----------------RK4------------------------------------------
def rk4(x, y, u, h):
    k1 = h * f(x, y, u)
    k2 = h * f(x + h/2, y + k1[0]/2, u + k1[1]/2)
    k3 = h * f(x + h/2, y + k2[0]/2, u + k2[1]/2)
    k4 = h * f(x + h, y + k3[0], u + k3[1])
    return y + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6, u + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6

def rk4_step(x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def solve_rk4(x0, y0, xf, h):
    steps = int((xf - x0) / h)
    x_values = np.linspace(x0, xf, steps + 1)
    y_values = [y0]
    for i in range(steps):
        y_values.append(rk4_step(x_values[i], y_values[-1], h))
    return x_values, y_values

#----------------------------------------------------------------------


#--------------------------Shooting_method-----------------------------
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

#-----------------------------------------------------------------------


#---------------------------Euler_method--------------------------------
# Define the ODE function
def f(x, y):
    return -y

# Forward Euler method
def forward_euler(f, x0, y0, h, x_end):
    x_values = [x0]
    y_values = [y0]
    while x_values[-1] < x_end:
        x_new = x_values[-1] + h
        y_new = y_values[-1] + h * f(x_values[-1], y_values[-1])
        x_values.append(x_new)
        y_values.append(y_new)
    return x_values, y_values

# Backward Euler method (using implicit approach)
def backward_euler(f, x0, y0, h, x_end):
    x_values = [x0]
    y_values = [y0]
    while x_values[-1] < x_end:
        x_new = x_values[-1] + h
        # Backward Euler formula (implicit)
        y_new = (y_values[-1] + h * f(x_values[-1], y_values[-1])) / (1 + h)
        x_values.append(x_new)
        y_values.append(y_new)
    return x_values, y_values


#-----------------------------------------------------------------------


#-----------------------Leap_frog--------------------------------------


# Define the ODE function
def f(x, y):
    return -y

# Leapfrog method
def leapfrog(f, x0, y0, h, x_end):
    x_values = [x0]
    y_values = [y0]
    # Use forward Euler to get the second point
    x_values.append(x0 + h)
    y_values.append(y0 + h * f(x0, y0))
    # Perform leapfrog iterations
    while x_values[-1] < x_end:
        x_new = x_values[-1] + h
        # Use the midpoint method to compute the next point
        y_new = y_values[-2] + 2 * h * f(x_values[-1], y_values[-1])
        x_values.append(x_new)
        y_values.append(y_new)
    return x_values, y_values


#----------------------------------------------------------------------


#----------------------------velocity_verlet---------------------------

# Velocity Verlet method
def velocity_verlet(x0, v0, dt, tf):
    t_values = np.arange(t0, tf, dt)
    x_values = np.zeros_like(t_values)
    v_values = np.zeros_like(t_values)
    
    # Initial conditions
    x_values[0] = x0
    v_values[0] = v0
    
    for i in range(1, len(t_values)):
        # Update position
        x_half = x_values[i-1] + v_values[i-1] * dt / 2
        v_values[i] = v_values[i-1] + acceleration(x_half) * dt
        
        # Update position
        x_values[i] = x_values[i-1] + v_values[i] * dt
        
    return t_values, x_values, v_values


# # Plot position vs. time
# plt.figure(figsize=(10, 6))
# plt.plot(t_values, x_values, label='Position (Velocity Verlet)')
# plt.plot(t_values, v_values, label='Velocity (Velocity Verlet)')
# plt.xlabel('Time')
# plt.ylabel('Position')
# plt.title('Position vs. Time (Harmonic Oscillator)')
# plt.grid(True)
# plt.legend()
# plt.show()
# #----------------------------------------------------------------------


#-------------------Forward_difference:Heat_equation---------------

#-------------------------------------------------------------------



#------------------Crank Nicolson----------------------------------
def crank_nicolson_heat_equation(u_initial, dx, dt, T, alpha):
    """
    Solve the heat equation using Crank-Nicolson method.
    
    Parameters:
        u_initial: array_like
            Initial condition.
        dx: float
            Spatial step size.
        dt: float
            Temporal step size.
        T: float
            Total time to evolve the system.
        alpha: float
            Stability parameter.
    
    Returns:
        u: array_like
            Solution to the heat equation.
    """
    N = len(u_initial) - 1  # Number of spatial segments
    M = int(T / dt)          # Number of time steps

    # Initialize solution matrix
    u = np.zeros((N+1, M+1))
    u[:,0] = u_initial
    
    # Construct the tridiagonal matrix
    A_diag = (2 + 2 * alpha) * np.ones(N-1)
    A_upper = -alpha * np.ones(N-2)
    A_lower = -alpha * np.ones(N-2)

    print(A_diag,A_upper,A_lower)
    
    ###A = np. diag(A_diag) + np.diag(A_upper, k=1) + np.diag(A_lower, k=-1)
    
    def create_tridiagonal_matrix(A_diag, A_upper, A_lower):
        n = len(A_diag)
        tridiagonal_matrix = [[0] * n for _ in range(n)]
    
        # Fill the main diagonal
        for i in range(n):
            tridiagonal_matrix[i][i] = A_diag[i]
    
        # Fill the upper diagonal
        for i in range(n-1):
            tridiagonal_matrix[i][i+1] = A_upper[i]
    
        # Fill the lower diagonal
        for i in range(1, n):
            tridiagonal_matrix[i][i-1] = A_lower[i-1]
        A=tridiagonal_matrix
    
    return A                    # tridiagonal matrix
    
    for j in range(1, M+1):
        b = np.zeros(N-1)
        b[0] = alpha * u[0,j-1]
        b[-1] = alpha * u[N,j-1]
        
        # Solve the system of equations using matrix inversion
        u_interior = np.linalg.solve(A, alpha * u[1:N,j-1] + (2 - 2 * alpha) * u[1:N,j-1] + alpha * u[1:N,j-1] + b)
        
        # Update solution matrix
        u[1:N, j] = u_interior
    
    return u


# # Parameters
# L = 8.0         # Length of the rod
# Nx = 200         # Number of spatial segments
# dx = L / Nx     # Spatial step size
# T = 5        # Total time
# dt = 1      # Temporal step size
# alpha = dt / (2 * dx**2)   # Stability parameter

# # Initial condition
# x_values = np.linspace(0, L, Nx+1)
# u_initial = 4 * x_values - 0.5 * x_values**2

# # Solve the heat equation
# u_solution = crank_nicolson_heat_equation(u_initial, dx, dt, T, alpha)




#-----------------------------------------------------------------------------


#--------------------------2D_poisson_eqation---------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx = 1.0  # Length of domain in the x-direction
Ly = 1.0  # Length of domain in the y-direction
Nx = 50   # Number of grid points in the x-direction
Ny = 50   # Number of grid points in the y-direction
dx = Lx / (Nx - 1)  # Grid spacing in the x-direction
dy = Ly / (Ny - 1)  # Grid spacing in the y-direction

# Define source term function
def source_term(x, y):
    return np.exp(-(x - 0.5)**2 - (y - 0.5)**2)

# Initialize solution
u = np.zeros((Ny, Nx))

# Define boundary conditions
u[:, 0] = 0.0  # Left boundary
u[:, -1] = 0.0  # Right boundary
u[0, :] = 0.0  # Bottom boundary
u[-1, :] = 0.0  # Top boundary

# Gauss-Seidel solver
def gauss_seidel(u, f, dx, dy, tol=1e-6, max_iter=1000):
    Ny, Nx = u.shape
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        max_diff = 0.0
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                u_old = u[i, j]
                u[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - dx**2 * f[i, j])
                max_diff = max(max_diff, abs(u[i, j] - u_old))
        if max_diff < tol:
            break
    return u, iter_count

# Construct the source term
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
f = source_term(X, Y)

# Solve using Gauss-Seidel method
u, iterations = gauss_seidel(u, f, dx, dy)


#-----------------------------------------------------------------------------



#______________________________________MATRICES____________________________________________

#-------------------------Inverse:GaussJordan----------------------------
def gauss_jordan(matrix):
    """
    Compute the inverse of a matrix using Gauss-Jordan elimination.

    Parameters:
    matrix (list of lists): The matrix to invert.

    Returns:
    list of lists or None: The inverse of the matrix if it exists, None otherwise.
    """
    n = len(matrix)
    augmented_matrix = [row[:] + [int(i == j) for j in range(n)] for i, row in enumerate(matrix)]

    for i in range(n):
        if augmented_matrix[i][i] == 0:
            return None  # Matrix is singular

        pivot = augmented_matrix[i][i]
        augmented_matrix[i] = [elem / pivot for elem in augmented_matrix[i]]

        for j in range(n):
            if j != i:
                factor = augmented_matrix[j][i]
                augmented_matrix[j] = [elem - factor * augmented_matrix[i][k] for k, elem in enumerate(augmented_matrix[j])]

    inverse_matrix = [row[n:] for row in augmented_matrix]
    return inverse_matrix

#----------------------------------------------------------------------



#------------------------solution of linear equations:Gauss Jordan---------------------------

def gauss_jordan_linear(A, b):
    """
    Solves the system of linear equations Ax = b using Gauss-Jordan elimination.
    
    Parameters:
        A: numpy.ndarray
            Coefficient matrix of shape (n, n).
        b: numpy.ndarray
            Constant vector of shape (n,).   
    Returns:
        numpy.ndarray
            Solution vector of shape (n,).
    """

    # Convert A and b to float dtype
    A = A.astype(float)
    b = b.astype(float)

    
    # Combine the coefficient matrix A and the constant vector b into an augmented matrix
    aug_matrix = np.column_stack((A, b))
    
    # Get the number of rows and columns in the augmented matrix
    num_rows, num_cols = aug_matrix.shape
    
    # Perform forward elimination
    for pivot_row in range(num_rows):
        # Normalize the pivot row so that the pivot element becomes 1
        pivot_element = aug_matrix[pivot_row, pivot_row]
        aug_matrix[pivot_row, :] /= pivot_element
        
        # Eliminate all other entries in the pivot column
        for row in range(num_rows):
            if row != pivot_row:
                multiplier = aug_matrix[row, pivot_row]
                aug_matrix[row, :] -= multiplier * aug_matrix[pivot_row, :]
    
    # Perform back substitution
    for pivot_row in reversed(range(num_rows)):
        # Eliminate all entries above the pivot element
        for row in range(pivot_row):
            multiplier = aug_matrix[row, pivot_row]
            aug_matrix[row, :] -= multiplier * aug_matrix[pivot_row, :]
    
    # The last column of the augmented matrix contains the solution vector
    solution = aug_matrix[:, -1]
    
    return solution

#-----------------------------------------------------------------


#---------------Inverse_matrix:LU_decomposition---------------------
def lu_decomposition(A):
    """
    Perform LU decomposition of a square matrix.

    Parameters:
    A (list of lists): The square matrix to decompose.

    Returns:
    (list of lists, list of lists): Lower triangular matrix, Upper triangular matrix.
    """
    n = len(A)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]

    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            sum_ = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_

        # Lower Triangular
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum_ = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - sum_) / U[i][i]

    return L, U

def solve_equations(L, U, b):
    """
    Solve a system of linear equations Ax = b using LU decomposition.

    Parameters:
    L (list of lists): Lower triangular matrix.
    U (list of lists): Upper triangular matrix.
    b (list): Right-hand side vector.

    Returns:
    list: Solution vector.
    """
    n = len(b)
    y = [0] * n
    x = [0] * n

    # Solve Ly = b for y
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    # Solve Ux = y for x
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]

    return x

def inverse_matrix(A):
    """
    Find the inverse of a square matrix using LU decomposition.

    Parameters:
    A (list of lists): The square matrix to invert.

    Returns:
    list of lists or None: The inverse of the matrix if it exists, None otherwise.
    """
    n = len(A)
    identity = [[int(i == j) for j in range(n)] for i in range(n)]

    L, U = lu_decomposition(A)
    inverse = []

    for col in identity:
        x = solve_linear_equations(L, U, col)
        inverse.append(x)

    return list(zip(*inverse))  # Transpose the result


#------------------------------------------------------------------



#-----------------Solve_linear_equation:LU_decomposition---------
def lu_decomposition_linear(A):
    """
    Perform LU decomposition of a square matrix.

    Parameters:
    A (list of lists): The square matrix to decompose.

    Returns:
    (list of lists, list of lists): Lower triangular matrix, Upper triangular matrix.
    """
    n = len(A)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]

    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            sum_ = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_

        # Lower Triangular
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum_ = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - sum_) / U[i][i]

    return L, U

def forward_substitution_LU(L, b):
    """
    Perform forward substitution to solve a lower triangular system of equations.

    Parameters:
    L (list of lists): Lower triangular matrix.
    b (list): Right-hand side vector.

    Returns:
    list: Solution vector.6
    """
    n = len(b)
    x = [0] * n

    for i in range(n):
        x[i] = (b[i] - sum(L[i][j] * x[j] for j in range(i))) / L[i][i]

    return x

def backward_substitution_LU(U, y):
    """
    Perform backward substitution to solve an upper triangular system of equations.

    Parameters:
    U (list of lists): Upper triangular matrix.
    y (list): Right-hand side vector.

    Returns:
    list: Solution vector.
    """
    n = len(y)
    x = [0] * n

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]

    return x

def solve_linear_equations_LU(A, b):
    """
    Solve a system of linear equations Ax = b using LU decomposition.

    Parameters:
    A (list of lists): Coefficient matrix.
    b (list): Right-hand side vector.

    Returns:
    list: Solution vector.
    """
    L, U = lu_decomposition(A)
    y = forward_substitution_LU(L, b)
    x = backward_substitution_LU(U, y)
    return x


#-------------------------------------------------------------------





#--------------linear_equations:Cholesky_decomposition------------
def is_symmetric(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True


def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition of a symmetric positive definite matrix.

    Parameters:
    A (list of lists): The symmetric positive definite matrix to decompose.

    Returns:
    list of lists: Lower triangular matrix.
    """
    # if not is_symmetric(matrix):
    #     print("Matrix is not symmetric. Cholesky decomposition is not applicable.")
    #     return None
    n = len(A)
    L = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i+1):
            if i == j:
                temp_sum = sum(L[i][k] ** 2 for k in range(j))
                L[i][j] = (A[i][j] - temp_sum) ** 0.5
            else:
                temp_sum = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = (A[i][j] - temp_sum) / L[j][j]

    return L

def forward_substitution_ch(L, b):
    """
    Perform forward substitution to solve a lower triangular system of equations.

    Parameters:
    L (list of lists): Lower triangular matrix.
    b (list): Right-hand side vector.

    Returns:
    list: Solution vector.
    """
    n = len(b)
    x = [0] * n

    for i in range(n):
        x[i] = (b[i] - sum(L[i][j] * x[j] for j in range(i))) / L[i][i]

    return x

def backward_substitution_ch(L_transpose, y):
    """
    Perform backward substitution to solve an upper triangular system of equations.

    Parameters:
    L_transpose (list of lists): Transpose of the lower triangular matrix.
    y (list): Right-hand side vector.

    Returns:
    list: Solution vector.
    """
    n = len(y)
    x = [0] * n

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L_transpose[i][j] * x[j] for j in range(i+1, n))) / L_transpose[i][i]

    return x

def solve_linear_equations_ch(A, b):
    """
    Solve a system of linear equations Ax = b using Cholesky decomposition.

    Parameters:
    A (list of lists): Coefficient matrix.
    b (list): Right-hand side vector.
*/
    Returns:
    list: Solution vector.

   """
    if is_symmetric(A):
        L = cholesky_decomposition(A)
    L_transpose = list(zip(*L))
    y = forward_substitution_ch(L, b)
    x = backward_substitution_ch(L_transpose, y)
    return x


def inverse_from_cholesky(L):
    n = len(L)
    A_inv = np.zeros_like(L)
    for i in range(n):
        A_inv[i, i] = 1 / L[i, i]
        for j in range(i+1, n):
            A_inv[i, j] = -np.dot(A_inv[i, :i], L[j, :i]) / L[j, j]
    return np.dot(A_inv, A_inv.T)
#----------------------------------------------------------------


#-----------------Gauss_Jacobi-----------------------------------
def gauss_jacobi(A, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Solve a system of linear equations Ax = b using Gauss-Jacobi method.

    Parameters:
    A (list of lists): Coefficient matrix.
    b (list): Right-hand side vector.
    x0 (list, optional): Initial guess for the solution. Defaults to None.
    tol (float, optional): Tolerance for convergence. Defaults to 1e-6.
    max_iter (int, optional): Maximum number of iterations. Defaults to 1000.

    Returns:
    list or None: Solution vector if convergence is achieved within tolerance, None otherwise.
    """
    n = len(b)
    x = x0 or [0] * n
    x_new = x.copy()

    for _ in range(max_iter):
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i][i]

        if all(abs(x_new[i] - x[i]) < tol for i in range(n)):
            return x_new  # Convergence achieved

        x = x_new.copy()

    print("Maximum number of iterations reached. Convergence not achieved.")
    return None

#----------------------------------------------------------------


#--------------------Gauss_seidel-------------------------------
def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Solve a system of linear equations Ax = b using Gauss-Seidel method.

    Parameters:
    A (list of lists): Coefficient matrix.
    b (list): Right-hand side vector.
    x0 (list, optional): Initial guess for the solution. Defaults to None.
    tol (float, optional): Tolerance for convergence. Defaults to 1e-6.
    max_iter (int, optional): Maximum number of iterations. Defaults to 1000.

    Returns:
    list or None: Solution vector if convergence is achieved within tolerance, None otherwise.
    """
    n = len(b)
    x = x0 or [0] * n

    for _ in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            sigma = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i][i]

        if all(abs(x_new[i] - x[i]) < tol for i in range(n)):
            return x_new  # Convergence achieved

        x = x_new

    print("Maximum number of iterations reached. Convergence not achieved.")
    return None

#----------------------------------------------------------------







#______________________MISCELLANEOUS_____________________________________

#_---------------------mat_mul-------------------------------
def matrix_multiplication(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Number of columns in matrix1 must equal the number of rows in matrix2")

    result = [[0] * len(matrix2[0]) for _ in range(len(matrix1))]

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result
#------------------------------------------------------------


#---------------------matrix_input_without_numpy--------------
def input_matrix(rows, columns):
    matrix = []
    print("Enter the elements of the matrix row-wise:")
    for i in range(rows):
        row = []
        for j in range(columns):
            element = float(input(f"Enter element for row {i+1}, column {j+1}: "))
            row.append(element)
        matrix.append(row)
    return matrix

#------------------------------------------------------------


#___________________________plots_____________________________

# # Plot results
# plt.figure(figsize=(10, 6))
# plt.plot(x_forward, y_forward, label='Forward Euler')
# plt.plot(x_backward, y_backward, label='Backward Euler')
# plt.plot(x_exact, y_exact, label='Exact Solution', linestyle='--')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Numerical Solutions of the ODE dy/dx = -y')
# plt.legend()
# plt.grid(True)
# plt.show()

