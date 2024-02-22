##Mid_sem
##Name: Dipanwita Mondal
##Roll No: 23124006



import my_library as lib
import math
import numpy as np

def f(x):
    return math.log(x/2)-math.sin(5*x/2)
def derivative():
    return 2/x-5/2*math.cos(5*x/2)

#----------------Newton_Raphson----------------------------------------
def newton_raphson(f, derivative, x0, tol=1e-6):
    i=0
    x = x0
    x_new= x - f(x) / derivative(x)
    while abs(x-x_new)>tol:
        x = x_new
        x_new = x - f(x) / derivative(x)
        i+=1
        if abs(x_new - x) < tol:
            print(f"Root found within tolerance at iteration for Newton_raphson {i+1}: {x_new}")
       
    return x_new  # Found the root within tolerance
        
    print(f"Iteration {i+1}: Current approximation = {x}")

    print("Maximum number of iterations reached. No root found within tolerance.")
    return None


# Define the function and its derivative
def f(x):
    return math.log(x/2)-math.sin(5*x/2)

def derivative(x):
    return  2/x-5/2*math.cos(5*x/2)

# Define the initial guess for the root
x0 = 1

# Set tolerance and maximum number of iterations
tolerance = 1e-6
max_iterations = 10

# Find the root using the Newton-Raphson method
root = newton_raphson(f, derivative, x0, tol=tolerance)
if root is not None:
    print("Approximate root found:", root)
else:
    print("No root found within the specified tolerance and maximum number of iterations.")


    
#-------------------------- Regula Falsi method------------------------
def regula_falsi(a, b, tol=1e-6):
    iterations = 0
    if f(a) * f(b) >= 0:
        print("The function does not change sign over the interval [a, b]")
    
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

# Define the interval
a, b = 1.5, 2.5

# Find root using Regula Falsi method
root, iterations = regula_falsi(a, b)

print("Root found using Regula Falsi method:", round(root, 4))
print("Iterations:", iterations)


#______________________OUTPUT____________________________
## Output_Newton_Raphson
#Root found within tolerance at iteration for Newton_raphson 9: 1.4019297804291229
#Approximate root found: 1.4019297804291229

## Output_Regula_Falsi
#The function does not change sign over the interval [a, b]
#Root found using Regula Falsi method: 2.6231
#Iterations: 12


#-----------------------Comment--------------------------
# The given function has multiple roots.Therefore it reaches to different roots for two different methods.




