##Mid_sem
##Name: Dipanwita Mondal
##Roll No: 23124006



import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 2  # length of the bar
T = 300  # temperature at the center of the bar
Nx = 11  # number of spatial grid points
Nt = 1000  # number of time steps

# Calculate dx and dt based on stability condition
dx = L / (Nx - 1)
dt = 0.5 * dx**2  # Ensuring stability condition

# Initialize temperature array
u = np.zeros((Nt, Nx))

# Set initial condition
u[0, int(Nx/2)] = T

# Apply boundary conditions
u[:, 0] = 0  # u(0, t) = 0
u[:, -1] = 0  # u(2, t) = 0

# Iterate over time steps
for n in range(Nt-1):
    for i in range(1, Nx-1):
        alpha = dt / dx**2
        u[n+1, i] = u[n, i] + alpha * (u[n, i+1] - 2*u[n, i] + u[n, i-1])


print(u)
# Plotting
x_values = np.linspace(0, L, Nx)
t_values = np.arange(0, Nt) * dt

plt.figure(figsize=(10, 6))
plt.contourf(x_values, t_values, u, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('Position (m)')
plt.ylabel('Time (s)')
plt.title('Temperature Distribution over Time')
plt.show()


# Function to create a table-like output
def create_table(x_values, t_values, u):
    
    print("{:<10} {:<10} {:<15}".format("x", "t", "u_sol"))
    print("{:<10} {:<10} {:<15}".format("------", "-----", "----------"))
    for i, t in enumerate(t_values):
        for j, x in enumerate(x_values):
            print("{:<10.2f} {:<10.2f} {:<15.6f}".format(x, t, u[j, i]))


