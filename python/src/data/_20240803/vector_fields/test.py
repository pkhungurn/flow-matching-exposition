import numpy as np
import matplotlib.pyplot as plt

# Define the grid of points
x = np.linspace(-5, 5, 21)
y = np.linspace(-5, 5, 21)
X, Y = np.meshgrid(x, y)

# Define the vector field components
#U = X*0 + 1
#V = X*0
U = -Y
V = X

# Create the plot
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, scale=40, color='blue')

# Add labels and title
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Vector Field Visualization')

# Show the plot
plt.grid()
plt.show()