import numpy as np
import matplotlib.pyplot as plt

def inverse_function(x_range):
    """
    Calculates the inverse function 1/x for the given range of x.
    
    Parameters:
    x_range (list or numpy array): The range of x values to calculate the inverse function for.
    
    Returns:
    numpy array: The values of the inverse function 1/x for the given range of x.
    """
    return 1 / np.array(x_range)

# Define the range of x values
x_values = np.arange(3000, 5001, 1)

# Calculate the inverse function
y_values = inverse_function(x_values)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values)
plt.xlabel('x')
plt.ylabel('1/x')
plt.title('Inverse Function (1/x)')
plt.yscale('log')

# Customize the y-axis tick labels to show only the base
ax = plt.gca()
ax.tick_params(axis='y', which='major', labelsize=10)
formatter = plt.ScalarFormatter(useMathText=True)
formatter.set_scientific(False)
formatter.set_powerlimits((-3, 4))  # Set the power limits to display only the base
ax.yaxis.set_major_formatter(formatter)
plt.show()
