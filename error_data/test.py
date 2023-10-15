import numpy as np
import matplotlib.pyplot as plt


# Load the data from the CSV file
loaded_data = np.loadtxt("error_data/error_data_0.4_0.1.csv", delimiter=",", skiprows=1)

# The 'delimiter' argument should match the delimiter used in the CSV file (a comma in this case).
# The 'skiprows' argument is set to 1 to skip the header row.

# Now, you can access the loaded x and y values as separate NumPy arrays
x_loaded = loaded_data[:, 0]  # First column (x-values)
y_loaded = loaded_data[:, 1]  # Second column (y-values)

plt.figure()
plt.plot(x_loaded, y_loaded)
plt.savefig('error_data/error_plot_0.4_0.1.png')