import numpy as np
import matplotlib.pyplot as plt


# Load the data from the CSV file
path = 'data/baseline_data/surface_dist_data_0.5_final.csv'
loaded_data = np.loadtxt(path, delimiter=",", skiprows=1)
goal_distance = 0.5

# The 'delimiter' argument should match the delimiter used in the CSV file (a comma in this case).
# The 'skiprows' argument is set to 1 to skip the header row.

# Now, you can access the loaded x and y values as separate NumPy arrays
y_dst_loaded = loaded_data[:, 0]  # First column (x-values)
surface_distance = loaded_data[:, 1]  # Second column (y-values)

plt.figure()
plt.plot(y_dst_loaded, surface_distance)

plt.xlabel('Y distance (m)')
plt.ylabel('Distance from surface (m)')
plt.ylim([0.35, 0.80])
plt.title('Distance from surface over duration of simulation trial')
plt.grid(True, 'both')

# Add text to the bottom center
# average_error = np.average(final_error_loaded)
# text = "Average error: %.3f" % (average_error)
# plt.text(0.95, -0.1, text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
# Draw a horizontal line at y = 0.5
plt.axhline(y=goal_distance, color='green', linestyle='--', label='goal distance')

#20% error lines
plt.axhline(y=goal_distance+(goal_distance*0.05), color='red', linestyle='--', label='+5% error')
plt.axhline(y=goal_distance-(goal_distance*0.05), color='red', linestyle='--', label='-5% error')
name = 'data/baseline_data/surface_distance_plot_final.png'
plt.savefig(name)

