import numpy as np
import matplotlib.pyplot as plt


# Load the data from the CSV file
monorail_vel = [0.2, 0.4, 0.6, 0.8, 1.0]
goal_dist = [0.2, 0.3, 0.4, 0.5]

for vel in monorail_vel:
    for dist in goal_dist:
        path = "error_data/error_data_%.1f_%.1f.csv" % (vel, dist)
        loaded_data = np.loadtxt(path, delimiter=",", skiprows=1)

        # The 'delimiter' argument should match the delimiter used in the CSV file (a comma in this case).
        # The 'skiprows' argument is set to 1 to skip the header row.

        # Now, you can access the loaded x and y values as separate NumPy arrays
        y_dst_loaded = loaded_data[:, 0]  # First column (x-values)
        error_loaded = loaded_data[:, 1]  # Second column (y-values)

        first = True
        final_error_loaded = []
        final_y_dst_loaded = []
        for i in range(len(error_loaded)):
            if error_loaded[i] < 0.01 and first:
                first = False
            if not first:
                final_error_loaded.append(error_loaded[i])
                final_y_dst_loaded.append(y_dst_loaded[i])





        plt.figure()
        plt.plot(final_y_dst_loaded, final_error_loaded)

        plt.xlabel('Y distance')
        plt.ylabel('Absolute error')
        plt.title('Error Over Time - Monorail velocity of %.1fm/s and goal distance of %.1fm' % (vel, dist))
        plt.grid(True, 'both')
        # Add text to the bottom center
        average_error = np.average(final_error_loaded)
        text = "Average error: %.3f" % (average_error)
        plt.text(0.95, -0.1, text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        name = 'error_abs_plots/error_plot_%.1f_%.1f.png'% (vel, dist)
        plt.savefig(name)