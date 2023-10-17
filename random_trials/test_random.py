import numpy as np
import matplotlib.pyplot as plt


# Load the data from the CSV file
trials = [1,2,3,4,5]
total_error = []

for t in trials:
    path = "random_trials/error_data_0.2_0.2_t%.0f.csv" % (t)
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
            total_error.append(error_loaded[i])




    
    plt.figure()
    plt.plot(final_y_dst_loaded, final_error_loaded)

    plt.xlabel('Y distance')
    plt.ylabel('Absolute error')
    plt.title('Error Over Time - Monorail velocity of 0.2m/s and goal distance of 0.2m')
    plt.grid(True, 'both')
    # Add text to the bottom center
    average_error = np.average(final_error_loaded)
    text = "Average error: %.3f" % (average_error)
    plt.text(0.95, -0.1, text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    name = 'random_trials/error_plot_t%.0f.png'% (t)
    plt.savefig(name)

total_average = np.average(np.array(total_error))
print(total_average)