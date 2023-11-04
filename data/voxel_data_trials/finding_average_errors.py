import numpy as np


# Load the data from the CSV file
voxel_size = ['001', '020', '040', '060', '080', '100', '120', '140', '160']
trials = ['a', 'b', 'c']
dst = 0.2

percentage_errors = {}
for voxl in voxel_size:
    voxl_percentage_errors = []
    for trial in trials:
        path = f"data/voxel_data_trials/error_data_File_{voxl}{trial}.csv"
        try:
            loaded_data = np.loadtxt(path, delimiter=",", skiprows=1)
        except:
            break

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

        final_error_loaded = np.array(final_error_loaded)
        percent_array = np.full(final_error_loaded.shape, 100/dst)
        percentage_error_values =  final_error_loaded * percent_array
        
        average_error = np.average(final_error_loaded)
        average_percentage_error = (average_error / dst) *100
        
        voxl_percentage_errors.append(average_percentage_error)

    average_voxel_p_error =  np.average(voxl_percentage_errors)
    percentage_errors[voxl] = average_voxel_p_error

errors = []
for key in percentage_errors.keys():
    errors.append(percentage_errors[key])
print(f"percentage_errors = {errors};")