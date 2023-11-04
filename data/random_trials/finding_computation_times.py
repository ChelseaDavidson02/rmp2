import numpy as np


# Load the data from the CSV file
voxel_size = ['0.04', '0.06', '0.08']
monorail_velocity = ['0.20', '0.40']
goal_distances = ['0.20', '0.30', '0.40']

obstacle_num = []
tree_build_time = []
camera_time_av = []
policy_time_av = []
step_time_av = []
total_time_step_average = []

for voxl in voxel_size:
    
    for vel in monorail_velocity:
        for dist in goal_distances:
            # time_data_File_0.04_0.20_0.20
            path = f"data/random_trials/time_data_File_{voxl}_{vel}_{dist}_t2.csv"
            try:
                loaded_data = np.loadtxt(path, delimiter=",", skiprows=1)
            except:
                print("couldnt find file", path)
                break

            # The 'delimiter' argument should match the delimiter used in the CSV file (a comma in this case).
            # The 'skiprows' argument is set to 1 to skip the header row.

            # Now, you can access the loaded x and y values as separate NumPy arrays
            obstacle_num.append(loaded_data[0])
            tree_build_time.append(loaded_data[1])  # Second row, second column 
            camera_time_av.append(loaded_data[2])  # Second row, third column
            policy_time_av.append(loaded_data[3])
            step_time_av.append(loaded_data[4])
            total_time_step_average.append(loaded_data[5])

print(f"obstacle_nums = {np.average(obstacle_num)};\n")
print(f"tree_build_times = {np.average(tree_build_time)};\n")
print(f"camera_time_avs = {np.average(camera_time_av)};\n")
print(f"policy_time_avs = {np.average(policy_time_av)};\n")
print(f"step_time_avs = {np.average(step_time_av)};\n")
print(f"total_time_step_averages = {np.average(total_time_step_average)};\n")