import numpy as np


# Load the data from the CSV file
voxel_size = ['001', '020', '040', '060', '080', '100', '120', '140', '160']
trials = ['a', 'b', 'c']
dst = 0.2

voxel_times = {}

for voxl in voxel_size:
    obstacle_num = []
    tree_build_time = []
    camera_time_av = []
    policy_time_av = []
    step_time_av = []
    total_time_step_average = []

    voxel_map = {}

    for trial in trials:
        path = f"data/voxel_data_trials/time_data_File_{voxl}{trial}.csv"
        try:
            loaded_data = np.loadtxt(path, delimiter=",", skiprows=1)
        except:
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

    
    voxel_map['av_obstacle_num'] = np.average(obstacle_num)
    voxel_map['av_tree_build_time'] = np.average(tree_build_time)
    voxel_map['av_camera_time_av'] = np.average(camera_time_av)
    voxel_map['av_policy_time_av'] = np.average(policy_time_av)
    voxel_map['av_step_time_av'] = np.average(step_time_av)
    voxel_map['av_total_time_step_average'] = np.average(total_time_step_average)

    voxel_times[voxl] = voxel_map

obstacle_nums = []
tree_build_times = []
camera_time_avs = []
policy_time_avs = []
step_time_avs = []
total_time_step_averages = []

for key in voxel_times.keys():
    obstacle_nums.append(voxel_times[key]['av_obstacle_num'])
    tree_build_times.append(voxel_times[key]['av_tree_build_time'])
    camera_time_avs.append(voxel_times[key]['av_camera_time_av'])
    policy_time_avs.append(voxel_times[key]['av_policy_time_av'])
    step_time_avs.append(voxel_times[key]['av_step_time_av'])
    total_time_step_averages.append(voxel_times[key]['av_total_time_step_average'])
    # print(f"{key}: {voxel_times[key]}\n")
# print(voxel_times)

print(f"obstacle_nums = {obstacle_nums};\n")
print(f"tree_build_times = {tree_build_times};\n")
print(f"camera_time_avs = {camera_time_avs};\n")
print(f"policy_time_avs = {policy_time_avs};\n")
print(f"step_time_avs = {step_time_avs};\n")
print(f"total_time_step_averages = {total_time_step_averages};\n")