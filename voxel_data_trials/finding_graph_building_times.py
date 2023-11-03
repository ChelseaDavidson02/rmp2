import numpy as np


# Load the data from the CSV file
voxel_size = ['001', '020', '040', '060', '080', '100', '120', '140', '160']
trials = ['a', 'b', 'c']
dst = 0.2

voxel_times = {}
for voxl in voxel_size:
    voxel_map = {}
    voxel_map['forward_pass'] = []
    voxel_map['rmp_evaluation'] = []
    voxel_map['backward_pass'] = []
    voxel_map['resolve'] = []

    voxel_times[voxl] = voxel_map

path = f"voxel_data_trials/rmp_timing.txt"
# loaded_data = np.loadtxt(path, delimiter=",", skiprows=1)

# voxel_map = {}
# for row in range(len(loaded_data[:, 0])):
#     name = loaded_data[row, 0]
#     vox_sz = name[5:8]
#     print(vox_sz)
#     voxel_map = voxel_times[vox_sz]
#     voxel_map['forward_pass'].append(loaded_data[row, 1])
#     voxel_map['rmp_evaluation'].append(loaded_data[row, 2])
#     voxel_map['backward_pass'].append(loaded_data[row, 3])
#     voxel_map['resolve'].append(loaded_data[row, 4])

voxel_map = {}

with open(path, 'r') as file:
    # Skip the header line
    next(file)
    
    for line in file:
        # Split the line into columns based on the comma delimiter
        columns = line.strip().split(',')

        name = columns[0]
        vox_sz = name[5:8]
        # print(vox_sz)

        voxel_map = voxel_times[vox_sz]
        voxel_map['forward_pass'].append(float(columns[1]))
        voxel_map['rmp_evaluation'].append(float(columns[2]))
        voxel_map['backward_pass'].append(float(columns[3]))
        voxel_map['resolve'].append(float(columns[4]))



graph_building_time_map = {}
for key in voxel_times.keys():
    voxel_map = voxel_times[key]
    new_voxel_map = {}
    times = []
    new_voxel_map['ave_forward_pass'] = np.average(np.array(voxel_map['forward_pass']))
    times.append(new_voxel_map['ave_forward_pass'])
    new_voxel_map['ave_rmp_evaluation'] = np.average(np.array(voxel_map['rmp_evaluation']))
    times.append(new_voxel_map['ave_rmp_evaluation'])
    new_voxel_map['ave_backward_pass'] = np.average(np.array(voxel_map['backward_pass']))
    times.append(new_voxel_map['ave_backward_pass'])
    new_voxel_map['ave_resolve'] = np.average(np.array(voxel_map['resolve']))
    times.append(new_voxel_map['ave_resolve'])
    new_voxel_map['total_time'] = np.sum(np.array(times))

    graph_building_time_map[key] = new_voxel_map


ave_forward_passes = []
ave_rmp_evaluations = []
ave_backward_passes = []
ave_resolves = []
total_times = []

for key in graph_building_time_map.keys():
    ave_forward_passes.append(graph_building_time_map[key]['ave_forward_pass'])
    ave_rmp_evaluations.append(graph_building_time_map[key]['ave_rmp_evaluation'])
    ave_backward_passes.append(graph_building_time_map[key]['ave_backward_pass'])
    ave_resolves.append(graph_building_time_map[key]['ave_resolve'])
    total_times.append(graph_building_time_map[key]['total_time'])
    # print(f"{key}: {voxel_times[key]}\n")
# print(voxel_times)

print(f"ave_forward_passes = {ave_forward_passes};\n")
print(f"ave_rmp_evaluations = {ave_rmp_evaluations};\n")
print(f"ave_backward_passes = {ave_backward_passes};\n")
print(f"ave_resolves = {ave_resolves};\n")
print(f"total_times = {total_times};\n")
