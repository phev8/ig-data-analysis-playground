import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

filename = '/Volumes/DataDrive/igroups_recordings/southampton_4/processed_data/images/general_class_rois/cam_2_detections_general.pkl'

with open(filename, 'rb') as f:
    data = pickle.load(f)

print(data)

def get_center(roi):
    cx = (roi[0] + roi[2])/2.0
    cy = (roi[1] + roi[3])/2.0
    return [cx, cy]


acc = np.zeros((1300, 800))
positions = []
region_data = np.zeros((0, 4))
for d in data['results']:
    # print(d)
    # print(d['class_names'])
    current_region = np.zeros((1, 4))
    for i in range(len(d['class_names'])):
        if d['class_names'][i] == 'person':
            center_point = get_center(d['rois'][i])
            if center_point[1] > 1118:
                current_region[0, 3] += 1
            elif center_point[1] > 562:
                current_region[0, 2] += 1
            elif center_point[1] > 315:
                current_region[0, 1] += 1
            else:
                current_region[0, 0] += 1
            acc[int(center_point[1]), int(center_point[0])] += 1
            positions.append(center_point)
    region_data = np.append(region_data, current_region, axis=0)

pos = np.array(positions)
reg = np.array(region_data)

print(reg)

plt.imshow(acc)
# plt.plot(pos[:, 0], pos[:, 1], 'x')
# plt.axis('equal')
plt.draw()

fig, axarr = plt.subplots(4, 1, sharex=True)
axarr[0].plot(reg[:, 0])
axarr[1].plot(reg[:, 1])
axarr[2].plot(reg[:, 2])
axarr[3].plot(reg[:, 3])
plt.show()