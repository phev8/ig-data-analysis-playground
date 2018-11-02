import pickle


def load_person_detections(path, video_fps, frame_count):
    f = open(path, 'rb')
    data = pickle.load(f)
    print(data)
    f.close()

    # TODO: generate list of person detections for each frame and convert frame number to video time using timedelta

    indeces = []
    for d in data['results']:
        for i in range(len(d['class_names'])):
            # print(d)
            indeces.append(d['frame_index'])
            if d['class_names'][i] == 'person':
                pass


    import numpy as np
    ind = np.array(indeces)
    for dif in np.diff(ind):
        print(dif)



"""


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
"""