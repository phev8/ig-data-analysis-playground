import os
import numpy as np
from common.io.labels import read_label_xls
from common.io.image_processing_results import load_person_detections
from common.io.metadata import load_video_infos


experiment_root = '/Volumes/DataDrive/igroups_recordings/southampton_5'

label_path = os.path.join(experiment_root, 'labels', 'Labels D2_S2.xlsx')

labels_g, labels_r, labels_b,  = read_label_xls(label_path)

print(labels_b.position)

class ImageGrid:
    def __init__(self, width, height, n_x, n_y):
        self.width = width
        self.height = height
        self.x_borders = np.linspace(0, width, n_x + 1, endpoint=True)
        self.y_borders = np.linspace(0, height, n_y + 1, endpoint=True)
        self.grid = np.zeros((n_y, n_x))

    def find_grid_for_coordinates(self, x, y):
        if x < 0 or x > self.width or y < 0 or y > height:
            raise ValueError('out of image boundaries')

        row = 0
        col = 0
        for i in range(0, len(self.y_borders) - 1):
            if self.y_borders[i] <= y <= self.y_borders[i+1]:
                row = i
                break

        for i in range(0, len(self.x_borders) - 1):
            if self.x_borders[i] <= x <= self.x_borders[i+1]:
                col = i
                break

        self.grid[row, col] += 1
        return col, row, self.grid.copy()


video_info = load_video_infos(experiment_root, 'cam_2.MP4')


grid_x = 4
grid_y = 3
height = video_info['height']
width = video_info['width']

current_grid = ImageGrid(width, height, grid_x, grid_y)

c, r, g = current_grid.find_grid_for_coordinates(300, 0)
print(c, r, g)



detection_file_path = os.path.join(experiment_root, 'processed_data', 'images', 'general_class_rois', 'cam_2_detections_general.pkl')

fps = video_info['fps']
frame_count = video_info['frame_count']

# TODO: implement this:
load_person_detections(detection_file_path, fps, frame_count)

