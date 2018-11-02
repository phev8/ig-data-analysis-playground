import os
import numpy as np
from common.io.labels import *
from common.io.image_processing_results import load_person_detections
from common.io.metadata import load_video_infos
from examples.read_person_detections import get_person_positions

# experiment_root = '/Volumes/DataDrive/igroups_recordings/southampton_5'

# Get absolut Path from current directory:      [Author: CM]
main_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
git_dir = str(os.path.abspath(os.path.join(main_dir, os.pardir)))
experiment_root = str(os.path.abspath(os.path.join(git_dir, os.pardir))) + "\\Daten\\igroups_student_project\\"
# print(experiment_root)


label_path = os.path.join(experiment_root, 'labels', 'Labels D2_S2.xlsx')

labels_g, labels_r, labels_b,  = read_label_xls(label_path)


# print(labels_b.position)
# print(labels_g.position)

class ImageGrid:
    def __init__(self, width, height, n_x, n_y):
        self.width = width
        self.height = height
        self.x_borders = np.linspace(0, width, n_x + 1, endpoint=True)
        self.y_borders = np.linspace(0, height, n_y + 1, endpoint=True)
        self.grid = np.zeros((n_y, n_x))
        self.grid_x = n_x
        self.grid_y = n_y

    def find_grid_for_coordinates(self, x, y):
        self.grid = np.zeros(shape=(self.grid_y, self.grid_x))      # Für nichtautomatisierte Addition

        if x < 0 or x > self.width or y < 0 or y > height:
            print("x =", x)
            print("y =", y)
            print("self.width =", self.width)
            print("height =", height)
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
grid_y = 2
height = video_info['height']
width = video_info['width']
current_grid = ImageGrid(width, height, grid_x, grid_y)


"""
c, r, g = current_grid.find_grid_for_coordinates(300, 0)
print("---------------------")
print("c =", c)
print("r =", r)
print("g =", g)
"""


"""
ToDo:
Iteriere über alle pos[i][1] und überprüfe mit Hilfe der Hilfsfunktion read_timestamp(path) aus labels.py ob der frame_index identisch ist. 
    Wenn ja überprüfe ob die Koordinaten aus pos[i][0] zu den Labels aus der Excel Datei passen
    Sonst mache nichts, denn der Frame ist nicht gelabelt.
"""


def convert_fuzzy_labels_to_grid(list, n_x, n_y):

    grid = np.zeros(shape=(n_x,n_y))
    wrong = False
    for i in list:
        if i == 'LH':
            grid[0, 0] = 1
        elif i == 'H':
            grid[0, 1] = 1
        elif i == 'RH':
            grid[0, 2] = 1
        elif i == 'MO':
            grid[0, 3] = 1
        elif i == 'LF':
            grid[1, 1] = 1
        elif i == 'F':
            grid[1, 2] = 1
        elif i == 'RF':
            grid[1, 3] = 1
        # else:
        #     grid[1, 0] = 1

    return grid



def compare_lables_with_detections():
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    temp = read_timestamp(label_path)
    all_labled_timestamps = []
    all_fuzzy_positions = []

    for i in range(len(temp)):
        all_labled_timestamps.append(temp[i][0])
        all_fuzzy_positions.append(temp[i][1])

    first_timestamp = all_labled_timestamps[0]
    last_timestamp = all_labled_timestamps[len(all_labled_timestamps)-1]
    print("all_timestamps =", all_labled_timestamps)
    print("first_timestamp =", first_timestamp)
    print("last_timestamp =", last_timestamp)

    person_positions = get_person_positions()

    print(person_positions)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    counter = 0
    correct_classified = 0
    wrong_classified = 0
    for labled_frames in all_labled_timestamps:
        print(counter, "of", len(all_labled_timestamps))

        all_person_grid = np.zeros(shape=(2, 4))

        # iterate over all timestamps:
        for i in range(len(person_positions)):
            # Is i a frame which is labled? If so, the take the detected person coordinates
            if person_positions[i][1] == labled_frames:
                coordinate, frame = person_positions[i]
                # print("coordinate =", coordinate)
                # print("frame =", frame)
                # print("i =", i)

                c, r, g = current_grid.find_grid_for_coordinates(coordinate[1], coordinate[0])
                all_person_grid += g
        # print(all_person_grid)

        fuzzy_grid = convert_fuzzy_labels_to_grid(all_fuzzy_positions[counter], grid_y, grid_x)
        erg = np.abs(all_person_grid - fuzzy_grid)

        if np.sum(erg) == 0:
            correct_classified += 1
        else:
            wrong_classified += 1

        counter += 1
    print("correct_classified =", correct_classified)
    print("wrong_classified =", wrong_classified)

    # return

compare_lables_with_detections()



detection_file_path = os.path.join(experiment_root, 'processed_data', 'images', 'general_class_rois', 'cam_2_detections_general.pkl')

fps = video_info['fps']
frame_count = video_info['frame_count']




# --------------------------------------------------------------------------------------------------------------------------------------------
# Wozu benötigen wir diese Funktion? In read_person_detection ist doch schon eine Methode, die alle Positionen ließt.

# TODO: implement this:
# load_person_detections(detection_file_path, fps, frame_count)
# --------------------------------------------------------------------------------------------------------------------------------------------
