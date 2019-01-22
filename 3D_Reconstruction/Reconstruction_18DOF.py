import pickle
import csv
import os

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np


# from Skeleton import Global_Constants

main_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
git_dir = str(os.path.abspath(os.path.join(main_dir, os.pardir)))
project_dir = str(os.path.abspath(os.path.join(git_dir, os.pardir)))  + "\\"
project_dir_ext = "H:\\Uni\\Statistische KI Projekt\\"
data_dir = project_dir + "Daten\\igroups_student_project\\"
data_dir_ext = project_dir_ext + "Daten\\igroups_student_project\\"
filename = project_dir + "Daten\\igroups_student_project\\videos\\cam_2.mp4"
output_dir = project_dir + "Daten\\igroups_student_project\\Frames_extracted\\"
skeletons_dir = data_dir + "processed_data\\images\\skeletons\\"


def read_feature_matrices_from_pickle_file():
    pkl_file = open(data_dir + 'Feature_Matrices.pkl', "rb")
    matrix = pickle.load(pkl_file)
    return matrix


huge_feature_matricies = read_feature_matrices_from_pickle_file()



csv_path = data_dir + "3D Reconstruction.csv"
csv_file = open(csv_path, "r")
csvReader = list(csv.reader(csv_file, delimiter=';'))
header = csvReader[0:2]
rest = csvReader[2:]

num_points = 7

coordinates = [elem[0:9] for elem in rest]
flatted_matrix = np.array([elem[0:6] for elem in rest], dtype=float)
matrix_for_plot = np.array(np.reshape([elem[0:6] for elem in rest], newshape=(num_points, 3, 2)), dtype=float)
point_in_3d = np.array([elem[6:9] for elem in rest], dtype=float)


print("coordinates =", coordinates)
print("flatted_matrix =\n", flatted_matrix)
print("point_in_3d =\n", point_in_3d)

A = np.reshape(flatted_matrix,newshape=(3*num_points,2))
b = np.reshape(point_in_3d,newshape=(3*num_points,1))

print("A =", A)
print("b =", b)

from scipy.optimize import least_squares

# Wichtig, die zu optimierende Funktion muss von der Dimension 1 sein. Falls also wie hier eine Matrix optimiert werden soll,
# muss die Matrix geflatted werden.
def optimize(x):
    x = np.reshape(x,newshape=(6,3))
    erg = []
    for i in range(len(flatted_matrix)):
        point = flatted_matrix[i]
        """
        print("point =", point)
        print("point.shape =", point.shape)
        print("x.shape =", x.shape)
        print("point_in_3d =", point_in_3d)
        print("point_in_3d.shape =", point_in_3d.shape)
        print("point_in_3d[i] =", point_in_3d[i])
        print("point_in_3d[i].shape =", point_in_3d[i].shape)
        """
        temp = np.matmul(point, x) - point_in_3d[i]
        # print("temp.shape =", temp.shape)
        erg.extend(temp)

    return np.array(erg)

x = -float(27081.0)/float(1192600.0)
y = float(27591.0)/float(298150.0)


test_array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
teil_erg = optimize(test_array)
print("teil_erg.shape =", teil_erg.shape)
print("optimize(test_array) =",teil_erg)

# erg = [elem**2 for elem in list(optimize([x,y]))]
# print("optimize()", sum(erg))


cost = 10**100
result = []

opt = True

a = 1.17748624  # -float(27081.0)/float(1192600.0)
b = -1.15452207  # float(27591.0)/float(298150.0)


if opt:
    for i in np.arange(-50, 0, 10):
        print("i =", i)
        for j in np.arange(50, 0, -10):
            start_parameter = np.random.rand(18,)
            result = least_squares(optimize, x0=start_parameter,
                                   method='trf', loss='soft_l1', tr_solver='exact')
            if result.cost < cost:
                cost = result.cost
                print("result.cost =", result.cost)
                print("result.x =", result.x)

opt_values = np.reshape(np.array(result.x),newshape=(6,3))
print("opt_values =", opt_values)
# c = result.x[2]
# d = result.x[3]


def plot_selfmade_3D_points():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    all_3D_points = np.matmul(flatted_matrix, opt_values)
    print("all_3D_points =", all_3D_points)
    print("all_3D_points.shape =", all_3D_points.shape)

    x_values = [row[0] for row in all_3D_points]
    y_values = [row[1] for row in all_3D_points]
    z_values = [row[2] for row in all_3D_points]


    ax.scatter(x_values, y_values, z_values, c='black', marker='x')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    return [x_values, y_values, z_values]

xs, ys, zs = plot_selfmade_3D_points()





def plot_3D_points(x_values, y_values, z_values, marker='o'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_values, y_values, z_values, c='black', marker=marker)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()





plot_MidHip_values = False
plot_counter = 0

if plot_MidHip_values:
    xs = []
    ys = []
    zs = []


    for frames in huge_feature_matricies:
        person_counter = 0
        for row in frames:
            person_counter += 1
            # print(person_counter, "of", len(frames))

            features = row[0]
            labels = row[1]
            # print(features)

            frame_number = features[0]
            print("Kamera 2 frame number =", frame_number)

            mid_hip_coordinates = np.array(features[1] + features[2] + features[3])


            print("mid_hip_coordinates =", mid_hip_coordinates)

            temp_3D_point = list(np.matmul(mid_hip_coordinates, opt_values))
            print("temp_3D_point =", temp_3D_point)

            # if cam1_x_size * cam1_y_size > 20000 and cam2_x_size * cam2_y_size > 20000 and cam3_x_size * cam3_y_size > 20000 and plot_counter % 20 == 0:
            if plot_counter % 25 == 0:
                xs.append(temp_3D_point[0])
                ys.append(temp_3D_point[1])
                zs.append(temp_3D_point[2])


            plot_counter += 1


    plot_3D_points(xs, ys, zs)


def test_new_point(coordinates):
    temp_3D_point = list(np.matmul(coordinates, opt_values))
    xs.append(temp_3D_point[0])
    ys.append(temp_3D_point[1])
    zs.append(temp_3D_point[2])

    plot_3D_points(xs, ys, zs, marker='o')

"""
def test_features():
    pkl_file2 = open(data_dir + 'Final_Feature_Matrix.pkl', "rb")
    feature_matrix = pickle.load(pkl_file2)

    for elem in feature_matrix:
        frame_number = elem[0]
        cam1_coordinates = elem[1]
        cam2_coordinates = elem[2]
        cam3_coordinates = elem[3]
        person_id_in_excel = elem[4]
        label = elem[5]

        print("elem =", elem)
        all_coordinates = cam1_coordinates + cam2_coordinates + cam3_coordinates
        print("all_coordinates =", all_coordinates)


test_new_point(np.array([1114,307,640,408,650,426]))

test_features()
"""

# 1114;307;640;408;650;426;1350;80;37;Falte Leintuch;496 nicht sync;6456 nicht sync;6375 nicht sync
