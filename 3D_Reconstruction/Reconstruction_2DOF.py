import pickle
import csv

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np


from Skeleton import Global_Constants

main_dir = Global_Constants.main_dir
git_dir = Global_Constants.git_dir
project_dir = Global_Constants.project_dir
data_dir = Global_Constants.data_dir
data_dir_ext = Global_Constants.data_dir_ext
filename = Global_Constants.filename
output_dir = Global_Constants.output_dir
skeletons_dir = Global_Constants.skeletons_dir



def read_feature_matrices_from_pickle_file():
    pkl_file = open(data_dir + 'Feature_Matrices.pkl', "rb")
    matrix = pickle.load(pkl_file)
    return matrix


huge_feature_matricies = read_feature_matrices_from_pickle_file()

xs = []
ys = []
zs = []

plot_counter = 0


csv_path = data_dir + "3D Reconstruction.csv"
csv_file = open(csv_path, "r")
csvReader = list(csv.reader(csv_file, delimiter=';'))
header = csvReader[0:2]
rest = csvReader[2:]

num_points = 7

coordinates = [elem[0:9] for elem in rest]
flatted_matrix_new = np.array(np.reshape([elem[0:6] for elem in rest], newshape=(num_points,3,2)),dtype=float)
flatted_matrix = np.array([elem for liste in rest for elem in liste[0:6]], dtype=float)
point_in_3d = np.array([elem for liste in rest for elem in liste[6:9]], dtype=float)
point_in_3d_new = np.array(np.reshape([elem[6:9] for elem in rest], newshape=(num_points,3)),dtype=float)

print("coordinates =", coordinates)
print("flatted_matrix_new =", flatted_matrix_new)
print("flatted_matrix =", flatted_matrix)
print("point_in_3d =", point_in_3d)

A = np.reshape(flatted_matrix,newshape=(3*num_points,2))
b = np.reshape(point_in_3d,newshape=(3*num_points,1))

print("A =", A)
print("b =", b)

from scipy.optimize import least_squares

# x ist eine 6 x 3 Matrix.
# Für jede Bildkoordinate der 3 Kameras => 6 Zeilen und
# wegen 3D Output noch drei Spalten.
def optimize(x):
    erg = []
    for i in range(len(flatted_matrix_new)):
        point = flatted_matrix_new[i]
        # print("point =", point)
        # print("point[0] =", point[0])
        # temp = []
        for j in range(len(point)):
            elem = point[j]
            erg.append(float(elem[0])*x[0] + float(elem[1])*x[1] - float(point_in_3d_new[i][j]))


    # erg = np.array([[row[0] * x[0] - b[0], row[1] * x[1] - b[1], row[2] * x[1] - b[1]] for row in A], dtype=float)
    return np.array(erg)

x = -float(27081.0)/float(1192600.0)
y = float(27591.0)/float(298150.0)

print("optimize([x,y]) =", optimize([x,y]))
print("optimize([x,y]).shape =", optimize([x,y]).shape)
# erg = [elem**2 for elem in list(optimize([x,y]))]
# print("optimize()", sum(erg))


cost = 10833583
result = []


opt = True

a = 1.17748624  # -float(27081.0)/float(1192600.0)
b = -1.15452207  # float(27591.0)/float(298150.0)
"""
# Cubic Approach
for i in np.arange(-5.1,0,1):
    print("i =", i)
    for j in np.arange(5.1,0,-1):
        for k in np.arange(-500,0,100):
            for l in np.arange(500,0,-100):
                result = least_squares(optimize, x0=np.array([i,j,k,l]))
                if result.cost < cost:
                    cost = result.cost
                    print("result.cost =", result.cost)
                    print("result.x =", result.x)
"""

if opt:
    for i in np.arange(-50, 0, 10):
        print("i =", i)
        for j in np.arange(50, 0, -10):
            result = least_squares(optimize, x0=np.array([i, j]),
                                   method='dogbox', loss='soft_l1', tr_solver='exact')
            if result.cost < cost:
                cost = result.cost
                print("result.cost =", result.cost)
                print("result.x =", result.x)
                a = result.x[0]
                b = result.x[1]

# c = result.x[2]
# d = result.x[3]
plotten = True

if plotten:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(flatted_matrix_new)):
        point = flatted_matrix_new[i]
        # print("point =", point)
        print("point[0] =", point[0])
        # temp = []
        x_lin = float(point[0][0]) * a + float(point[0][1]) * b
        y_lin = float(point[1][0]) * a + float(point[1][1]) * b
        z_lin = float(point[2][0]) * a + float(point[2][1]) * b
        print("The point: \n", point, " becomes to (", x_lin, ", ", y_lin, ", ", z_lin)
        """
        x_cubic = float(point[0][0]) * c**3 + float(point[0][1]) * d**3
        y_cubic = float(point[1][0]) * c**3 + float(point[1][1]) * d**3
        z_cubic = float(point[2][0]) * d**3 + float(point[2][1]) * d**3
        """

        xs.append(x_lin) # + x_cubic)
        ys.append(y_lin) # + y_cubic)
        zs.append(z_lin) # + z_cubic)


        if i == 1:
            print("\t\t m=x")
            ax.scatter(x_lin, y_lin, z_lin, c='black', marker='o')
        else:
            ax.scatter(x_lin, y_lin, z_lin, c='black', marker='x')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

"""
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


        # Beginn Brechnung der Hoehe und Breite des Skeletons: --------------------------------------------------------
        minmax = features[4]
        minmax_cam1 = minmax[0]
        minmax_cam2 = minmax[1]
        minmax_cam3 = minmax[2]

        min_x_cam1 = int(minmax_cam1[1])
        max_x_cam1 = int(minmax_cam1[2])
        min_y_cam1 = int(minmax_cam1[3])
        max_y_cam1 = int(minmax_cam1[4])
        cam1_x_size = max_x_cam1 - min_x_cam1
        cam1_y_size = max_y_cam1 - min_y_cam1

        min_x_cam2 = int(minmax_cam2[1])
        max_x_cam2 = int(minmax_cam2[2])
        min_y_cam2 = int(minmax_cam2[3])
        max_y_cam2 = int(minmax_cam2[4])
        cam2_x_size = max_x_cam2 - min_x_cam2
        cam2_y_size = max_y_cam2 - min_y_cam2

        min_x_cam3 = int(minmax_cam3[1])
        max_x_cam3 = int(minmax_cam3[2])
        min_y_cam3 = int(minmax_cam3[3])
        max_y_cam3 = int(minmax_cam3[4])
        cam3_x_size = max_x_cam3 - min_x_cam3
        cam3_y_size = max_y_cam3 - min_y_cam3
        # Ende Brechnung der Hoehe und Breite des Skeletons: --------------------------------------------------------


        mid_hip_xy_cam1 = features[1]
        mid_hip_xy_cam2 = features[2]
        mid_hip_xy_cam3 = features[3]

        a = 1.17748624 # -float(27081.0)/float(1192600.0)
        b = -1.15452207 # float(27591.0)/float(298150.0)

        x = mid_hip_xy_cam1[0] * a + mid_hip_xy_cam1[1] * b
        y = mid_hip_xy_cam2[0] * a + mid_hip_xy_cam2[1] * b
        z = mid_hip_xy_cam3[0] * a + mid_hip_xy_cam3[1] * b





        # if cam1_x_size * cam1_y_size > 20000 and cam2_x_size * cam2_y_size > 20000 and cam3_x_size * cam3_y_size > 20000 and plot_counter % 20 == 0:
        if plot_counter % 25 == 0:
            xs.append(x)
            ys.append(y)
            zs.append(z)


        plot_counter += 1
"""


