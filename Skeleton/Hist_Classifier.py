import os
import glob
import pickle
import cv2

from Skeleton import Global_Constants
from Video_Processing.Histograms import calc_group_histogramms, compare_histograms

main_dir = Global_Constants.main_dir
git_dir = Global_Constants.git_dir
project_dir = Global_Constants.project_dir
data_dir = Global_Constants.data_dir
data_dir_ext = Global_Constants.data_dir_ext
filename = Global_Constants.filename
output_dir = Global_Constants.output_dir
skeletons_dir = Global_Constants.skeletons_dir



crop_size_width = 75
crop_size_height = 75

height = 719
width = 1279


def read_feature_matrices_from_pickle_file():
    pkl_file = open(data_dir + 'Feature_Matrices.pkl', "rb")
    matrix = pickle.load(pkl_file)
    return matrix

"""

def classify_image(img, handsorted_group_histograms):

    # Mögliche Klassen oder Gruppen:
    groups = ["Blue Button", "Green Button", "Red Button", "Dummy", "Schwarze Weste", "Sonstiges"]

    # Berechne für alle vorsortierten Bilder der aktuelle Gruppe das gemittelte und normalisierte Histogramm über alle Kanäle (BGR, statt RGB, da cv2):
    color = ('b', 'g', 'r')
    all_hists_cur_group = []
    hist_all_channels = []
    for i, col in enumerate(color):
        # Die Funktion calcHist berechnet Histogramme aus eine Liste und mittelt diese.
        hist = cv2.calcHist(data, [i], None, [256], [0, 256])
        hist = cv2.normalize(hist, None)
        hist_all_channels.append(hist)
    all_hists_cur_group.append(hist_all_channels)

    hist_all_channels = []
    

"""

def read_and_crop_img(path, midhip_coordinates):
    x_coordinate, y_coordinate = midhip_coordinates

    min_x_cam1 = max(int(x_coordinate) - crop_size_width, 0)
    min_y_cam1 = max(int(y_coordinate) - crop_size_height, 0)
    max_x_cam1 = min(int(x_coordinate) + crop_size_width, width)
    max_y_cam1 = min(int(y_coordinate) + crop_size_height, height)

    print("min_x_cam1 =", min_x_cam1)
    print("min_y_cam1 =", min_y_cam1)
    print("max_x_cam1 =", max_x_cam1)
    print("max_y_cam1 =", max_y_cam1)


    img = cv2.imread(path)[min_y_cam1:max_y_cam1, min_x_cam1:max_x_cam1]

    return img


def classify_online(is_midhip_approach=True):
    huge_feature_matricies = read_feature_matrices_from_pickle_file()

    # Definiere Pfad zu den extrahierten Personen. Je nach Ansatz (Bildausschnitt um MidHip oder minmax, ändert sich der Pfad, ggf anpassen):
    if is_midhip_approach:
        path_extracted_persons = data_dir + "Skeletons_extracted_around_MidHip\\"
    else:
        path_extracted_persons = data_dir + "Skeletons_extracted_minmax\\"


    handsorted_group_histograms = calc_group_histogramms()

    for frames in huge_feature_matricies:
        print(frames)
        for person_id in range(len(frames)):
            row = frames[person_id]

            features = row[0]
            labels = row[1]

            # Aktuelle Frames:
            frame_cam1 = features[1][0]
            frame_cam2 = features[2][0]
            frame_cam3 = features[3][0]

            mid_hip_xy_cam1 = features[1][1:]
            mid_hip_xy_cam2 = features[2][1:]
            mid_hip_xy_cam3 = features[3][1:]
            # print("mid_hip_xy_cam1 =", mid_hip_xy_cam1, "mid_hip_xy_cam2 =", mid_hip_xy_cam2, "mid_hip_xy_cam3 =", mid_hip_xy_cam3, "\n")


            # Nächster Frame bzw gesamtes Bild wird eingelesen:

            temp_frames_cam1_path = path_extracted_persons + "cam_1_frame_" + str(frame_cam1) + "_" + str(person_id + 1) + ".png"
            temp_frames_cam2_path = path_extracted_persons + "cam_2_frame_" + str(frame_cam2) + "_" + str(person_id + 1) + ".png"
            temp_frames_cam3_path = path_extracted_persons + "cam_3_frame_" + str(frame_cam3) + "_" + str(person_id + 1) + ".png"

            print("temp_frames_cam1_path =", temp_frames_cam1_path)
            print("temp_frames_cam2_path =", temp_frames_cam2_path)
            print("temp_frames_cam3_path =", temp_frames_cam3_path)

            person_cam1 = cv2.imread(temp_frames_cam1_path)
            person_cam2 = cv2.imread(temp_frames_cam2_path)
            person_cam3 = cv2.imread(temp_frames_cam3_path)


            color = ('b', 'g', 'r')
            for person in [person_cam1,person_cam2,person_cam3]:
                hist_new_person = []
                for i, col in enumerate(color):
                    hist = cv2.calcHist([person], [i], None, [256], [0, 256])
                    hist = cv2.normalize(hist, None)
                    hist_new_person.append(hist)
                group_id = compare_histograms(handsorted_group_histograms, hist_new_person)
                cv2.imshow(str(group_id), person)
                cv2.waitKey()


            """
            cv2.imshow("person_cam1", person_cam1)
            cv2.imshow("person_cam2", person_cam2)
            cv2.imshow("person_cam3", person_cam3)
            cv2.waitKey()
            """



def classify_offline(is_midhip_approach=True):
    huge_feature_matricies = read_feature_matrices_from_pickle_file()

    # Definiere Pfad zu den extrahierten Personen. Je nach Ansatz (Bildausschnitt um MidHip oder minmax, ändert sich der Pfad, ggf anpassen):
    if is_midhip_approach:
        path_extracted_persons = data_dir + "Skeletons_separated\\"
    else:
        path_extracted_persons = data_dir + "...\\"

    # Mögliche Endungen: "Blue Button", "Green Button", "Red Button" oder Dummy
    groups = ["Blue Button", "Green Button", "Red Button", "Dummy", "Schwarze Weste", "Sonstiges"]

    show_img = False

    classified_persons = []

    num_blue_prints = 0
    num_green_prints = 0
    num_red_prints = 0

    for frames in huge_feature_matricies:
        # print(frames)

        cam1_persons = []
        cam2_persons = []
        cam3_persons = []

        person_counter_cam1 = 0
        person_counter_cam2 = 0
        person_counter_cam3 = 0

        for person_id in range(len(frames)):
            row = frames[person_id]
            # print("row =", row)
            features = row[0]
            labels = row[1]

            is_detection_cam1 = len(features[1]) > 0
            is_detection_cam2 = len(features[2]) > 0
            is_detection_cam3 = len(features[3]) > 0

            """
            print("is_detection_cam1 =", is_detection_cam1)
            print("is_detection_cam2 =", is_detection_cam2)
            print("is_detection_cam3 =", is_detection_cam3)
            """

            # if len(features[1]) != 0:
            if is_detection_cam1:
                # features = features[1]
                # print("features =",features)
                # Aktuelle Frames:
                frame_cam1 = features[1][0]

                mid_hip_xy_cam1 = features[1][1:]

                for group_id in range(len(groups)):
                    # Nächster Frame bzw gesamtes Bild wird eingelesen:
                    temp_frames_cam1_path = path_extracted_persons + str(groups[group_id]) + "\\cam_1_frame_" + str(frame_cam1) + "_" + str(person_counter_cam1+1) + ".png"
                    # print("temp_frames_cam1_path =", temp_frames_cam1_path)


                    if os.path.isfile(temp_frames_cam1_path):
                        cam1_persons.append([frame_cam1, mid_hip_xy_cam1, group_id])

                        if show_img:
                            cv2.imshow(groups[group_id] + " = " + str(group_id), cv2.imread(temp_frames_cam1_path))
                            cv2.waitKey()

                        person_counter_cam1 += 1
                        break

            # if len(features[2]) != 0:
            if is_detection_cam2:
                # features = frames[person_counter_cam2][0]

                # Aktuelle Frames:
                frame_cam2 = features[2][0]

                mid_hip_xy_cam2 = features[2][1:]

                for group_id in range(len(groups)):
                    # Nächster Frame bzw gesamtes Bild wird eingelesen:
                    temp_frames_cam2_path = path_extracted_persons + str(groups[group_id]) + "\\cam_2_frame_" + str(frame_cam2) + "_" + str(person_counter_cam2+1) + ".png"

                    if os.path.isfile(temp_frames_cam2_path):
                        cam2_persons.append([frame_cam2, mid_hip_xy_cam2, group_id])

                        if show_img:
                            cv2.imshow(groups[group_id] + " = " + str(group_id), cv2.imread(temp_frames_cam2_path))
                            cv2.waitKey()

                        person_counter_cam2 += 1
                        break

            # if len(features[3]) != 0:
            if is_detection_cam3:
                # features = frames[person_counter_cam3][0]

                # Aktuelle Frames:
                frame_cam3 = features[3][0]

                mid_hip_xy_cam3 = features[3][1:]

                for group_id in range(len(groups)):
                    # Nächster Frame bzw gesamtes Bild wird eingelesen:
                    temp_frames_cam3_path = path_extracted_persons + str(groups[group_id]) + "\\cam_3_frame_" + str(frame_cam3) + "_" + str(person_counter_cam3+1) + ".png"
                    # print("temp_frames_cam3_path =", temp_frames_cam3_path)
                    if os.path.isfile(temp_frames_cam3_path):
                        cam3_persons.append([frame_cam3, mid_hip_xy_cam3, group_id])

                        if show_img:
                            cv2.imshow(groups[group_id] + " = " + str(group_id), cv2.imread(temp_frames_cam3_path))
                            cv2.waitKey()

                        person_counter_cam3 += 1
                        break

            # print("mid_hip_xy_cam1 =", mid_hip_xy_cam1, "mid_hip_xy_cam2 =", mid_hip_xy_cam2, "mid_hip_xy_cam3 =", mid_hip_xy_cam3, "\n")
        """
        print("cam1_persons =", cam1_persons)
        print("cam2_persons =", cam2_persons)
        print("cam3_persons =", cam3_persons)
        """

        """
        Hier machen wir als nächstes weiter:
            1. Lege die folgenden drei Listen an: 
                    blue_Button_coordinates, 
                    green_Button_coordinates, 
                    red_Button_coordinates
            2. In cam1_persons, cam2_persons und cam3_persons steht die Framenummer inkl MidHip Koordinaten und group_id.
               Daher sollten wir vermutlich über alle drei Listen insgesamt drei mal (für jede Gruppe ein Mal) iterieren, um nach der jeweiligen Gruppe zu suchen und  
                
        """

        blue_Button_coordinates_cam1 = [elem for elem in cam1_persons if elem[-1] == 0]
        green_Button_coordinates_cam1 = [elem for elem in cam1_persons if elem[-1] == 1]
        red_Button_coordinates_cam1 = [elem for elem in cam1_persons if elem[-1] == 2]

        blue_Button_coordinates_cam2 = [elem for elem in cam2_persons if elem[-1] == 0]
        green_Button_coordinates_cam2 = [elem for elem in cam2_persons if elem[-1] == 1]
        red_Button_coordinates_cam2 = [elem for elem in cam2_persons if elem[-1] == 2]

        blue_Button_coordinates_cam3 = [elem for elem in cam3_persons if elem[-1] == 0]
        green_Button_coordinates_cam3 = [elem for elem in cam3_persons if elem[-1] == 1]
        red_Button_coordinates_cam3 = [elem for elem in cam3_persons if elem[-1] == 2]

        blue_Button_coordinates = blue_Button_coordinates_cam1 + blue_Button_coordinates_cam2 + blue_Button_coordinates_cam3
        green_Button_coordinates = green_Button_coordinates_cam1 + green_Button_coordinates_cam2 + green_Button_coordinates_cam3
        red_Button_coordinates = red_Button_coordinates_cam1 + red_Button_coordinates_cam2 + red_Button_coordinates_cam3

        if len(blue_Button_coordinates) == 3:
            classified_persons.append(blue_Button_coordinates)
            print("blue_Button_coordinates =", blue_Button_coordinates)
            num_blue_prints += 1

        if len(green_Button_coordinates) == 3:
            classified_persons.append(green_Button_coordinates)
            print("green_Button_coordinates =", green_Button_coordinates)
            num_green_prints += 1

        if len(red_Button_coordinates) == 3:
            classified_persons.append(red_Button_coordinates)
            print("red_Button_coordinates =", red_Button_coordinates)
            num_red_prints += 1

    print("num_blue_prints =", num_blue_prints)
    print("green_Button_coordinates =", num_green_prints)
    print("red_Button_coordinates =", num_red_prints)

    pkl_file = open(data_dir + 'Classified_Persons.pkl', 'wb')
    pickle.dump(classified_persons, pkl_file)

    print("\n\n\n")
    print(classified_persons)

classify_offline()









def extract_all_persons_detections():
    huge_feature_matricies = read_feature_matrices_from_pickle_file()

    # Daten liegen auf der externen Festplatte werden aber intern gespeichert. Das ist der Unterschied zwischen data_dir_ext und data_dir:
    path_frames_cam1 = data_dir_ext + "Frames_extracted_cam_1\\"
    path_frames_cam2 = data_dir_ext + "Frames_extracted_cam_2\\"
    path_frames_cam3 = data_dir_ext + "Frames_extracted_cam_3\\"

    save_dir = data_dir + "Skeletons_extracted_around_MidHip_all_detections\\"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    use_minmax = False

    # handsorted_group_histograms = calc_group_histogramms()

    # img_counter wird nur für die Terminalausgabe benötigt:
    img_counter = 0

    for frames in huge_feature_matricies:
        # print("curruent feature matrix =", frames)
        person_counter = 0





        for person_id in range(len(frames)):
            img_counter += 1
            print("img_counter =", img_counter)


            person_counter += 1

            row = frames[person_id]
            # print("row =", row)
            features = row[0]
            labels = row[1]

            is_detection_cam1 = len(features[1]) > 0
            is_detection_cam2 = len(features[2]) > 0
            is_detection_cam3 = len(features[3]) > 0

            """
            print("is_detection_cam1 =", is_detection_cam1)
            print("is_detection_cam2 =", is_detection_cam2)
            print("is_detection_cam3 =", is_detection_cam3)
            """

            frame_number = features[0]
            # print("Kamera 2 frame number:")

            mid_hip_xy_cam1 = features[1]
            mid_hip_xy_cam2 = features[2]
            mid_hip_xy_cam3 = features[3]

            if not use_minmax:

                # if len(features[1]) != 0:
                if is_detection_cam1:
                    # print("features =", features)

                    # Aktuelle Frames:
                    frame_cam1 = features[1][0]
                    # print("frame_cam1 =", frame_cam1)


                    min_x_cam1 = max(int(mid_hip_xy_cam1[1]) - crop_size_width, 0)
                    min_y_cam1 = max(int(mid_hip_xy_cam1[2]) - crop_size_height, 0)
                    max_x_cam1 = min(int(mid_hip_xy_cam1[1]) + crop_size_width, width)
                    max_y_cam1 = min(int(mid_hip_xy_cam1[2]) + crop_size_height, height)

                    # Nächster Frame bzw gesamtes Bild wird eingelesen:
                    temp_frames_cam1_path = path_frames_cam1 + "frame_" + str(frame_cam1) + ".png"

                    person_cam1 = cv2.imread(temp_frames_cam1_path)[min_y_cam1:max_y_cam1, min_x_cam1:max_x_cam1]
                    cv2.imwrite(save_dir + "cam_1_frame_" + str(frame_cam1) + "_" + str(person_counter) + ".png", person_cam1)

                if is_detection_cam2:
                    frame_cam2 = features[2][0]

                    min_x_cam2 = max(int(mid_hip_xy_cam2[1]) - crop_size_width, 0)
                    min_y_cam2 = max(int(mid_hip_xy_cam2[2]) - crop_size_height, 0)
                    max_x_cam2 = min(int(mid_hip_xy_cam2[1]) + crop_size_width, width)
                    max_y_cam2 = min(int(mid_hip_xy_cam2[2]) + crop_size_height, height)

                    temp_frames_cam2_path = path_frames_cam2 + "frame_" + str(frame_cam2) + ".png"
                    person_cam2 = cv2.imread(temp_frames_cam2_path)[min_y_cam2:max_y_cam2, min_x_cam2:max_x_cam2]
                    cv2.imwrite(save_dir + "cam_2_frame_" + str(frame_cam2) + "_" + str(person_counter) + ".png", person_cam2)

                if is_detection_cam3:
                    frame_cam3 = features[3][0]

                    min_x_cam3 = max(int(mid_hip_xy_cam3[1]) - crop_size_width, 0)
                    min_y_cam3 = max(int(mid_hip_xy_cam3[2]) - crop_size_height, 0)
                    max_x_cam3 = min(int(mid_hip_xy_cam3[1]) + crop_size_width, width)
                    max_y_cam3 = min(int(mid_hip_xy_cam3[2]) + crop_size_height, height)

                    temp_frames_cam3_path = path_frames_cam3 + "frame_" + str(frame_cam3) + ".png"
                    person_cam3 = cv2.imread(temp_frames_cam3_path)[min_y_cam3:max_y_cam3, min_x_cam3:max_x_cam3]
                    cv2.imwrite(save_dir + "cam_3_frame_" + str(frame_cam3) + "_" + str(person_counter) + ".png", person_cam3)









                # print("Bildquelle:", temp_frames_cam1_path)







                """
                print("min_x_cam1 =", min_x_cam1)
                print("min_y_cam1 =", min_y_cam1)
                print("max_x_cam1 =", max_x_cam1)
                print("max_y_cam1 =", max_y_cam1)

                print("min_x_cam2 =", min_x_cam2)
                print("min_y_cam2 =", min_y_cam2)
                print("max_x_cam2 =", max_x_cam2)
                print("max_y_cam2 =", max_y_cam2)

                print("min_x_cam3 =", min_x_cam3)
                print("min_y_cam3 =", min_y_cam3)
                print("max_x_cam3 =", max_x_cam3)
                print("max_y_cam3 =", max_y_cam3)
                """




                """
                cv2.imshow("person_cam1", person_cam1)
                cv2.imshow("person_cam2", person_cam2)
                cv2.imshow("person_cam3", person_cam3)
                cv2.waitKey()
                """
            else:
                # minmax Ansatz
                """
                print("minmax_cam1 =", minmax_cam1)
                print("minmax_cam1[0] =", minmax_cam1[0])
                print("minmax_cam1[1] =", minmax_cam1[1])
                print("minmax_cam1[2] =", minmax_cam1[2])
                print("minmax_cam1[3] =", minmax_cam1[3])
                print("minmax_cam1[4] =", minmax_cam1[4])
                """

                minmax = features[4]

                minmax_cam1 = minmax[0]
                minmax_cam2 = minmax[1]
                minmax_cam3 = minmax[2]

                # Beginn Brechnung der Hoehe und Breite des Skeletons: --------------------------------------------------------
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



                if not os.path.isdir(data_dir + "Skeletons_extracted\\"):
                    os.makedirs(data_dir + "Skeletons_extracted\\")

                # Falls eine das zugeschnittene Teilbild eine Breite oder Höhe von 0 hat, schreiben wir das Bild natürlich nicht (kann passieren, wenn min == max).
                if cam1_x_size > 0 and cam1_y_size > 0:
                    temp_frames_cam1_path = path_frames_cam1 + "frame_" + str(minmax_cam1[0]) + ".png"
                    print(temp_frames_cam1_path)
                    person_cam1 = cv2.imread(temp_frames_cam1_path)[min_y_cam1:max_y_cam1, min_x_cam1:max_x_cam1]
                    cv2.imwrite(data_dir + "Skeletons_extracted\\cam_1_frame_" + str(minmax_cam1[0]) + "_" + str(person_counter) + ".png", person_cam1)

                if cam2_x_size > 0 and cam2_y_size > 0:
                    temp_frames_cam2_path = path_frames_cam2 + "frame_" + str(minmax_cam2[0]) + ".png"
                    person_cam2 = cv2.imread(temp_frames_cam2_path)[min_y_cam2:max_y_cam2, min_x_cam2:max_x_cam2]
                    cv2.imwrite(data_dir + "Skeletons_extracted\\cam_2_frame_" + str(minmax_cam2[0]) + "_" + str(person_counter) + ".png", person_cam2)

                if cam3_x_size > 0 and cam3_y_size > 0:
                    temp_frames_cam3_path = path_frames_cam3 + "frame_" + str(minmax_cam3[0]) + ".png"
                    person_cam3 = cv2.imread(temp_frames_cam3_path)[min_y_cam3:max_y_cam3, min_x_cam3:max_x_cam3]
                    cv2.imwrite(data_dir + "Skeletons_extracted\\cam_3_frame_" + str(minmax_cam3[0]) + "_" + str(person_counter) + ".png", person_cam3)

                """
                cv2.imshow("person_cam1", person_cam1)
                cv2.imshow("person_cam2", person_cam2)
                cv2.imshow("person_cam3", person_cam3)
                cv2.waitKey()
                """

                # print("minmax_cam1 =", minmax_cam1, "\tminmax_cam2 =", minmax_cam2, "\tminmax_cam3 =", minmax_cam3)
                # print("frame number =", frame_number, "\tmid_hip_xy_cam1 =", mid_hip_xy_cam1, "\tmid_hip_xy_cam2 =", mid_hip_xy_cam2, "\tmid_hip_xy_cam3 =", mid_hip_xy_cam3, "\tminmax =", minmax, "\tlabels =", labels)


# extract_all_persons_detections()






def extract_person():
    huge_feature_matricies = read_feature_matrices_from_pickle_file()

    # Daten liegen auf der externen Festplatte werden aber intern gespeichert. Das ist der Unterschied zwischen data_dir_ext und data_dir:
    path_frames_cam1 = data_dir_ext + "Frames_extracted_cam_1\\"
    path_frames_cam2 = data_dir_ext + "Frames_extracted_cam_2\\"
    path_frames_cam3 = data_dir_ext + "Frames_extracted_cam_3\\"

    save_dir = data_dir + "Skeletons_extracted_around_MidHip\\"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    use_minmax = False

    # handsorted_group_histograms = calc_group_histogramms()

    for frames in huge_feature_matricies:
        # print("curruent feature matrix =", frames)
        """
        Mögliche Ausgabe:
        curruent feature matrix = [
        [[7426, [1199.5, 80.808], [558.858, 404.081], [153.279, 431.519], [[7265, 1138.79, 1228.76, 73.0296, 261.072], [7426, 419.777, 743.012, 180.797, 437.382], [7013, 39.6315, 274.698, 143.521, 715.632]]], ['W', 'W', nan]], 
        [[7426, [0.0, 0.0], [0.0, 0.0], [837.07, 362.945], [[7265, 1011.46, 1260.21, 153.334, 699.899], [7426, 4.39787, 120.007, 78.8985, 458.876], [7013, 758.67, 923.283, 139.604, 478.538]]], ['W', 'W', nan]], 
        [[7426, [0.0, 0.0], [1123.12, 578.438], [870.379, 462.901], [[7265, 631.337, 909.524, 223.821, 447.173], [7426, 813.521, 1246.5, 73.0046, 602.019], [7013, 823.322, 1068.19, 259.051, 492.302]]], ['W', 'W', nan]]
                                    ]
        
        Extrahiere zunächst alle minmax-Werte aller drei Kameras und von allen erkannten Personen pro Frame.
        
        Anschließend erzeuge die entsprechenden Teilbilder der minmax-Werte und übergib sie dem Histogramm Klassifizierer.
        
        Der Histogramm Klassifizierer soll für ein übergebenes Bild eine der Klassen "Blue Button", "Green Button", "Red Button", "Dummy", "Schwarze Weste", "Sonstiges" zurückgeben
        
        Entferne die MidHip Koordinaten, die als "Dummy", "Schwarze Weste" oder "Sonstiges" klassifiziert wurde.
        
        Jetzt wissen wir, dass z.B. für den ersten von drei Einträgen von Frame t, die Kamera 1 "Red Button" erkannt hat, dann ist MidHip_cam_1_xy das Feature zu dem Bild und
        wir schauen in der interpolierten Excel Label-Datei (siehe in labels.py die Funktion read_timestamp()) welches Label (W, LH, RF etc) die Person "Red Button" trägt. Auf diese
        Weise erstellen wir unsere reduzierte Feature Matrix für den Frame t.
        
        Beachte, dass wir für drei Einträge pro Frame, 9 Teilbilder Klassifizieren müssen und alle MidHip Koordinaten so sortieren müssen, dass pro neue Feature Matrix Zeile
        nur die Koordinaten vorkommen, die zur gleichen Person gehören.  
        """
        person_counter = 0
        for row in frames:
            person_counter += 1
            # print(person_counter, "of", len(frames))

            features = row[0]
            labels = row[1]
            # print(features)

            frame_number = features[0]
            print("Kamera 2 frame number =", frame_number)
            mid_hip_xy_cam1 = features[1]
            mid_hip_xy_cam2 = features[2]
            mid_hip_xy_cam3 = features[3]

            # print("minmax =", minmax)






            if not use_minmax:
                # Aktuelle Frames:
                frame_cam1 = mid_hip_xy_cam1[0]
                frame_cam2 = mid_hip_xy_cam2[0]
                frame_cam3 = mid_hip_xy_cam3[0]

                # Nächster Frame bzw gesamtes Bild wird eingelesen:
                temp_frames_cam1_path = path_frames_cam1 + "frame_" + str(frame_cam1) + ".png"
                temp_frames_cam2_path = path_frames_cam2 + "frame_" + str(frame_cam2) + ".png"
                temp_frames_cam3_path = path_frames_cam3 + "frame_" + str(frame_cam3) + ".png"

                # print("Bildquelle:", temp_frames_cam1_path)

                min_x_cam1 = max(int(mid_hip_xy_cam1[1]) - crop_size_width, 0)
                min_y_cam1 = max(int(mid_hip_xy_cam1[2]) - crop_size_height, 0)
                max_x_cam1 = min(int(mid_hip_xy_cam1[1]) + crop_size_width, width)
                max_y_cam1 = min(int(mid_hip_xy_cam1[2]) + crop_size_height, height)

                min_x_cam2 = max(int(mid_hip_xy_cam2[1]) - crop_size_width, 0)
                min_y_cam2 = max(int(mid_hip_xy_cam2[2]) - crop_size_height, 0)
                max_x_cam2 = min(int(mid_hip_xy_cam2[1]) + crop_size_width, width)
                max_y_cam2 = min(int(mid_hip_xy_cam2[2]) + crop_size_height, height)

                min_x_cam3 = max(int(mid_hip_xy_cam3[1]) - crop_size_width, 0)
                min_y_cam3 = max(int(mid_hip_xy_cam3[2]) - crop_size_height, 0)
                max_x_cam3 = min(int(mid_hip_xy_cam3[1]) + crop_size_width, width)
                max_y_cam3 = min(int(mid_hip_xy_cam3[2]) + crop_size_height, height)

                """
                print("min_x_cam1 =", min_x_cam1)
                print("min_y_cam1 =", min_y_cam1)
                print("max_x_cam1 =", max_x_cam1)
                print("max_y_cam1 =", max_y_cam1)

                print("min_x_cam2 =", min_x_cam2)
                print("min_y_cam2 =", min_y_cam2)
                print("max_x_cam2 =", max_x_cam2)
                print("max_y_cam2 =", max_y_cam2)

                print("min_x_cam3 =", min_x_cam3)
                print("min_y_cam3 =", min_y_cam3)
                print("max_x_cam3 =", max_x_cam3)
                print("max_y_cam3 =", max_y_cam3)
                """

                person_cam1 = cv2.imread(temp_frames_cam1_path)[min_y_cam1:max_y_cam1, min_x_cam1:max_x_cam1]
                person_cam2 = cv2.imread(temp_frames_cam2_path)[min_y_cam2:max_y_cam2, min_x_cam2:max_x_cam2]
                person_cam3 = cv2.imread(temp_frames_cam3_path)[min_y_cam3:max_y_cam3, min_x_cam3:max_x_cam3]

                cv2.imwrite(save_dir + "cam_1_frame_" + str(frame_cam1) + "_" + str(person_counter) + ".png", person_cam1)
                cv2.imwrite(save_dir + "cam_2_frame_" + str(frame_cam2) + "_" + str(person_counter) + ".png", person_cam2)
                cv2.imwrite(save_dir + "cam_3_frame_" + str(frame_cam3) + "_" + str(person_counter) + ".png", person_cam3)
                """
                cv2.imshow("person_cam1", person_cam1)
                cv2.imshow("person_cam2", person_cam2)
                cv2.imshow("person_cam3", person_cam3)
                cv2.waitKey()
                """
            else:
                # minmax Ansatz
                """
                print("minmax_cam1 =", minmax_cam1)
                print("minmax_cam1[0] =", minmax_cam1[0])
                print("minmax_cam1[1] =", minmax_cam1[1])
                print("minmax_cam1[2] =", minmax_cam1[2])
                print("minmax_cam1[3] =", minmax_cam1[3])
                print("minmax_cam1[4] =", minmax_cam1[4])
                """

                minmax = features[4]
                
                minmax_cam1 =  minmax[0]
                minmax_cam2 = minmax[1]
                minmax_cam3 = minmax[2]
                
                # Beginn Brechnung der Hoehe und Breite des Skeletons: --------------------------------------------------------
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
                
    
    
                if not os.path.isdir(data_dir + "Skeletons_extracted\\"):
                    os.makedirs(data_dir + "Skeletons_extracted\\")
    
                # Falls eine das zugeschnittene Teilbild eine Breite oder Höhe von 0 hat, schreiben wir das Bild natürlich nicht (kann passieren, wenn min == max).
                if cam1_x_size > 0 and cam1_y_size > 0:
                    temp_frames_cam1_path = path_frames_cam1 + "frame_" + str(minmax_cam1[0]) + ".png"
                    print(temp_frames_cam1_path)
                    person_cam1 = cv2.imread(temp_frames_cam1_path)[min_y_cam1:max_y_cam1, min_x_cam1:max_x_cam1]
                    cv2.imwrite(data_dir + "Skeletons_extracted\\cam_1_frame_" + str(minmax_cam1[0]) + "_" + str(person_counter) +  ".png" ,person_cam1)
    
                if cam2_x_size > 0 and cam2_y_size > 0:
                    temp_frames_cam2_path = path_frames_cam2 + "frame_" + str(minmax_cam2[0]) + ".png"
                    person_cam2 = cv2.imread(temp_frames_cam2_path)[min_y_cam2:max_y_cam2, min_x_cam2:max_x_cam2]
                    cv2.imwrite(data_dir + "Skeletons_extracted\\cam_2_frame_" + str(minmax_cam2[0]) + "_" + str(person_counter) +  ".png", person_cam2)
    
                if cam3_x_size > 0 and cam3_y_size > 0:
                    temp_frames_cam3_path = path_frames_cam3 + "frame_" + str(minmax_cam3[0]) + ".png"
                    person_cam3 = cv2.imread(temp_frames_cam3_path)[min_y_cam3:max_y_cam3, min_x_cam3:max_x_cam3]
                    cv2.imwrite(data_dir + "Skeletons_extracted\\cam_3_frame_" + str(minmax_cam3[0]) + "_" + str(person_counter) +  ".png", person_cam3)


                """
                cv2.imshow("person_cam1", person_cam1)
                cv2.imshow("person_cam2", person_cam2)
                cv2.imshow("person_cam3", person_cam3)
                cv2.waitKey()
                """

                # print("minmax_cam1 =", minmax_cam1, "\tminmax_cam2 =", minmax_cam2, "\tminmax_cam3 =", minmax_cam3)
                # print("frame number =", frame_number, "\tmid_hip_xy_cam1 =", mid_hip_xy_cam1, "\tmid_hip_xy_cam2 =", mid_hip_xy_cam2, "\tmid_hip_xy_cam3 =", mid_hip_xy_cam3, "\tminmax =", minmax, "\tlabels =", labels)


# extract_person()


"""
# Falls mal ein Bild kaputt sein sollte (z.B. wegen Breite oder Höhe = 0)
def main2():
    huge_feature_matricies = read_feature_matrices_from_pickle_file()

    path_frames_cam1 = data_dir + "Frames_extracted_cam_1\\"

    for i in range(len(huge_feature_matricies)):
        # cam_1_frame_10426_3
        frames = huge_feature_matricies[3190]

        person_counter = 0
        for row in frames:
            person_counter += 1
            # print(person_counter, "of", len(frames))

            features = row[0]
            labels = row[1]
            print(features)

            frame_number = features[0]
            print("Kamera 2 frame number =", frame_number)

            minmax = features[4]

            minmax_cam1 = minmax[0]

            min_x_cam1 = int(minmax_cam1[1])
            max_x_cam1 = int(minmax_cam1[2])
            min_y_cam1 = int(minmax_cam1[3])
            max_y_cam1 = int(minmax_cam1[4])

            cam1_x_size = max_x_cam1 - min_x_cam1
            cam1_y_size = max_y_cam1 - min_y_cam1
            
            print("min_x_cam1 =", min_x_cam1)
            print("max_x_cam1 =", max_x_cam1)
            print("min_y_cam1 =", min_y_cam1)
            print("max_y_cam1 =", max_y_cam1)

            if cam1_x_size > 0 and cam1_y_size > 0:
                temp_frames_cam1_path = path_frames_cam1 + "frame_" + str(minmax_cam1[0]) + ".png"


                person_cam1 = cv2.imread(temp_frames_cam1_path)[min_y_cam1:max_y_cam1, min_x_cam1:max_x_cam1]
                temp_filename = "Skeletons_extracted\\cam_1_frame_" + str(minmax_cam1[0]) + "_" + str(person_counter) + ".png"
                print(temp_filename)
                cv2.imshow("Test" + str(person_counter), person_cam1)
                cv2.waitKey()
                # cv2.imwrite(data_dir + temp_filename, person_cam1)
        break
main2()
"""