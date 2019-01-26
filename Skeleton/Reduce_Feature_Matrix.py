import os
import glob
import numpy as np
import pickle
import cv2

from Skeleton import Global_Constants


main_dir = Global_Constants.main_dir
git_dir = Global_Constants.git_dir
project_dir = Global_Constants.project_dir
data_dir = Global_Constants.data_dir
data_dir_ext = Global_Constants.data_dir_ext
filename = Global_Constants.filename
output_dir = Global_Constants.output_dir
skeletons_dir = Global_Constants.skeletons_dir





def read_pickle_files():
    print("Read Feature Matrix ...")
    pkl_file1 = open(data_dir + 'Feature_Matrices.pkl', "rb")
    huge_feature_matrix = pickle.load(pkl_file1)
    print("Done. \n")
    print("Read List of classified Persons ...")
    pkl_file2 = open(data_dir + 'Classified_Persons.pkl', "rb")
    classified_persons = pickle.load(pkl_file2)
    print("Done.\n\n")

    return [huge_feature_matrix, classified_persons]



def find_next_entry(huge_feature_matrix, frame_cam2, start_id):

    # Für den Fall, dass die huge_feature_matrix ein bestimmtes Frame nicht enthält. Das kann passieren, wenn es keine passenden synchronisieren Frames gibt.
    # Dann werden die Labels aus dem Frame zuvor genommen. Bsp.: Der Frame 26217 fehlt, also nehmen wir die Labels vom Frame 26216. Grund: So schnell können sich die Personen nicht bewegen, dass dadurch die
    # Ergebnisse dadruch sehr viel schlechter werden würden.
    last_labels = []

    for i in range(start_id, len(huge_feature_matrix)):
        elem = huge_feature_matrix[i]
        first_entry = elem[0]
        first_feature = first_entry[0]
        labels = first_entry[1]

        """
        if frame_cam2 == 26317:
            print("first_entry =", first_entry)
        """

        """
        print("elem =", elem)
        print("first_entry =", first_entry)
        print("first_feature =", first_feature)
        print("labels =", labels)
        """

        if first_feature[0] == frame_cam2:
            return [labels, i]

        if first_feature[0] > frame_cam2:
            return [last_labels, i]

        last_labels = labels


    return [[], -1]


def main():
    huge_feature_matrix, classified_persons = read_pickle_files()
    """
    print("huge_feature_matrix:", huge_feature_matrix)
    print("\n\n\n")
    print("classified_persons:", classified_persons)
    """
    last_search_id = 0
    new_feature_matrix = []

    for elem in classified_persons:
        cam1_entries = elem[0]
        cam2_entries = elem[1]
        cam3_entries = elem[2]

        cam1_frame = cam1_entries[0]
        cam2_frame = cam2_entries[0]
        cam3_frame = cam3_entries[0]

        cam1_coordinates = cam1_entries[1]
        cam2_coordinates = cam2_entries[1]
        cam3_coordinates = cam3_entries[1]

        person_id = cam1_entries[-1]

        print("cam2_frame =", cam2_frame)

        """
        So kann z.B. ein Label aussehen:
                ['W', 'W', nan]
                
        Hierbei steht der:
            0. Eintrag stets für Green Button
            1. Eintrag stets für Red Button
            2. Eintrag stets für Blue Button
        siehe hierzu auch die Labels D2_S2.xlsx Datei.
        """

        labels, _ = find_next_entry(huge_feature_matrix, cam2_frame, last_search_id)

        label = 'unknown'

        # Leider haben wir in der classified_persons Liste eine andere Personenbezeichnung wie in der Excel Datei. In der classified_persons gilt:
        # 0 entspricht Blue Button
        # 1 entspricht Green Button
        # 2 entspricht Red Button
        # Dies kommt daher, dass wir beim Klassifizieren die Ordnung in eine Liste geschrieben haben und uns nicht bewusst war, dass diese Reihenfolge irgendwann mal wieder
        # eine Rolle spielen könnte. Kann man aber in Histogramms.py ändern, dann muss man aber alle Daten neu Klassifizieren lassen.

        new_person_id = -1

        possible_labels = ['B2', 'CO', 'CT', 'F', 'H', 'LF', 'LH', 'MO', 'RF', 'RH', 'ST', 'W']

        # 0 entspricht in classified_persons Blue Button, ist in der Excel Datei aber an Position 2:
        if person_id == 0 and labels[2] in possible_labels:
            label = labels[2]
            new_person_id = 2

        # 1 entspricht in classified_persons Green Button, ist in der Excel Datei aber an Position 0:
        if person_id == 1 and labels[0] in possible_labels:
            label = labels[0]
            new_person_id = 0

        # 2 entspricht in classified_persons Red Button, ist in der Excel Datei aber an Position 1:
        if person_id == 2 and labels[1] in possible_labels:
            label = labels[1]
            new_person_id = 1

        if label != 'unknown':
            # if last_search_id != -1:
            """
            print("person_id =", person_id)
            """
            print("elem =", elem)
            print("labels =", labels)

            minute = int(float(cam2_frame)/(25.0*60.0))
            sekunde = int(float(cam2_frame - (minute*25*60))/25.0)
            print("Person", person_id, "wurde in Kamera 2 Frame", cam2_frame, "(=" , minute, ":", sekunde ,")","an der Stelle", label, "erkannt.")

            # In der Ausgabedatei ist der zweitletzte Eintrag die Personennummer, passend zur Excel Datei.
            next_row = [cam2_frame, cam1_coordinates, cam2_coordinates, cam3_coordinates, new_person_id, label]
            new_feature_matrix.append(next_row)
            print("next_row =", next_row, "\n\n")

    pkl_file = open(data_dir + 'Final_Feature_Matrix.pkl', 'wb')
    pickle.dump(new_feature_matrix, pkl_file)



"""
Achtung:
Die Datei "cam_1_frame_17843_2.png" liegt in Blue Button, darauf ist aber eigentlich Green Button zu sehen. Wenn der Kopiervorgang auf die externe Festplatte abgeschlossen ist, nochmal manuell durchgehen und
falsch klassifizierte Bilder richtig einordnen. Vermutlich entspricht dieses Bild dem falschen Blauen Punkt.
"""



if __name__ == '__main__':
    main()