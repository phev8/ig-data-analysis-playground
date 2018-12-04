import os
import csv
import numpy as np
from location_testing import explore_labels_vs_person_detections

main_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
git_dir = str(os.path.abspath(os.path.join(main_dir, os.pardir)))
project_dir = str(os.path.abspath(os.path.join(git_dir, os.pardir)))
data_dir = project_dir + "\\Daten\\igroups_student_project\\"
filename = project_dir + "\\Daten\\igroups_student_project\\videos\\cam_2.mp4"
output_dir = project_dir + "\\Daten\\igroups_student_project\\Frames_extracted\\"
skeletons_dir = data_dir + "processed_data\\images\\skeletons\\"


def get_project_dir():
    return project_dir


def get_data_dir():
    return data_dir


def get_video_path():
    return filename


def get_output_dir():
    return output_dir



def read_skeleton_csv(cam_number):
    filename = skeletons_dir + "cam_"+ str(cam_number) + "_poses.csv"

    file = open(filename, 'r')
    csv_Reader = list(csv.reader(file, delimiter=','))
    header = csv_Reader[0]
    data = csv_Reader[1:]

    # Find the column numbers for: Image_ID, Person_ID, MidHip_x, MidHip_y
    important_columns = []
    for i in range(len(header)):
        if header[i] == 'Image_ID' or header[i] == 'Person_ID' or header[i] == 'MidHip_x' or header[i] == 'MidHip_y':
            print(i)
            important_columns.append(i)

    # for i in range(len(header)):

    csv_Reader = np.array(csv_Reader)
    erg = [list(entry[important_columns]) for entry in csv_Reader]
    print(erg[0:3])
    res = []
    for i in range(1,len(erg)):
        frame_number = int(erg[i][0].split(':')[-1].split('.')[0])
        person_id = erg[i][1]
        mid_Hip_x = float(erg[i][2])
        mid_Hip_y = float(erg[i][3])
        # print("frame_number =", frame_number, "\tperson_id =", person_id, "\tMid_Hip_x =", mid_Hip_x, "\tMid_Hip_y =", mid_Hip_y)
        res.append([frame_number,person_id,mid_Hip_x,mid_Hip_y])
        print(res.__getitem__(-1))

    return res




"""
Idea for calculating the syncronisation:
By using an extantion for VLC Player we know the following:
    Frame 19530 from cam 1 = Frame 19691 from cam 2 = Frame 19278 from cam 3
We call the offset of camera i: ti
This gives us:
    t1 + 19530 = t2 + 19691 = t3 + 19278
Since, camera 2 was turned on first, we set t2 := 0. Therefore we get:
    19691 = t1 + 19530 = t3 + 19278
Let xi be an arbitrary frame number from camera i, then it hols:
    x2 = t1 + x1 = t3 + x3
"""


offset_cam1_org_video = 161   # 19691 - 19530
offset_cam3_org_video = 413   # 19691 - 19278

def sync_given_cam1_org_video(frame_from_cam1):
    frame_from_cam2 = frame_from_cam1 + offset_cam1_org_video
    frame_from_cam3 = frame_from_cam1 + offset_cam1_org_video - offset_cam3_org_video
    return [frame_from_cam2, frame_from_cam3]

def sync_given_cam2_org_video(frame_from_cam2):
    frame_from_cam1 = frame_from_cam2 - offset_cam1_org_video
    frame_from_cam3 = frame_from_cam2 - offset_cam3_org_video
    return [frame_from_cam1, frame_from_cam3]

def sync_given_cam3_org_video(frame_from_cam3):
    frame_from_cam1 = frame_from_cam3 + offset_cam3_org_video - offset_cam1_org_video
    frame_from_cam2 = frame_from_cam3 + offset_cam3_org_video
    return [frame_from_cam1, frame_from_cam2]


"""
In the skeleton videos, camera 3 was turned on first (respectively the crop was choosed that way). Therefore, we have t3 := 0 and by this it holds:
    12544 = t1 + 12423 = t2 + 12537
"""

offset_cam1_skeleton = 121      # 12544 - 12423
offset_cam2_skeleton = 7        # 12544 - 12537

def sync_given_cam1_skeleton(frame_from_cam1):
    frame_from_cam2 = frame_from_cam1 + offset_cam1_skeleton - offset_cam2_skeleton
    frame_from_cam3 = frame_from_cam1 + offset_cam1_skeleton
    return [frame_from_cam2, frame_from_cam3]

def sync_given_cam2_skeleton(frame_from_cam2):
    frame_from_cam1 = frame_from_cam2 + offset_cam2_skeleton - offset_cam1_skeleton
    frame_from_cam3 = frame_from_cam2 + offset_cam2_skeleton
    return [frame_from_cam1, frame_from_cam3]

def sync_given_cam3_skeleton(frame_from_cam3):
    frame_from_cam1 = frame_from_cam3 - offset_cam1_skeleton
    frame_from_cam2 = frame_from_cam3 - offset_cam2_skeleton
    return [frame_from_cam1, frame_from_cam2]


offset_cam1_between_skeleton_and_org = 7108      # 19531 - 12423
offset_cam2_between_skeleton_and_org = 7155      # 19692 - 12537
offset_cam3_between_skeleton_and_org = 6735      # 19279 - 12544

def sync_given_cam1_skeleton_to_org(frame_from_cam1_skeleton):
    org_frame_cam1 = frame_from_cam1_skeleton + offset_cam1_between_skeleton_and_org
    return org_frame_cam1

def sync_given_cam2_skeleton_to_org(frame_from_cam2_skeleton):
    org_frame_cam2 = frame_from_cam2_skeleton + offset_cam2_between_skeleton_and_org
    return org_frame_cam2

def sync_given_cam3_skeleton_to_org(frame_from_cam3_skeleton):
    org_frame_cam3 = frame_from_cam3_skeleton + offset_cam3_between_skeleton_and_org
    return org_frame_cam3



def sync_given_cam1_org_to_skeleton(frame_from_cam1_org):
    skeleton_frame_cam1 = frame_from_cam1_org - offset_cam1_between_skeleton_and_org
    return skeleton_frame_cam1

def sync_given_cam2_org_to_skeleton(frame_from_cam2_org):
    skeleton_frame_cam2 = frame_from_cam2_org - offset_cam2_between_skeleton_and_org
    return skeleton_frame_cam2

def sync_given_cam3_org_to_skeleton(frame_from_cam3_org):
    skeleton_frame_cam3 = frame_from_cam3_org - offset_cam3_between_skeleton_and_org
    return skeleton_frame_cam3





# print(sync_given_cam1_org_video(19363))
print(sync_given_cam2_org_video(7375))
print(sync_given_cam1_org_video(25030))

"""
cur_entries_cam1 = [[25030, 'P0', 86.7095, 329.632], [25030, 'P1', 1240.64, 78.8706], [25030, 'P2', 631.317, 347.256]]
cur_entries_cam2 = [[25191, 'P0', 341.407, 313.948], [25191, 'P1', 0.0, 0.0]]
cur_entries_cam3 = [[24778, 'P0', 118.045, 308.138], [24778, 'P1', 1025.05, 411.935], [24778, 'P2', 0.0, 0.0]]
"""


def search_for_frame_number(data, frame, possible_start=0):

    for i in range(possible_start,len(data)):
        cur_frame = data[i][0]
        if cur_frame == frame:
            return [True, i]
        if cur_frame > frame:
            print("You are looking for frame number", frame, " but in this dataset there is no such frame number.")
            return [False, i-1]

def find_all_labels_per_frame(data, frame_number, possible_start=0):
    res = []
    last_index = possible_start
    for i in range(possible_start,len(data)):
        cur_frame = data[i][0]
        print("cur_frame =", cur_frame)
        if cur_frame == frame_number:
            res.append(data[i])
            if len(res) == 1:
                last_index = i
        if cur_frame > frame_number:
            break


    if len(res) > 0:
        return [True, res, last_index]
    else:
        return [False, [], last_index]


def find_next_common_frames(cam1_data, cam2_data, cam3_data, cam, next_frame_number, start_index_cam1, start_index_cam2, start_index_cam3):

    if cam == 3:
        end_frame = cam3_data[-1][0]
        print("end_frame =", end_frame)
        possible_start_cam1 = start_index_cam1
        possible_start_cam2 = start_index_cam2
        possible_start_cam3 = start_index_cam3
        while next_frame_number <= end_frame:
            if search_for_frame_number(cam3_data,next_frame_number,start_index_cam3)[0]:
                cam1_frame, cam2_frame = sync_given_cam3_org_video(next_frame_number)
                print("\t\t\t\tWe are looking for the Triple:", cam1_frame, cam2_frame, next_frame_number, "(cam1, cam2, cam3)")
                print("Looking for next syncronized frame for camera 1")
                temp_cam1 = find_all_labels_per_frame(cam1_data, frame_number=cam1_frame, possible_start=possible_start_cam1)
                print("\tCamera 1 Done.")
                print("Looking for next syncronized frame for camera 2")
                temp_cam2 = find_all_labels_per_frame(cam2_data, frame_number=cam2_frame, possible_start=possible_start_cam2)
                print("\tCamera 2 Done.")
                print("Looking for next syncronized frame for camera 3")
                temp_cam3 = find_all_labels_per_frame(cam3_data, frame_number=next_frame_number, possible_start=possible_start_cam3)
                print("\tCamera 3 Done.")

                possible_start_cam1 = temp_cam1[-1]
                possible_start_cam2 = temp_cam2[-1]
                possible_start_cam3 = temp_cam3[-1]
                print("camera 1 can start by index", possible_start_cam1)
                print("camera 2 can start by index", possible_start_cam2)
                print("camera 3 can start by index", possible_start_cam3)

                if temp_cam1[0] and temp_cam2[0] and temp_cam3[0]:
                    return [[possible_start_cam1, possible_start_cam2, possible_start_cam3], [cam1_frame, cam2_frame, next_frame_number]]

                next_frame_number += 1

    return [False]





def generate_feature_matrix():
    cam1_labels = read_skeleton_csv(1)
    cam2_labels = read_skeleton_csv(2)
    cam3_labels = read_skeleton_csv(3)

    start_cam1 = cam1_labels[0][0]
    start_cam2 = cam2_labels[0][0]
    start_cam3 = cam3_labels[0][0]
    print("Skeleton Labeling for camera 1 starts with frame", start_cam1)
    print("Skeleton Labeling for camera 2 starts with frame", start_cam2)
    print("Skeleton Labeling for camera 3 starts with frame", start_cam3)

    """
    The Labeling File which contains "RH", "LH" etc., starts with frame number 7375 and we guess that camera 2 was used for the timestamps (every other camera would not make sense for the timestamps).
    
    Hence, we use the frame number 7375 as input for syncronising:
    """

    start_frame_cam2 = 7375
    sync_res = sync_given_cam2_org_video(7375)
    start_frame_cam1 = sync_res[0]
    start_frame_cam3 = sync_res[1]

    print(sync_given_cam2_org_video(7375))

    cam3_end_frame = 33461


    start_entry_in_cam1_labels = search_for_frame_number(cam1_labels,start_frame_cam1)[-1]
    start_entry_in_cam2_labels = search_for_frame_number(cam2_labels,start_frame_cam2)[-1]
    start_entry_in_cam3_labels = search_for_frame_number(cam3_labels,start_frame_cam3)[-1]

    print(start_entry_in_cam1_labels)
    print(start_entry_in_cam2_labels)
    print(start_entry_in_cam3_labels)

    cur_frame_cam1 = start_frame_cam1
    cur_frame_cam2 = start_frame_cam2
    cur_frame_cam3 = start_frame_cam3

    all_feature_matricies = []

    while cur_frame_cam3 <= cam3_end_frame:
        # Hier liefert find_all_labels_per_frame() an erster Stelle der Rückgabeliste immer True, da wir die Einträge vorher genau so wählen.
        cur_entries_cam1 = find_all_labels_per_frame(cam1_labels, cur_frame_cam1, start_entry_in_cam1_labels)[1]
        _, cur_entries_cam2, _ = find_all_labels_per_frame(cam2_labels, cur_frame_cam2, start_entry_in_cam2_labels)
        cur_entries_cam3 = find_all_labels_per_frame(cam3_labels, cur_frame_cam3, start_entry_in_cam3_labels)[1]

        print("cur_entries_cam1 =", cur_entries_cam1)
        print("cur_entries_cam2 =", cur_entries_cam2)
        print("cur_entries_cam3 =", cur_entries_cam3)

        smallest_number_of_found_entries = min([len(cur_entries_cam1), len(cur_entries_cam2), len(cur_entries_cam3)])

        feature_matrix = []

        for i in range(smallest_number_of_found_entries):
            cur_row = []
            frame_number_cam2 = cur_entries_cam2[i][0]
            print("frame_number_cam2 =", frame_number_cam2)
            print("cur_entries_cam1[i][2:4] =", cur_entries_cam1[i][2:4])
            print("cur_entries_cam2[i][2:4] =", cur_entries_cam2[i][2:4])
            print("cur_entries_cam3[i][2:4] =", cur_entries_cam3[i][2:4])
            # In die Feature Matrix schreibe ich zur Zeit noch die Framenummer hin. Diese benötigen wir später nicht mehr wenn wir an die Matirx die labels als neue Spalte anhängen.
            # Dazu müssen wir die Labels aber erst mal zuordnen. D.h. Nimm Framnummer von Feature Matrix und suche in der RH, LH Label Datei nach dem entsprechenden Timestamp -> auslesen und als neuen Eintrag
            # hinzufügen.
            cur_row.extend([frame_number_cam2, cur_entries_cam1[i][2:4], cur_entries_cam2[i][2:4], cur_entries_cam3[i][2:4]])
            feature_matrix.append(cur_row)

        print(feature_matrix)
        all_feature_matricies.append(feature_matrix)
        # print(all_feature_matricies)

        # Berechne nächster gemeinsamer Frame, statt einfach um eins auf gut Glück zu erhöhen.
        cur_frame_cam3 += 1
        temp = find_next_common_frames(cam1_labels, cam2_labels, cam3_labels,
                                        cam=3, next_frame_number=cur_frame_cam3,
                                        start_index_cam1=start_entry_in_cam1_labels,
                                        start_index_cam2=start_entry_in_cam2_labels,
                                        start_index_cam3=start_entry_in_cam3_labels)
        if len(temp) == 1:
            break
        start_entry_in_cam1_labels, start_entry_in_cam2_labels, start_entry_in_cam3_labels = temp[0]
        print("+++++++++++", start_entry_in_cam1_labels, start_entry_in_cam2_labels, start_entry_in_cam3_labels)


        cur_frame_cam1, cur_frame_cam2, cur_frame_cam3 = temp[-1]

    print("\n\n\n\nAll Feature Matrices:\n", all_feature_matricies)

    for i in range(len(all_feature_matricies)):
        frame_a = all_feature_matricies[i][0][0]
        frame_b = all_feature_matricies[i+1][0][0]
        print(frame_a)
        print(frame_b)
        if frame_b - frame_a >= 2:
            print("Der Frame", frame_a + 1, "fehlt in der Liste, dies kann daran liegen, dass der Frame in Kamera 2 fehlt oder, dass die äquivalenten syncronisierten Frames",
                  sync_given_cam2_org_video(frame_a + 1) ,"für Kamera 1 oder Kamera 3 fehlen.")
            break

    return all_feature_matricies


def find_index_for_label_frame(all_interpolated_labels, frame, possible_start = 0):
    all_frames = all_interpolated_labels[0]
    all_labeled_fuzzy_positions = all_interpolated_labels[1]

    res = []
    for i in range(possible_start, len(all_frames)):
        if all_frames[i] == frame:
            res.append(all_labeled_fuzzy_positions[i])
        if all_frames[i] > frame and len(res) > 0:
            return [i, res]
        if all_frames[i] > frame and len(res) == 0:
            return [-1, []]

    return [-1, []]


def join_feature_matricies_with_labels():
    feature_matricies = generate_feature_matrix()
    all_interpolated_labels = explore_labels_vs_person_detections.extend_labels()

    print(all_interpolated_labels)

    res = []
    last_index = 0
    for feature_matrix in feature_matricies:
        feature_frame = feature_matrix[0][0]
        last_index, fuzzy_labels = find_index_for_label_frame(all_interpolated_labels, feature_frame, last_index)

        print("last_index =", last_index)
        print("fuzzy_labels =", fuzzy_labels)

        # Falls der Skeleton-Frame größer ist als der gelabelte Frame (das kann passieren, wenn die Skeleton Detections länger liefen als die Labels aufgeschrieben wurden)
        #  Dann können wir direkt aufhören, da danach definitv kein gelabelter Frame mehr vorkommen kann.
        if feature_frame > all_interpolated_labels[0][-1]:
            break
        else:
            if last_index >= 0:
                fuzzy_labels = fuzzy_labels[0].tolist()
                print("feature_matrix =", feature_matrix)
                print("min(len(feature_matrix),len(fuzzy_labels)) =", min(len(feature_matrix),len(fuzzy_labels)))
                temp = []
                for i in range(min(len(feature_matrix),len(fuzzy_labels))):
                    print("feature_matrix[i] =", feature_matrix[i])
                    temp.append([feature_matrix[i], fuzzy_labels])              # Variante 1
                    # temp.append([feature_matrix[i], fuzzy_labels[i]])         # Variante 2
                res.append(temp)                                    # Wird nur für Variante 1 und 2 benötigt
                # res.append([feature_matrix, fuzzy_labels])                   # Variante 3
                # res.append([feature_matrix, fuzzy_labels[0:min(len(feature_matrix),len(fuzzy_labels))]])                     # Variante 4



    print(res)

join_feature_matricies_with_labels()