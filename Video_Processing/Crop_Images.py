import numpy as np
import cv2
import os
from examples import read_person_detections
from Video_Processing import Video_to_Images



def get_bounding_box_positions():

    data = read_person_detections.get_data()


    # print(data)
    # print("----------------------")

    positions = []
    region_data = np.zeros((0, 4))
    for d in data['results']:
        next_entry = []
        for i in range(len(d['class_names'])):
            if d['class_names'][i] == 'person' and d['scores'][i] >= 0.95:
                # print(d['rois'][i])
                next_entry.append(d['rois'][i])
        if not (next_entry == []):
            positions.append([next_entry, d['frame_index']])


    return positions

def crop_images():

    bounding_box_coordinates = get_bounding_box_positions()
    # print("bounding_box_coordinates =", bounding_box_coordinates)

    # frame_counter = bounding_box_coordinates[0][1]
    first_frame = bounding_box_coordinates[0][1]
    last_frame = bounding_box_coordinates[len(bounding_box_coordinates)-1][1]
    # print("First Frame =", first_frame)
    # print("Last Frame =", last_frame)
    img_counter = 1
    bb_id = 0
    for cur_frame in range(first_frame, last_frame+1):
        # print("bb_id =", bb_id)
        # print("cur_frame =", cur_frame)
        next_entry = bounding_box_coordinates[bb_id]
        # print("Next Entry:", next_entry)
        print("Current Frame Number:", next_entry[1])
        if cur_frame == next_entry[1]:
            all_coordinates_next_frame = next_entry[0]
            # print("all_coordinates_next_frame =", all_coordinates_next_frame)

            for j in range(len(next_entry[0])):
                next_frame_coordinates = all_coordinates_next_frame[j]
                # So kann ein Eintrag von [d['rois'][i] aussehen: [189, 1197, 354, 1279]
                x1_temp = next_frame_coordinates[1]
                x2_temp = next_frame_coordinates[3]
                y1_temp = next_frame_coordinates[0]
                y2_temp = next_frame_coordinates[2]

                breite = np.abs(x1_temp - x2_temp)
                hoehe = np.abs(y1_temp - y2_temp)

                if y1_temp < y2_temp:
                    y_start = y1_temp
                    y_ende = y2_temp
                else:
                    y_start = y2_temp
                    y_ende = y1_temp

                if x1_temp < x2_temp:
                    x_start = x1_temp
                    x_ende = x2_temp
                else:
                    x_start = x2_temp
                    x_ende = x1_temp

                # print("next_entry =", next_entry)

                # print("next_frame_coordinates =", next_frame_coordinates)

                # print(x1_temp)
                # print(y1_temp)
                # print(x2_temp)
                # print(y2_temp)

                # print("Breite =", breite)
                # print("Höhe =", hoehe)

                image_dir = Video_to_Images.get_output_dir()
                output_dir = Video_to_Images.get_data_dir() + "Persons_extracted\\"

                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                temp_img = image_dir + "frame_" + str(cur_frame) + ".png"
                img = cv2.imread(temp_img)
                # crop_img = img[y_start:(y_start + hoehe), x_start:(x_start + breite)]
                crop_img = img[y_start:y_ende, x_start:x_ende]
                filename = "Frame_" + str(cur_frame) + "_Person_" + str(j) + ".png"
                cv2.imwrite(output_dir + filename, crop_img)
                img_counter += 1
            bb_id += 1


crop_images()




"""
    next_entry = bounding_box_coordinates[0]
    all_coordinates_next_frame = bounding_box_coordinates[0][0]
    next_frame_coordinates = bounding_box_coordinates[0][0][0]

    # So kann ein Eintrag von [d['rois'][i] aussehen: [189, 1197, 354, 1279]
    x1_temp = next_frame_coordinates[1]
    x2_temp = next_frame_coordinates[3]
    y1_temp = next_frame_coordinates[0]
    y2_temp = next_frame_coordinates[2]

    breite = np.abs(x1_temp - x2_temp)
    hoehe = np.abs(y1_temp - y2_temp)



    print(bounding_box_coordinates)
    print(next_entry)
    print(all_coordinates_next_frame)
    print(next_frame_coordinates)

    print(x1_temp)
    print(y1_temp)
    print(x2_temp)
    print(y2_temp)

    print("Breite =", breite)
    print("Höhe =", hoehe)

    image_dir = Video_to_Images.get_output_dir()
    output_dir = Video_to_Images.get_data_dir() + "Persons_extracted"
    temp_img = image_dir + "frame_1.png"
    img = cv2.imread(temp_img)
    crop_img = img[y1_temp:y1_temp + hoehe, x1_temp:x1_temp + breite]
    cv2.imwrite(output_dir + "Person_Img_" + str(1) + ".png", crop_img)

"""