import os
import cv2


main_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
git_dir = str(os.path.abspath(os.path.join(main_dir, os.pardir)))
project_dir = str(os.path.abspath(os.path.join(git_dir, os.pardir)))
data_dir = project_dir + "\\Daten\\igroups_student_project\\"
filename = project_dir + "\\Daten\\igroups_student_project\\videos\\cam_2.mp4"
output_dir = project_dir + "\\Daten\\igroups_student_project\\Frames_extracted\\"


def get_project_dir():
    return project_dir


def get_data_dir():
    return data_dir


def get_video_path():
    return filename


def get_output_dir():
    return output_dir


def convert_Video_to_single_Frames():
    if os.path.isdir(output_dir):
        print("Exist:", output_dir)
    else:
        os.makedirs(output_dir)
        print("Created Directory:", output_dir)


    vidcap = cv2.VideoCapture(filename)
    success, image = vidcap.read()
    counter = 0

    while success:
        cur_img_path = output_dir + "frame_" + str(counter) + ".png"
        cv2.imwrite(cur_img_path, image)     # save frame as JPEG file
        success,image = vidcap.read()
        if success and counter % 100 == 0:
            print("Read frame", counter, "successfully")
        counter += 1

    print("\n\n--------------------------\nYour Frames are stored in:\n" + output_dir)
