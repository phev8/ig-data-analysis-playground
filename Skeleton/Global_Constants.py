import os


main_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
git_dir = str(os.path.abspath(os.path.join(main_dir, os.pardir)))
project_dir = str(os.path.abspath(os.path.join(git_dir, os.pardir)))  + "\\"
project_dir_ext = "H:\\Uni\\Statistische KI Projekt\\"
data_dir = project_dir + "Daten\\igroups_student_project\\"
data_dir_ext = project_dir_ext + "Daten\\igroups_student_project\\"
filename = project_dir + "Daten\\igroups_student_project\\videos\\cam_2.mp4"
cam_2_labels_pickle = project_dir + "Daten\\igroups_student_project\\processed_data\\images\\general_class_rois\\cam_2_detections_general.pkl"
output_dir = project_dir + "Daten\\igroups_student_project\\Frames_extracted\\"
skeletons_dir = data_dir + "processed_data\\images\\skeletons\\"