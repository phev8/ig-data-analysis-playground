import os
import cv2
import json

experiment_root = '/Volumes/DataDrive/igroups_recordings/southampton_5'

video_path = os.path.join(experiment_root, 'videos')

videos = [os.path.join(video_path, f) for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f)) and f.split('.')[-1] == 'MP4']

print(videos)

video_infos = {}

for video in videos:
    cap = cv2.VideoCapture(video)

    infos = {
        "orig_path": video,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }

    video_infos[os.path.basename(video)] = infos

print(video_infos)

output_name = os.path.join(experiment_root, 'video_infos.json')
with open(output_name, 'w') as f:
    json.dump(video_infos, f, indent=2)
