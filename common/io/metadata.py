import os
import json


def load_video_infos(exp_root, video_name=None):
    path = os.path.join(exp_root, 'video_infos.json')

    data = json.load(open(path, 'r'))

    if video_name is not None:
        data = data[video_name]

    return data
