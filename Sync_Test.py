




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

print("sync_given_cam1_org_video:", sync_given_cam1_org_video(21000))
print("sync_given_cam2_org_video:", sync_given_cam2_org_video(7375))
print("sync_given_cam3_org_video:", sync_given_cam3_org_video(34125))











offset_cam1_between_skeleton_and_org = 7109      # 19531 - 12423
offset_cam2_between_skeleton_and_org = 7156      # 19692 - 12537
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

print("cam 1: ", sync_given_cam1_org_to_skeleton(21000))
print("cam 2: ", sync_given_cam2_org_to_skeleton(21161))
print("cam 3: ", sync_given_cam3_org_to_skeleton(20748))