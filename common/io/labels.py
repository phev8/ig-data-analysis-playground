import pandas as pd
import numpy as np
from datetime import timedelta

def read_label_xls(path):
    """
    Read label xlsx and return pandas dataframe for each person's labels
    """
    label_data = pd.read_excel(path, skiprows=2)

    green_labels = []
    red_labels = []
    blue_labels = []

    for index, row in label_data.iterrows():
        # print(index, row)
        gl = {
            "timestamp": row["Timestamp"],
            "position": row["postion"],
            "view point": row["view point"],
            "WC": row["WC"],
            "SA": row["SA"],
            "AA": row["AA"],
            "body posture": row["body posture"],
            "devices used": [] if pd.isnull(row["devices used"]) else str(row["devices used"]).split(','),
        }
        rl = {
            "timestamp": row["Timestamp"],
            "position": row["postion.1"],
            "view point": row["view point.1"],
            "WC": row["WC.1"],
            "SA": row["SA.1"],
            "AA": row["AA.1"],
            "body posture": row["body posture.1"],
            "devices used": [] if pd.isnull(row["devices used.1"]) else str(row["devices used.1"]).split(','),
        }

        bl = {
            "timestamp": row["Timestamp"],
            "position": row["postion.2"],
            "view point": row["view point.2"],
            "WC": row["WC.2"],
            "SA": row["SA.2"],
            "AA": row["AA.2"],
            "body posture": row["body posture.2"],
            "devices used": [] if pd.isnull(row["devices used.2"]) else str(row["devices used.2"]).split(','),
        }
        green_labels.append(gl)
        red_labels.append(rl)
        blue_labels.append(bl)

    green_labels = pd.DataFrame(green_labels).set_index('timestamp')
    red_labels = pd.DataFrame(red_labels).set_index('timestamp')
    blue_labels = pd.DataFrame(blue_labels).set_index('timestamp')

    # convert timestamp
    return green_labels, red_labels, blue_labels



# This method reads all timestamps from the excel file:     [Author: CM]
def read_timestamp(path):
    file = pd.read_excel(path, skiprows=2)
    value = np.array(file)[:,[0,2,10,18]]


    num_timestamps = len(value)


    print("Test\t\t", value[0,0])
    print("Test\t\t", value[0,0].minute)
    print("Test\t\t", value[0,0].second)
    all_timestamps = []
    for i in range(num_timestamps):
        minutes = value[i,0].minute
        seconds = value[i,0].second

        erg = minutes * 60 * 25 + seconds * 25
        all_timestamps.append([erg, value[i,[1,2,3]]])
        # print(erg)

    return all_timestamps



