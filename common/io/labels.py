import pandas as pd


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