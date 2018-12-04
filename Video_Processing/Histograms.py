import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import time
from Video_Processing import Video_to_Images



def calc_group_histogramms():

    prefix_path = Video_to_Images.get_data_dir() + "Persons_Handsorted\\"

    # Mögliche Endungen: "Blue Button", "Green Button", "Red Button" oder Dummy
    groups = ["Blue Button", "Green Button", "Red Button", "Dummy", "Schwarze Weste", "Sonstiges"]

    group_histograms = []

    for group in groups:
        data_path = os.path.join(prefix_path+group+"\\", '*g')

        # Speichere alle Dateinen aus dem Ordner data_path in der Variable files:
        files = sorted(glob.glob(data_path))

        # Anzahl der Vorsortierten Bilder aus der aktuellen Gruppe:
        print("Gruppe", group, "enthält", len(files), "vorsortierte Bilder.")
        data = []

        # Einlesen aller vorsortierten Daten:
        for img in files:
            image = cv2.imread(img)
            image = cv2.resize(image, dsize=(300, 300))
            data.append(image)

        # Berechne für alle vorsortierten Bilder der aktuelle Gruppe das gemittelte und normalisierte Histogramm über alle Kanäle (BGR, statt RGB, da cv2):
        color = ('b', 'g', 'r')
        all_hists_cur_group = []
        hist_all_channels = []
        for i, col in enumerate(color):
            # Die Funktion calcHist berechnet Histogramme aus eine Liste und mittelt diese.
            hist = cv2.calcHist(data, [i], None, [256], [0, 256])
            hist = cv2.normalize(hist, None)
            hist_all_channels.append(hist)
        all_hists_cur_group.append(hist_all_channels)

        for img_id in range(len(files)):
            hist_all_channels = []
            for i, col in enumerate(color):
                # Die Funktion calcHist berechnet Histogramme aus eine Liste und mittelt diese.
                hist = cv2.calcHist([data[img_id]], [i], None, [256], [0, 256])
                hist = cv2.normalize(hist, None)
                hist_all_channels.append(hist)
            all_hists_cur_group.append(hist_all_channels)

        group_histograms.append(all_hists_cur_group)
    return group_histograms




def compare_histograms(estimated_histograms, hist_new_person):
    # estimated_histograms ist eine Liste mit vier Elementen. Jedes Elemente besteht aus drei Arrays:
    #       geschätztes normalisiertes Histogramm des blauen Kanals
    #       geschätztes normalisiertes Histogramm des gruenen Kanals
    #       geschätztes normalisiertes Histogramm des roten Kanals
    # Das erste Element repäsentiert die Frau mit dem blauen Button, das zweite Element die Frau mit dem gruenen Buttonm, das dritte
    # Element dir Frau mit dem roten Button und das vierte Element repräsentiert die geschätzten Histogramme des Dummys.

    # Bisher beste Übereinstimmung, zu beginn extrem große Abweichung:
    best_coincidence = 10**9
    group_id = -1
    for group in range(len(estimated_histograms)):
        for img_id in range(len(estimated_histograms[group])):
            coincidence = 0
            for chanel_id in range(len(estimated_histograms[group][img_id])):
                summe = 0
                for k in range(len(estimated_histograms[group][img_id][chanel_id])):
                    summe += np.abs(estimated_histograms[group][img_id][chanel_id][k][0] - hist_new_person[chanel_id][k][0])
                coincidence += summe
            if best_coincidence > coincidence:
                best_coincidence = coincidence
                group_id = group

    if best_coincidence > 14.00:
        group_id = 5

    print("best_coincidence =", best_coincidence)
    # Zum Schluss steht in group_id der Index, der die geringste Abweichung aufwies
    return group_id



def split_Persons(estimated_histograms):
    starttime = time.time()
    prefix_path = Video_to_Images.get_data_dir() + "Persons_separated\\"
    # Mögliche Endungen: "Blue Button", "Green Button", "Red Button" oder Dummy
    groups = ["Blue Button", "Green Button", "Red Button", "Dummy", "Schwarze Weste", "Sonstiges"]
    group_histograms = []

    # Lege für jede Gruppe einen neuen Ordner an, falls nötig:
    for group in groups:
        separated_group_path = prefix_path + group + "\\"
        if not os.path.isdir(separated_group_path):
            os.makedirs(separated_group_path)
        data_path = os.path.join(separated_group_path, '*g')


    all_persons_path = os.path.join(Video_to_Images.get_data_dir() + "Persons_extracted\\", '*g')

    # Speichere alle Dateinen aus dem Ordner data_path in der Variable files:
    files = sorted(glob.glob(all_persons_path), key=os.path.getmtime)
    print("üüüü", files[0])
    # Anzahl Bilder die zugeordnet werden sollen:
    print("len(files) =", len(files))


    print("Beginn reading all Images")
    progress_counter = 0
    # Einlesen aller vorsortierten Daten:
    for img in files:
        progress_counter += 1
        print(progress_counter)
        image = cv2.imread(img)
        image_resized = cv2.resize(image, (300,300))

        # Berechne für aktuelles Bild das normalisierte Histogramm über alle Kanäle (BGR, statt RGB, da cv2):
        color = ('b', 'g', 'r')
        hist_all_channels = []
        for i, col in enumerate(color):
            hist = cv2.calcHist([image_resized], [i], None, [256], [0, 256])
            hist = cv2.normalize(hist, None)
            hist_all_channels.append(hist)
            # print(len(histr))
            # plt.plot(hist, color=col)
            # plt.xlim([0, 256])
        # plt.show()
        group_id = compare_histograms(estimated_histograms, hist_all_channels)
        filepath = prefix_path + groups[group_id] + "\\" + os.path.basename(img)
        cv2.imwrite(filename=filepath, img=image)

    print("Read process done.")
    runtime = time.time() - starttime
    print("Runtime:", runtime)



estimated_histograms = calc_group_histogramms()
split_Persons(estimated_histograms)


def plot_histogram():
    # Mögliche Endungen: "Blue Button", "Green Button", "Red Button" oder Dummy
    path = Video_to_Images.get_data_dir() + "Persons_Handsorted\\Red Button\\"

    data_path = os.path.join(path, '*g')
    files = sorted(glob.glob(data_path))

    print("len(files) =", len(files))

    verteilung = []
    data = []
    for f1 in files:
        img = cv2.imread(f1)
        print(type(img))
        data.append(img)
        """
        normalizedImg = cv2.normalize(img, None)
        cv2.imshow('Hallo', img)
        cv2.waitKey(0)
        """

        test = np.zeros(shape=(100,100,3))
        test[:,:,0] = 1     # Blau
        # test[:,:,1] = 1     # Gruen
        # test[:,:,2] = 1     # Rot

        num_pixel = img.shape[0] * img.shape[1]
        print("num_pixel =", num_pixel)

        blue = np.sum(img[:, :, 0]) / num_pixel
        green = np.sum(img[:, :, 1]) / num_pixel
        red = np.sum(img[:, :, 2]) / num_pixel
        verteilung.append([blue, green, red])

        print("Blau-Anteil:", blue)
        print("Grün-Anteil:", green)
        print("Rot-Anteil:", red)

        # cv2.imshow('Hallo', img)
        # cv2.waitKey(0)
    print("verteilung =", verteilung)
    erg_blau = [item[0] for item in verteilung]
    erg_green = [item[1] for item in verteilung]
    erg_red = [item[2] for item in verteilung]

    print("verteilung[:] =", erg_blau)
    print(erg_blau)
    erg_blau_normalized = np.sum(erg_blau) / len(files)
    erg_green_normalized = np.sum(erg_green) / len(files)
    erg_red_normalized = np.sum(erg_red) / len(files)

    print(erg_blau_normalized)
    print(erg_green_normalized)
    print(erg_red_normalized)


    # len(files)

    # img = cv2.imread(filepath_red_dot)
    # img = img[300:400, 0:300]

    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # os.mkdir(directory + col)





    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist(data, [i], None, [256], [0, 256])
        hist = cv2.normalize(histr, None)
        # print(len(histr))
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()

# plot_histogram()