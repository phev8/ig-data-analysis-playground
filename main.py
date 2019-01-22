# from Skeleton import Global_Constants
from Skeleton import Hist_Classifier as classifier
from Skeleton import Utils as utils
from Skeleton import Reduce_Feature_Matrix as rfm
import os
import sys

if __name__ == '__main__':
    # Wenn die Feature Matrix schon existiert kann sie auch direkt gelesen werden. Dann setzt die Varibale create_Feature_Matrix auf False:
    create_Feature_Matrix = False
    utils.set_paths()
    data_dir = utils.get_data_dir()
    print("data_dir =", data_dir)
    file = data_dir + 'Feature_Matrices.pkl'
    sys.path.append(file)
    if os.path.isdir(data_dir):
        print("Ja, file =", file)
    else:
        print("Nein, file =", file)
    utils.main(create_Feature_Matrix)

    # Angenommen die Daten wurden bereits klassifiziert (das macht man mit Histogramms.py), dann hat man einen Ordner mit dem Namen "Skeletons_separated" Somit können wir die dort gespeicherten Ergebnisse
    # direkt verwenden um eine neue Feature Matrix zu erstellen, die nun Informationen über die MidHip Daten enthält. Die so erstellte Feature Matrix wird als Pickle Datei "lassified_Persons.pkl" gespeichert.
    classifier.classify_offline()

    # Da jetzt in der Feature Matrix noch alle möglichen Labels drin stehen, müssen die passenden gespeichert und falsche entfernt werden. Dies geschieht im nächsten Schritt:
    rfm.main()


"""
Aufrufreihenfolge:
1. Klassifizieren mit Handsortierten Bildern. Dazu verwende Histograms.py
2. Erstelle Huge Feature Matrix. Dazu Utils.py mit main(True) ausführen
3. Finde Klassifizierte Bilder als Pfad wieder. Verwende hierfür Hist_Classifier.py mit dem Aufruf classify_offline()
4. Entferne die nicht benötigten Labels. Hierfür verwende Reduce_Feature_Matrix.py.
5. Anwenden der Ergebnisse:
    5.1 Klassifizierer verwenden mit Aufruf von Classifier.py
    5.2 3D Rekonstruktion mit Aufruf von Reconstruct.py
"""
