import os
import cv2
import sys
import csv
import numpy as np
from PyQt5.QtWidgets import QApplication


sys.path.append("scripts/")
from UI import MainWindow

CSV_PATH = "model/data.csv"
DATASET_PATH = "dataset/"

def CreateWindow():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.trainClicked.connect(train_function) 
    window.show()
    sys.exit(app.exec_())
    
def extractHistogram(imagePath):
    
    # first convert image to HSV
    image = cv2.imread(imagePath)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # now i find the histogram for each h, s, v channel
    h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 256]).flatten()
    s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256]).flatten()
    
    # flattening the histograms
    features = np.concatenate((h_hist, s_hist, v_hist))
    
    return features

def train_function():
    class_folders = os.listdir(DATASET_PATH)  # get all class folders
    
    with open(CSV_PATH, mode = "w", newline = "") as file:
        writer = csv.writer(file)

        for class_name in class_folders:
            class_path = os.path.join(DATASET_PATH, class_name)
            
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                print(f"Processing image: {image_path}")
                features = extractHistogram(image_path)
                writer.writerow(list(features) + [class_name])


    
    print("Training finished!")
    
    
if __name__ == '__main__':
    CreateWindow()