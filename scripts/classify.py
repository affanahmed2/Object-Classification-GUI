from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import cv2
import math


class classifyKNN:

    csv = "model/data.csv"
    background = None
    scaler = StandardScaler()
    knn = KNeighborsClassifier(n_neighbors = 3)
    object_colors = {}

    def __init__(self, bg):
        super().__init__()
        self.background = bg
        self.train()
    

    def train(self):
        data = pd.read_csv(self.csv, header=None)
        X = data.iloc[:, :-1].values  # all columns except last one
        y = data.iloc[:, -1].values   # last column is labels
        
        # normalizing values
        X = self.scaler.fit_transform(X)
        
        # training model
        self.knn.fit(X, y)
    
    def predict(self, frame, new_width, new_height):
        print("predicting")
        mask = self.apply_background_subtraction(frame, new_width, new_height)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        min_area = 5000
    
        updated_colors = {}
        
        for i in range(1, num_labels):  # skipping background label (0)
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                cx, cy = centroids[i]

                # create a mask for the current object
                object_mask = (labels == i).astype(np.uint8) * 255

                # classify the detected pixels only
                label = self.classify_object(frame, object_mask)

                # try to match this object with a previously seen object
                closest_key = None
                min_distance = float("inf")
                threshold_distance = 30  # max distance to be considered the same object

                for prev_key in self.object_colors.keys():
                    prev_cx, prev_cy, prev_label = prev_key
                    distance = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                    
                    if distance < min_distance and prev_label == label:
                        min_distance = distance
                        closest_key = prev_key

                # assign or reuse color
                if closest_key and min_distance < threshold_distance:
                    color = self.object_colors[closest_key]  # reuse previous color
                    updated_colors[(cx, cy, label)] = color  # update tracking dictionary
                else:
                    color = tuple(np.random.randint(0, 255, size=3, dtype=np.uint8))  # new color
                    updated_colors[(cx, cy, label)] = color  # store for next frame

                # assign the color to the component
                colored_mask[labels == i] = color

                # draw classification label near the detected region
                cv2.putText(colored_mask, f"Rock type: {label}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (255, 255, 255), 2)

        # Update global dictionary for tracking across frames
        self.object_colors = updated_colors.copy()
        
        
        return colored_mask
    
    
    def apply_background_subtraction(self, frame, new_width, new_height):
        
        threshold_value = 10
        kernel_size = 5

        # converting to gray for better accuracy
        newBG = cv2.resize(self.background, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        gray_background = cv2.cvtColor(newBG, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        fg_mask = cv2.absdiff(gray_background, gray_frame)
        
        _, thresh = cv2.threshold(fg_mask, threshold_value, 255, cv2.THRESH_BINARY)

        # applying morphological operations to remove noise
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        kernel = np.ones((kernel_size, kernel_size), np.uint8) 
        clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

        return clean_mask
    
    
    def classify_object(self, image, mask):
        """
        Classifies a detected object using the trained KNN model.
        """

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # compute color histograms for H, S, and V channels
        h_hist = cv2.calcHist([hsv_image], [0], mask, [180], [0, 256]).flatten()
        s_hist = cv2.calcHist([hsv_image], [1], mask, [256], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv_image], [2], mask, [256], [0, 256]).flatten()

        # flatten histograms and normalize
        features = np.concatenate((h_hist, s_hist, v_hist)).reshape(1, -1)
        features = self.scaler.transform(features)

        # predict using KNN
        label = self.knn.predict(features)[0]

        return label
