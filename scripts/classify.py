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
    threshold = None
    kernel = None
    area = None

    def __init__(self, bg, th, kr, ar):
        super().__init__()
        self.background = bg
        self.threshold = th
        self.kernel = kr
        self.area = ar
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
        mask = self.apply_background_subtraction(frame, new_width, new_height)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        updated_colors = {}
        
        for i in range(1, num_labels):  # skipping background label (0)
            if stats[i, cv2.CC_STAT_AREA] > self.area:
                cx, cy = centroids[i]

                # Extracting bounding box information
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
                # Create a mask for the current object
                object_mask = (labels == i).astype(np.uint8) * 255

                # Classify the detected object
                label = self.classify_object(frame, object_mask)

                # Track the object color
                closest_key = None
                min_distance = float("inf")
                threshold_distance = 30  # max distance to be considered the same object

                # Find the closest matching object based on centroid position and label
                for prev_key in self.object_colors.keys():
                    prev_cx, prev_cy, prev_label = prev_key
                    distance = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                    
                    if distance < min_distance and prev_label == label:
                        min_distance = distance
                        closest_key = prev_key

                # Assign or reuse color
                if closest_key and min_distance < threshold_distance:
                    color = self.object_colors[closest_key]  # reuse previous color
                    updated_colors[(cx, cy, label)] = color  # update tracking dictionary
                else:
                    color = tuple(np.random.randint(0, 255, size=3, dtype=np.uint8))  # new color
                    updated_colors[(cx, cy, label)] = color  # store for next frame

                color = tuple(int(c) for c in color)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # Draw bounding box on the original frame

                # Add classification label near the bounding box
                cv2.putText(frame, f"Class: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)

        # Update global dictionary for tracking across frames
        self.object_colors = updated_colors.copy()

        return frame



    
    
    def apply_background_subtraction(self, frame, new_width, new_height):
        
        # converting to gray for better accuracy
        newBG = cv2.resize(self.background, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        gray_background = cv2.cvtColor(newBG, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        fg_mask = cv2.absdiff(gray_background, gray_frame)
        
        _, thresh = cv2.threshold(fg_mask, self.threshold, 255, cv2.THRESH_BINARY)

        # applying morphological operations to remove noise
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        kernel = np.ones((self.kernel, self.kernel), np.uint8) 
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
