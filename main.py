import sys
import ctypes
import time
import cv2
import numpy as np
import csv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PIL import Image

# Append camera SDK helper path
sys.path.append("imports/")
from MvCameraControl_class import *

#############################
# Global Variables & Objects
#############################

scaler = StandardScaler()
knn = KNeighborsClassifier(n_neighbors=3)
trained = False  # becomes True once the model is trained

# Lists to store training features and labels
training_data = []
training_labels = []
# A dictionary mapping class numbers to names (for display)
labels_dict = {}

# This variable will store details about the last prediction overlay
# (so it can be drawn on the live feed for a short time)
lastPrediction = None

#############################################
# Camera Initialization and Frame Acquisition
#############################################

def init_camera():
    # Initialize the camera SDK
    MvCamera().MV_CC_Initialize()

    # Enumerate devices
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0:
        print(f"Device enumeration failed! Error code: 0x{ret:X}")
        sys.exit()
    if deviceList.nDeviceNum == 0:
        print("No camera devices found.")
        sys.exit()
    print(f"Found {deviceList.nDeviceNum} device(s).")

    # Use the first device in the list
    stDeviceList = ctypes.cast(deviceList.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents

    # Create camera handle
    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print(f"Failed to create handle! Error code: 0x{ret:X}")
        sys.exit()

    # Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"Failed to open device! Error code: 0x{ret:X}")
        cam.MV_CC_DestroyHandle()
        sys.exit()

    # Set parameters (adjust these values as needed)
    cam.MV_CC_SetFloatValue("ExposureTime", 60000.0)
    cam.MV_CC_SetEnumValue("GainAuto", 0)

    # Start grabbing frames
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"Failed to start grabbing! Error code: 0x{ret:X}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        sys.exit()

    print("Camera is grabbing frames...")
    return cam

# Create a global camera object for use in the GUI
cam = init_camera()

def getOpenCVImage():
    """
    Grabs an image frame from the camera using the SDK and converts it to an OpenCV BGR image.
    """
    stOutFrame = MV_FRAME_OUT()
    ctypes.memset(ctypes.byref(stOutFrame), 0, ctypes.sizeof(stOutFrame))
    
    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
    if ret != 0:
        print(f"Failed to get image buffer! Error code: 0x{ret:X}")
        return None

    # Copy the image buffer into a numpy array
    buf_cache = (ctypes.c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
    ctypes.memmove(ctypes.byref(buf_cache), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
    
    width = stOutFrame.stFrameInfo.nWidth
    height = stOutFrame.stFrameInfo.nHeight
    # Scale the image for display in the UI (here max size is 640x480)
    scale_factor = min(640 / width, 480 / height)
    
    np_image = np.ctypeslib.as_array(buf_cache).reshape(height, width)
    # Convert Bayer image to BGR format
    cv_image = cv2.cvtColor(np_image, cv2.COLOR_BayerBG2BGR)
    cv_image = cv2.resize(cv_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    cam.MV_CC_FreeImageBuffer(stOutFrame)
    return cv_image

#############################################
# Feature Extraction Function (HSV Histogram)
#############################################

def extractHistogram(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        print(f"Failed to load image: {imagePath}")
        return None
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 256]).flatten()
    s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256]).flatten()
    features = np.concatenate((h_hist, s_hist, v_hist))
    return features

#############################################
# PyQt5 UI Classes and Functions
#############################################

# A custom QLabel to capture mouse clicks on the video feed
class VideoLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(QtCore.QPoint)
    
    def __init__(self, parent=None):
        super(VideoLabel, self).__init__(parent)
        # Disable background clearing to reduce flickering
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
    
    def mousePressEvent(self, event):
        self.clicked.emit(event.pos())

# Main application window with two tabs: Training and Classification
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("General Object Classification")
        self.resize(800, 600)
        self.current_frame = None  # Holds the latest frame from the camera
        self.lastPrediction = None # To store prediction details for overlay
        
        self.initUI()
        
        # QTimer to update the video feed
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)  # Update roughly every 30 ms

    def initUI(self):
        # Create a tab widget
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # ----- Training Mode Tab -----
        self.train_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.train_tab, "Training Mode")
        self.trainLayout = QtWidgets.QVBoxLayout(self.train_tab)
        
        # Number of classes input
        num_layout = QtWidgets.QHBoxLayout()
        num_label = QtWidgets.QLabel("Number of Classes:")
        self.numClassesSpin = QtWidgets.QSpinBox()
        self.numClassesSpin.setMinimum(1)
        self.numClassesSpin.setMaximum(10)
        self.numClassesSpin.setValue(3)
        num_layout.addWidget(num_label)
        num_layout.addWidget(self.numClassesSpin)
        self.trainLayout.addLayout(num_layout)
        
        # Button to create class-specific image selection buttons
        self.setupBtn = QtWidgets.QPushButton("Setup Classes")
        self.setupBtn.clicked.connect(self.setupClassButtons)
        self.trainLayout.addWidget(self.setupBtn)
        
        # Widget to hold class buttons
        self.classButtonsWidget = QtWidgets.QWidget()
        self.classButtonsLayout = QtWidgets.QVBoxLayout(self.classButtonsWidget)
        self.trainLayout.addWidget(self.classButtonsWidget)
        
        # Button to train the model
        self.trainModelBtn = QtWidgets.QPushButton("Train Model")
        self.trainModelBtn.clicked.connect(self.trainModel)
        self.trainLayout.addWidget(self.trainModelBtn)
        
        # ----- Classification Mode Tab -----
        self.class_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.class_tab, "Classification Mode")
        self.classLayout = QtWidgets.QVBoxLayout(self.class_tab)
        
        # Video display label
        self.videoLabel = VideoLabel()
        self.videoLabel.setFixedSize(640, 480)
        self.videoLabel.setStyleSheet("background-color: black;")
        self.videoLabel.clicked.connect(self.onVideoClicked)
        self.classLayout.addWidget(self.videoLabel)
    
    def setupClassButtons(self):
        # Clear any existing buttons
        for i in reversed(range(self.classButtonsLayout.count())):
            widget = self.classButtonsLayout.takeAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        num = self.numClassesSpin.value()
        for i in range(1, num + 1):
            btn = QtWidgets.QPushButton(f"Add Images for Class {i}")
            # Use lambda with default argument to capture current index
            btn.clicked.connect(lambda checked, idx=i: self.addClassData(idx))
            self.classButtonsLayout.addWidget(btn)
    
    def addClassData(self, class_index):
        global training_data, training_labels, labels_dict
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self,
                                                f"Select Images for Class {class_index}",
                                                "",
                                                "Images (*.png *.jpg *.jpeg *.bmp)",
                                                options=options)
        if not files:
            QMessageBox.warning(self, "No Selection", f"No images selected for Class {class_index}")
            return
        count = 0
        for filePath in files:
            features = extractHistogram(filePath)
            if features is not None:
                training_data.append(features)
                training_labels.append(class_index)
                count += 1
        labels_dict[class_index] = f"Class {class_index}"
        QMessageBox.information(self, "Success", f"Added {count} images for Class {class_index}")
    
    def trainModel(self):
        global trained, scaler, knn
        if not training_data or not training_labels:
            QMessageBox.critical(self, "Error", "No training data available. Please add images first.")
            return
        X = np.array(training_data)
        y = np.array(training_labels)
        X_scaled = scaler.fit_transform(X)
        knn.fit(X_scaled, y)
        trained = True
        QMessageBox.information(self, "Training Complete", "The model has been trained successfully.")
    
    def updateFrame(self):
        # Get a new frame from the camera
        frame = getOpenCVImage()
        if frame is None:
            return
        self.current_frame = frame.copy()
        
        # If there is a recent prediction, overlay the rectangle and label
        if self.lastPrediction is not None:
            if time.time() - self.lastPrediction['time'] < 2.0:  # Show overlay for 2 seconds
                x_start = self.lastPrediction['x_start']
                y_start = self.lastPrediction['y_start']
                x_end = self.lastPrediction['x_end']
                y_end = self.lastPrediction['y_end']
                label_text = self.lastPrediction['label']
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 255), 2)
                cv2.putText(frame, label_text, (x_start, y_start - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                self.lastPrediction = None

        # Convert frame to QImage for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.videoLabel.setPixmap(pixmap)
        
        cv2.waitKey(1)  # Call waitKey to allow OpenCV to process events (helps with flicker)
    
    def onVideoClicked(self, pos):
        global knn, scaler, trained, labels_dict
        if not trained or self.current_frame is None:
            QMessageBox.warning(self, "Not Trained", "Please train the model before classification.")
            return
        
        half_window = 50  # Define the size of the region around the click
        x = pos.x()
        y = pos.y()
        h, w, _ = self.current_frame.shape
        x_start = max(x - half_window, 0)
        x_end = min(x + half_window, w)
        y_start = max(y - half_window, 0)
        y_end = min(y + half_window, h)
        
        region = self.current_frame[y_start:y_end, x_start:x_end]
        if region.size == 0:
            return
        
        # Compute HSV histogram features for the selected region
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv_region], [0], None, [180], [0, 256]).flatten()
        s_hist = cv2.calcHist([hsv_region], [1], None, [256], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv_region], [2], None, [256], [0, 256]).flatten()
        features = np.concatenate((h_hist, s_hist, v_hist)).reshape(1, -1)
        features_scaled = scaler.transform(features)
        predicted_class = knn.predict(features_scaled)[0]
        label_text = labels_dict.get(predicted_class, f"Class {predicted_class}")
        
        # Save prediction overlay details with a timestamp so it can be shown briefly
        self.lastPrediction = {
            'x_start': x_start,
            'y_start': y_start,
            'x_end': x_end,
            'y_end': y_end,
            'label': label_text,
            'time': time.time()
        }
    
    def closeEvent(self, event):
        # Stop the timer and release camera resources on exit
        self.timer.stop()
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        event.accept()

#############################################
# Main Execution
#############################################

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
