import sys
import ctypes
import numpy as np
import cv2
import csv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time

# Append the path for the camera SDK
sys.path.append("imports/")
from MvCameraControl_class import *

###########################
# GLOBAL VARIABLES & SETUP
###########################

# Initialize scaler and classifier (using 3 neighbors as before)
scaler = StandardScaler()
knn = KNeighborsClassifier(n_neighbors=3)
trained = False  # flag to indicate if the model has been trained

# These lists will store the feature vectors and class labels from training images
training_data = []
training_labels = []
# A dictionary mapping class numbers to names (for display)
labels_dict = {}

##########################################
# Camera Initialization & Image Acquisition
##########################################

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

    # Open the device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"Failed to open device! Error code: 0x{ret:X}")
        cam.MV_CC_DestroyHandle()
        sys.exit()

    # Set camera parameters
    cam.MV_CC_SetFloatValue("ExposureTime", 60000.0)  # Set exposure time (adjust as needed)
    cam.MV_CC_SetEnumValue("GainAuto", 0)  # Disable auto gain

    # Start grabbing frames
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"Failed to start grabbing! Error code: 0x{ret:X}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        sys.exit()

    print("Camera is grabbing frames...")
    return cam

# Create a global camera object (used in getOpenCVImage below)
cam = init_camera()

def getOpenCVImage():
    """
    Grabs an image from the camera using the SDK and converts it to an OpenCV BGR image.
    """
    # Prepare the frame structure
    stOutFrame = MV_FRAME_OUT()
    ctypes.memset(ctypes.byref(stOutFrame), 0, ctypes.sizeof(stOutFrame))
    
    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
    if ret != 0:
        print(f"Failed to get image buffer! Error code: 0x{ret:X}")
        return None

    # Create a buffer cache from the image data
    buf_cache = (ctypes.c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
    ctypes.memmove(ctypes.byref(buf_cache), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)

    width, height = stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight
    # Scale the image to a maximum size for the GUI display (here 640x480)
    scale_factor = min(640 / width, 480 / height)
    
    np_image = np.ctypeslib.as_array(buf_cache).reshape(height, width)
    # Convert Bayer image to BGR format
    cv_image = cv2.cvtColor(np_image, cv2.COLOR_BayerBG2BGR)
    cv_image = cv2.resize(cv_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Free the buffer once done
    cam.MV_CC_FreeImageBuffer(stOutFrame)
    return cv_image

#############################################
# Feature Extraction (HSV Histogram)
#############################################

def extractHistogram(imagePath):
    """
    Reads an image from disk, converts to HSV and computes concatenated histograms for H, S, and V.
    """
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
# Training Mode Functions
#############################################

def select_images_for_class(class_index):
    """
    Opens a file dialog for the user to select one or more images for a given class.
    """
    file_paths = filedialog.askopenfilenames(title=f"Select images for Class {class_index}")
    return list(file_paths)

def add_class_data(class_index):
    """
    Processes the selected images for a class by extracting features and updating training data.
    """
    global training_data, training_labels, labels_dict
    image_paths = select_images_for_class(class_index)
    if not image_paths:
        messagebox.showwarning("No Selection", f"No images selected for Class {class_index}")
        return
    for path in image_paths:
        features = extractHistogram(path)
        if features is not None:
            training_data.append(features)
            training_labels.append(class_index)
    # For simplicity, we assign a default class name; you can modify this to ask for custom names.
    labels_dict[class_index] = f"Class {class_index}"
    messagebox.showinfo("Success", f"Added {len(image_paths)} images for Class {class_index}")

def train_model():
    """
    Trains the KNN classifier using the collected training data.
    """
    global trained, scaler, knn
    if not training_data or not training_labels:
        messagebox.showerror("Error", "No training data available. Please add training images first.")
        return
    X = np.array(training_data)
    y = np.array(training_labels)
    X_scaled = scaler.fit_transform(X)
    knn.fit(X_scaled, y)
    trained = True
    messagebox.showinfo("Training Complete", "The model has been trained successfully.")

#############################################
# Classification Mode Functions & Camera Stream
#############################################

class CameraStream:
    """
    Handles capturing images from the camera in a separate thread and updating a Tkinter canvas.
    """
    def __init__(self, canvas):
        self.canvas = canvas
        self.running = False
        self.thread = None
        self.current_frame = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            frame = getOpenCVImage()
            if frame is not None:
                self.current_frame = frame.copy()
                # Convert BGR to RGB for Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.imgtk = imgtk  # Keep a reference
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            time.sleep(0.03)  # Small delay to keep GUI responsive

    def stop(self):
        self.running = False

def classify_region(event, cam_stream):
    """
    On mouse click in the classification canvas, extracts a region around the click,
    computes its HSV histogram, and predicts the class.
    """
    global knn, scaler, trained, labels_dict
    if not trained:
        messagebox.showwarning("Not Trained", "Please train the model before classification.")
        return

    # Get click coordinates from the canvas
    x, y = event.x, event.y
    half_window = 50  # You can adjust the window size as needed
    frame = cam_stream.current_frame
    if frame is None:
        return
    h, w, _ = frame.shape
    x_start = max(x - half_window, 0)
    x_end = min(x + half_window, w)
    y_start = max(y - half_window, 0)
    y_end = min(y + half_window, h)
    
    region = frame[y_start:y_end, x_start:x_end]
    if region.size == 0:
        return
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv_region], [0], None, [180], [0, 256]).flatten()
    s_hist = cv2.calcHist([hsv_region], [1], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv_region], [2], None, [256], [0, 256]).flatten()
    features = np.concatenate((h_hist, s_hist, v_hist)).reshape(1, -1)
    features_scaled = scaler.transform(features)
    predicted_class = knn.predict(features_scaled)[0]
    label_text = labels_dict.get(predicted_class, f"Class {predicted_class}")
    
    # Draw a rectangle around the region and overlay the predicted class
    cam_stream.canvas.create_rectangle(x_start, y_start, x_end, y_end, outline="magenta", width=2)
    cam_stream.canvas.create_text(x, y - 20, text=label_text, fill="yellow", font=("Arial", 16, "bold"))

#############################################
# Main GUI Setup (Tkinter)
#############################################

root = tk.Tk()
root.title("General Object Classification GUI")

# Create a Notebook with two tabs: Training Mode and Classification Mode
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# ----- Training Mode Tab -----
train_tab = ttk.Frame(notebook)
notebook.add(train_tab, text="Training Mode")

# Option for user to choose number of classes
num_classes_label = ttk.Label(train_tab, text="Number of Classes:")
num_classes_label.pack(pady=5)
num_classes_var = tk.IntVar(value=3)
num_classes_spin = ttk.Spinbox(train_tab, from_=1, to=10, textvariable=num_classes_var)
num_classes_spin.pack(pady=5)

# Frame to hold buttons for each class
classes_frame = ttk.Frame(train_tab)
classes_frame.pack(pady=10)

def setup_class_buttons():
    # Clear previous class buttons if any
    for widget in classes_frame.winfo_children():
        widget.destroy()
    num = num_classes_var.get()
    for i in range(1, num+1):
        btn = ttk.Button(classes_frame, text=f"Add Images for Class {i}", command=lambda idx=i: add_class_data(idx))
        btn.pack(pady=2)

setup_btn = ttk.Button(train_tab, text="Setup Classes", command=setup_class_buttons)
setup_btn.pack(pady=5)

# Button to train the model with the collected data
train_model_btn = ttk.Button(train_tab, text="Train Model", command=train_model)
train_model_btn.pack(pady=10)

# ----- Classification Mode Tab -----
class_tab = ttk.Frame(notebook)
notebook.add(class_tab, text="Classification Mode")

# Canvas to display the live camera feed
cam_canvas = tk.Canvas(class_tab, width=640, height=480)
cam_canvas.pack()

# Start the camera stream in a separate thread
cam_stream = CameraStream(cam_canvas)
cam_stream.start()

# Bind left mouse click on the canvas to trigger classification on that region
cam_canvas.bind("<Button-1>", lambda event: classify_region(event, cam_stream))

#############################################
# Cleanup on Exit
#############################################

def on_close():
    # Stop the camera stream and release camera resources
    cam_stream.stop()
    ret = cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    root.destroy()
    print("Camera resources released.")

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
