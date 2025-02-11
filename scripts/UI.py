import os
import shutil
import sys
import ctypes
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QScrollArea, QComboBox,
    QSizePolicy, QFrame
)
from PyQt5.QtGui import QDesktopServices, QImage, QPixmap
from PyQt5.QtCore import QUrl, pyqtSignal, Qt, QThread

from ctypes import cast, POINTER, byref

sys.path.append("imports/")
from MvCameraControl_class import *

# --- New Video Thread for continuous camera capture ---
class VideoThread(QThread):
    changePixmap = pyqtSignal(QImage)
    
    def __init__(self, cam):
        super(VideoThread, self).__init__()
        self.cam = cam
        self._running = True
        
    def run(self):
        while self._running:
            
            # Initialize the frame structure and clear it
            stOutFrame = MV_FRAME_OUT()
            ctypes.memset(ctypes.byref(stOutFrame), 0, ctypes.sizeof(stOutFrame))
            
            # Get the image buffer with a timeout of 1000ms
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if ret != 0:
                continue  # Skip this iteration if failed
            
            # Allocate a buffer and copy image data from the device pointer
            buf_cache = (ctypes.c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
            ctypes.memmove(ctypes.byref(buf_cache), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
            
            width, height = stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight
            scale_factor = min(1920 / width, 1080 / height)
            
            np_image = np.ctypeslib.as_array(buf_cache).reshape(height, width)
            cv_image = cv2.cvtColor(np_image, cv2.COLOR_BayerBG2BGR)
            cv_image = cv2.resize(cv_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            
            # Free the image buffer after copying the data
            self.cam.MV_CC_FreeImageBuffer(stOutFrame)
            
            # Convert BGR to RGB for proper QImage formatting
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            self.changePixmap.emit(qt_image)
            
        print("Video thread stopped.")
        
    def stop(self):
        self._running = False



# --- Main Window (UI) ---
class MainWindow(QWidget):
    # Existing signal for training plus new signals for device control
    trainClicked = pyqtSignal()
    cam = None
    deviceList = []

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Object Classification")
        self.resize(800, 600)
        
        # Create the tab widget
        self.tabs = QTabWidget()
        
        # Setup Training tab
        self.training_tab = QWidget()
        self.setup_training_tab()
        
        # Setup Classification tab with updated layout
        self.classification_tab = QWidget()
        self.setup_classification_tab()
        
        # Add tabs to the tab widget
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.classification_tab, "Classification")
        
        # Set the main layout of the window
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def setup_training_tab(self):
        # [Existing training tab code remains unchanged]
        layout = QVBoxLayout()
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.classes_container = QWidget()
        self.classes_layout = QVBoxLayout()
        self.classes_container.setLayout(self.classes_layout)
        self.scroll_area.setWidget(self.classes_container)
        
        self.classes_group = QGroupBox("Classes List")
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.scroll_area)
        self.classes_group.setLayout(group_layout)
        layout.addWidget(self.classes_group)
        
        self.manage_group = QGroupBox("Manage Classes")
        self.manage_group.setFixedHeight(100)
        manage_layout = QHBoxLayout()
        self.class_name_input = QLineEdit()
        self.class_name_input.setPlaceholderText("Enter new class name")
        self.add_class_button = QPushButton("Add New Class")
        self.add_class_button.clicked.connect(self.create_class)
        self.delete_all_button = QPushButton("Delete All Classes")
        self.delete_all_button.clicked.connect(self.delete_all_classes)
        manage_layout.addWidget(self.class_name_input)
        manage_layout.addWidget(self.add_class_button)
        manage_layout.addWidget(self.delete_all_button)
        self.manage_group.setLayout(manage_layout)
        layout.addWidget(self.manage_group)
        
        train_button_layout = QHBoxLayout()
        train_button_layout.setAlignment(Qt.AlignCenter)
        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.handle_train_button)
        train_button_layout.addWidget(self.train_button)
        layout.addLayout(train_button_layout)
        
        self.training_tab.setLayout(layout)
        
        self.dataset_dir = os.path.join(os.getcwd(), "dataset")
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        
        self.class_widgets = {}
        self.load_existing_classes()

    def setup_classification_tab(self):
        
        # Create a horizontal layout to split the classification tab into two sections
        classification_layout = QHBoxLayout()
        
        # LEFT SIDE: Contains the device toolbar (dropdown, "Find Devices" button, and video feed)
        left_side_layout = QVBoxLayout()
        device_layout = QHBoxLayout()
        
        # Create the dropdown menu and set its size policy to expand horizontally.
        self.device_dropdown = QComboBox()  # Initially empty
        self.device_dropdown.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Create a smaller "Find Devices" button by setting a fixed width.
        self.find_devices_button = QPushButton("Find Devices")
        self.find_devices_button.setFixedWidth(100)
        self.find_devices_button.clicked.connect(self.find_devices)
        
        # Add widgets to the device layout in the desired order.
        device_layout.addWidget(self.device_dropdown)
        device_layout.addWidget(self.find_devices_button)
        left_side_layout.addLayout(device_layout)
        
        # Add a QLabel to display the video feed (initially showing placeholder text)
        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setFixedSize(640, 480)  # adjust size as needed
        left_side_layout.addWidget(self.video_label)
        left_side_layout.addStretch()
        self.video_label.setScaledContents(True)
        
        # MIDDLE: Add a vertical separator between left and right sections.
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        
        # RIGHT SIDE: Contains the new device control buttons.
        right_side_layout = QVBoxLayout()
        self.open_device_button = QPushButton("Open Device")
        self.open_device_button.clicked.connect(self.on_open_device)
        self.close_device_button = QPushButton("Close Device")
        self.close_device_button.clicked.connect(self.on_close_device)
        self.close_device_button.setEnabled(False)  # Initially disabled
        
        right_side_layout.addWidget(self.open_device_button)
        right_side_layout.addWidget(self.close_device_button)
        right_side_layout.addStretch()
        
        # Add the left layout, separator, and right layout to the main classification layout.
        classification_layout.addLayout(left_side_layout, 3)
        classification_layout.addWidget(separator)
        classification_layout.addLayout(right_side_layout, 1)
        
        self.classification_tab.setLayout(classification_layout)

    def load_existing_classes(self):
        for entry in os.listdir(self.dataset_dir):
            class_folder = os.path.join(self.dataset_dir, entry)
            if os.path.isdir(class_folder):
                widget = self.create_class_widget(entry, class_folder)
                self.classes_layout.addWidget(widget)
                self.class_widgets[entry] = widget

    def create_class(self):
        class_name = self.class_name_input.text().strip()
        if not class_name:
            QMessageBox.warning(self, "Input Error", "Please enter a class name.")
            return
        class_folder = os.path.join(self.dataset_dir, class_name)
        if os.path.exists(class_folder):
            QMessageBox.warning(self, "Error", f"Class '{class_name}' already exists.")
            return
        os.makedirs(class_folder)
        widget = self.create_class_widget(class_name, class_folder)
        self.classes_layout.addWidget(widget)
        self.class_widgets[class_name] = widget
        self.class_name_input.clear()

    def create_class_widget(self, class_name, class_folder):
        widget = QWidget()
        layout = QHBoxLayout()
        
        name_label = QLabel(class_name)
        open_folder_button = QPushButton("Open Folder")
        open_folder_button.clicked.connect(lambda: self.open_folder(class_folder))
        
        add_images_button = QPushButton("Add Images")
        add_images_button.clicked.connect(lambda: self.add_images(class_folder))
        
        delete_class_button = QPushButton("Delete Class")
        delete_class_button.clicked.connect(lambda: self.delete_class(class_name, class_folder, widget))
        
        layout.addWidget(name_label)
        layout.addWidget(open_folder_button)
        layout.addWidget(add_images_button)
        layout.addWidget(delete_class_button)
        widget.setLayout(layout)
        return widget

    def open_folder(self, folder_path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
    
    def add_images(self, class_folder):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Training Images", "",
            "Image Files (.png *.jpg *.jpeg *.bmp);;All Files ()", options=options
        )
        if files:
            for file in files:
                try:
                    shutil.copy(file, class_folder)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to copy {file}: {str(e)}")
            QMessageBox.information(self, "Images Added",
                                    f"Added {len(files)} images to {os.path.basename(class_folder)}.")

    def delete_class(self, class_name, class_folder, widget):
        reply = QMessageBox.question(
            self, 'Delete Class',
            f"Are you sure you want to delete class '{class_name}' and all its images?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                shutil.rmtree(class_folder)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error deleting folder: {str(e)}")
                return
            widget.setParent(None)
            del self.class_widgets[class_name]
    
    def delete_all_classes(self):
        reply = QMessageBox.question(
            self, 'Delete All Classes',
            "Are you sure you want to delete ALL classes and their images?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            for class_name, widget in list(self.class_widgets.items()):
                class_folder = os.path.join(self.dataset_dir, class_name)
                try:
                    shutil.rmtree(class_folder)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Error deleting {class_name}: {str(e)}")
                    continue
                widget.setParent(None)
                del self.class_widgets[class_name]
    
    def handle_train_button(self):
        self.trainClicked.emit()

    # --- New Device Control Methods ---
    def on_open_device(self):
        """Called when the Open Device button is pressed."""
        selected_index = self.device_dropdown.currentIndex()
        if selected_index == -1:
            QMessageBox.warning(self, "No Device Selected", "Please select a device from the dropdown.")
            return

        stDevice = ctypes.cast(self.deviceList.pDeviceInfo[selected_index], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
        # Open the selected camera using its index (modify as needed for your camera API)
        ret = self.cam.MV_CC_CreateHandle(stDevice)
        if ret != 0:
            print(f"Failed to create handle! Error code: 0x{ret:X}")
            return

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print(f"Failed to open device! Error code: 0x{ret:X}")
            self.cam.MV_CC_DestroyHandle()
            return
        
        self.open_device_button.setEnabled(False)
        self.close_device_button.setEnabled(True)
        print("Device Opened", f"Camera {selected_index} opened successfully!")

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"Failed to start grabbing! Error code: 0x{ret:X}")
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
        else:
            # Start the video thread to continuously fetch frames
            self.video_thread = VideoThread(self.cam)
            self.video_thread.changePixmap.connect(self.update_image)
            self.video_thread.start()

    def update_image(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    
    def on_close_device(self):
        """Called when the Close Device button is pressed."""
        if hasattr(self, 'video_thread'):
            self.video_thread.stop()
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.video_thread.wait()  # Wait for the thread to finish
        
        self.open_device_button.setEnabled(True)
        self.close_device_button.setEnabled(False)
        print("Camera resources released.")

    def decoding_char(self, c_ubyte_value):
        c_char_p_value = ctypes.cast(c_ubyte_value, ctypes.c_char_p)
        try:
            decode_str = c_char_p_value.value.decode('gbk')
        except UnicodeDecodeError:
            decode_str = str(c_char_p_value.value)
        return decode_str
        
    def find_devices(self):
        self.cam = MvCamera()
        self.cam.MV_CC_Initialize()
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.deviceList)
        self.device_dropdown.clear()
        if ret != 0:
            QMessageBox.warning(self, "Error", f"Device enumeration failed! Error code: 0x{ret:X}")
            return
        if self.deviceList.nDeviceNum == 0:
            QMessageBox.information(self, "No Devices", "No camera devices found.")
            return
        for i in range(self.deviceList.nDeviceNum):
            mvcc_dev_info = cast(self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            user_defined_name = self.decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName)
            model_name = self.decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName)
            print("device user define name: " + user_defined_name)
            print("device model name: " + model_name)
            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber += chr(per)
            print("user serial number: " + strSerialNumber)
            self.device_dropdown.addItem(f"[{i}]USB: {user_defined_name} {model_name} ({strSerialNumber})")
