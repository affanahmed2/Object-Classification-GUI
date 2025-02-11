import os
import shutil
import sys
from PyQt5.QtWidgets import (
    QWidget, QTabWidget, QVBoxLayout, QLabel, QGroupBox, QHBoxLayout,
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QScrollArea, QComboBox
)
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl, pyqtSignal, Qt

sys.path.append("imports/")
from MvCameraControl_class import *

class MainWindow(QWidget):
    # Signal that will be emitted when the Train button is pressed.
    trainClicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Classification")
        self.resize(800, 600)
        
        # Create the tab widget
        self.tabs = QTabWidget()
        
        # Setup Training tab with class management features
        self.training_tab = QWidget()
        self.setup_training_tab()
        
        # Setup Classification tab with device detection controls
        self.classification_tab = QWidget()
        classification_layout = QVBoxLayout()
        
        # Create a horizontal layout with an empty dropdown menu and "Find Devices" button
        device_layout = QHBoxLayout()
        self.device_dropdown = QComboBox()  # Initially empty
        self.find_devices_button = QPushButton("Find Devices")
        self.find_devices_button.clicked.connect(self.find_devices)
        device_layout.addWidget(self.device_dropdown)
        device_layout.addWidget(self.find_devices_button)
        
        classification_layout.addLayout(device_layout)
        self.classification_tab.setLayout(classification_layout)
        
        # Add tabs to the tab widget
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.classification_tab, "Classification")
        
        # Set the main layout of the window
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)
    
    def setup_training_tab(self):
        # Main layout for training tab
        layout = QVBoxLayout()
        
        # Create a scrollable area for the classes list.
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.classes_container = QWidget()
        self.classes_layout = QVBoxLayout()
        self.classes_container.setLayout(self.classes_layout)
        self.scroll_area.setWidget(self.classes_container)
        
        # Group Box for classes list
        self.classes_group = QGroupBox("Classes List")
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.scroll_area)
        self.classes_group.setLayout(group_layout)
        layout.addWidget(self.classes_group)
        
        # Management area to add a new class and delete all classes,
        # with fixed height.
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
        
        # Train button at the bottom centre.
        train_button_layout = QHBoxLayout()
        train_button_layout.setAlignment(Qt.AlignCenter)
        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.handle_train_button)
        train_button_layout.addWidget(self.train_button)
        layout.addLayout(train_button_layout)
        
        self.training_tab.setLayout(layout)
        
        # Ensure the "dataset" folder exists.
        self.dataset_dir = os.path.join(os.getcwd(), "dataset")
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        
        # Dictionary to keep track of class widgets: {class_name: widget}
        self.class_widgets = {}
        
        # Load preexisting classes from the dataset folder.
        self.load_existing_classes()
    
    def load_existing_classes(self):
        # Scan the dataset directory for subdirectories (each is a class)
        for entry in os.listdir(self.dataset_dir):
            class_folder = os.path.join(self.dataset_dir, entry)
            if os.path.isdir(class_folder):
                widget = self.create_class_widget(entry, class_folder)
                self.classes_layout.addWidget(widget)
                self.class_widgets[entry] = widget
    
    def create_class(self):
        # Retrieve the class name from input.
        class_name = self.class_name_input.text().strip()
        if not class_name:
            QMessageBox.warning(self, "Input Error", "Please enter a class name.")
            return
        # Define the folder for this class under the dataset directory.
        class_folder = os.path.join(self.dataset_dir, class_name)
        if os.path.exists(class_folder):
            QMessageBox.warning(self, "Error", f"Class '{class_name}' already exists.")
            return
        os.makedirs(class_folder)
        # Create a widget representing this class and add it to the layout.
        widget = self.create_class_widget(class_name, class_folder)
        self.classes_layout.addWidget(widget)
        self.class_widgets[class_name] = widget
        self.class_name_input.clear()
    
    def create_class_widget(self, class_name, class_folder):
        # Create a widget with a horizontal layout for class actions.
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
        # Open the folder using the default file explorer.
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
    
    def add_images(self, class_folder):
        # Open a file dialog for the user to select images to add to the class folder.
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Training Images", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options
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
        # Confirm deletion of the class and its folder.
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
            # Remove the widget from the layout and the dictionary.
            widget.setParent(None)
            del self.class_widgets[class_name]
    
    def delete_all_classes(self):
        # Confirm deletion of all classes.
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
        # Emit the trainClicked signal when the Train button is pressed.
        self.trainClicked.emit()
    
    def decoding_char(self, c_ubyte_value):
        c_char_p_value = ctypes.cast(c_ubyte_value, ctypes.c_char_p)
        try:
            decode_str = c_char_p_value.value.decode('gbk')  # Chinese characters
        except UnicodeDecodeError:
            decode_str = str(c_char_p_value.value)
        return decode_str
        
    def find_devices(self):
        
        # Initialize Camera SDK
        cam = MvCamera()
        cam.MV_CC_Initialize()
        
        # Enumerate Devices
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
        
        # Clear the dropdown before adding new devices.
        self.device_dropdown.clear()
        
        if ret != 0:
            QMessageBox.warning(self, "Error", f"Device enumeration failed! Error code: 0x{ret:X}")
            return
        
        if deviceList.nDeviceNum == 0:
            QMessageBox.information(self, "No Devices", "No camera devices found.")
            return
        
        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            user_defined_name = self.decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName)
            model_name = self.decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName)
            print("device user define name: " + user_defined_name)
            print("device model name: " + model_name)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: " + strSerialNumber)
            self.device_dropdown.addItem("[" + str(i) + "]USB: " + user_defined_name + " " + model_name + "(" + str(strSerialNumber) + ")")
