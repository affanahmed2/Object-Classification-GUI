# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PyUICBasicDemo.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(747, 486)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.ComboDevices = QtWidgets.QComboBox(self.centralWidget)
        self.ComboDevices.setGeometry(QtCore.QRect(10, 20, 511, 22))
        self.ComboDevices.setObjectName("ComboDevices")
        self.widgetDisplay = QtWidgets.QWidget(self.centralWidget)
        self.widgetDisplay.setGeometry(QtCore.QRect(10, 60, 511, 401))
        self.widgetDisplay.setObjectName("widgetDisplay")
        self.groupInit = QtWidgets.QGroupBox(self.centralWidget)
        self.groupInit.setGeometry(QtCore.QRect(530, 20, 211, 101))
        self.groupInit.setObjectName("groupInit")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupInit)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(9, 19, 201, 81))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.bnClose = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.bnClose.setEnabled(False)
        self.bnClose.setObjectName("bnClose")
        self.gridLayout.addWidget(self.bnClose, 2, 2, 1, 1)
        self.bnOpen = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.bnOpen.setObjectName("bnOpen")
        self.gridLayout.addWidget(self.bnOpen, 2, 1, 1, 1)
        self.bnEnum = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.bnEnum.setObjectName("bnEnum")
        self.gridLayout.addWidget(self.bnEnum, 1, 1, 1, 2)
        self.groupGrab = QtWidgets.QGroupBox(self.centralWidget)
        self.groupGrab.setEnabled(False)
        self.groupGrab.setGeometry(QtCore.QRect(530, 130, 211, 171))
        self.groupGrab.setObjectName("groupGrab")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.groupGrab)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(9, 19, 202, 141))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.bnSaveImage = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.bnSaveImage.setEnabled(False)
        self.bnSaveImage.setObjectName("bnSaveImage")
        self.gridLayout_2.addWidget(self.bnSaveImage, 4, 0, 1, 2)
        self.radioContinueMode = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.radioContinueMode.setObjectName("radioContinueMode")
        self.gridLayout_2.addWidget(self.radioContinueMode, 0, 0, 1, 1)
        self.radioTriggerMode = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.radioTriggerMode.setObjectName("radioTriggerMode")
        self.gridLayout_2.addWidget(self.radioTriggerMode, 0, 1, 1, 1)
        self.bnStop = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.bnStop.setEnabled(False)
        self.bnStop.setObjectName("bnStop")
        self.gridLayout_2.addWidget(self.bnStop, 2, 1, 1, 1)
        self.bnStart = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.bnStart.setEnabled(False)
        self.bnStart.setObjectName("bnStart")
        self.gridLayout_2.addWidget(self.bnStart, 2, 0, 1, 1)
        self.bnSoftwareTrigger = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.bnSoftwareTrigger.setEnabled(False)
        self.bnSoftwareTrigger.setObjectName("bnSoftwareTrigger")
        self.gridLayout_2.addWidget(self.bnSoftwareTrigger, 3, 0, 1, 2)
        self.groupParam = QtWidgets.QGroupBox(self.centralWidget)
        self.groupParam.setEnabled(False)
        self.groupParam.setGeometry(QtCore.QRect(530, 310, 211, 151))
        self.groupParam.setObjectName("groupParam")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.groupParam)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(10, 20, 201, 131))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayoutParam = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayoutParam.setContentsMargins(11, 11, 11, 11)
        self.gridLayoutParam.setSpacing(6)
        self.gridLayoutParam.setObjectName("gridLayoutParam")
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_6.setObjectName("label_6")
        self.gridLayoutParam.addWidget(self.label_6, 3, 0, 1, 1)
        self.edtGain = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.edtGain.setObjectName("edtGain")
        self.gridLayoutParam.addWidget(self.edtGain, 1, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_5.setObjectName("label_5")
        self.gridLayoutParam.addWidget(self.label_5, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_4.setObjectName("label_4")
        self.gridLayoutParam.addWidget(self.label_4, 0, 0, 1, 1)
        self.edtExposureTime = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.edtExposureTime.setObjectName("edtExposureTime")
        self.gridLayoutParam.addWidget(self.edtExposureTime, 0, 1, 1, 1)
        self.bnGetParam = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.bnGetParam.setObjectName("bnGetParam")
        self.gridLayoutParam.addWidget(self.bnGetParam, 4, 0, 1, 1)
        self.bnSetParam = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.bnSetParam.setObjectName("bnSetParam")
        self.gridLayoutParam.addWidget(self.bnSetParam, 4, 1, 1, 1)
        self.edtFrameRate = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.edtFrameRate.setObjectName("edtFrameRate")
        self.gridLayoutParam.addWidget(self.edtFrameRate, 3, 1, 1, 1)
        self.gridLayoutParam.setColumnStretch(0, 2)
        self.gridLayoutParam.setColumnStretch(1, 3)
        MainWindow.setCentralWidget(self.centralWidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupInit.setTitle(_translate("MainWindow", "Initialization"))
        self.bnClose.setText(_translate("MainWindow", "Close Device"))
        self.bnOpen.setText(_translate("MainWindow", "Open Device"))
        self.bnEnum.setText(_translate("MainWindow", "Find Device"))
        self.groupGrab.setTitle(_translate("MainWindow", "Acquisition"))
        self.bnSaveImage.setText(_translate("MainWindow", "Save Image"))
        self.radioContinueMode.setText(_translate("MainWindow", "Continuous Mode"))
        self.radioTriggerMode.setText(_translate("MainWindow", "Trigger Mode"))
        self.bnStop.setText(_translate("MainWindow", "Stop Acquisition"))
        self.bnStart.setText(_translate("MainWindow", "Start Acquisition"))
        self.bnSoftwareTrigger.setText(_translate("MainWindow", "Software Trigger Once"))
        self.groupParam.setTitle(_translate("MainWindow", "Parameters"))
        self.label_6.setText(_translate("MainWindow", "Frame Rate"))
        self.edtGain.setText(_translate("MainWindow", "0"))
        self.label_5.setText(_translate("MainWindow", "Gain"))
        self.label_4.setText(_translate("MainWindow", "Exposure"))
        self.edtExposureTime.setText(_translate("MainWindow", "0"))
        self.bnGetParam.setText(_translate("MainWindow", "Get Parameters"))
        self.bnSetParam.setText(_translate("MainWindow", "Set Parameters"))
        self.edtFrameRate.setText(_translate("MainWindow", "0"))

