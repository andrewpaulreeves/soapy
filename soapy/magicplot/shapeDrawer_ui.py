# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'shapeDrawer.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ShapeDrawer(object):
    def setupUi(self, ShapeDrawer):
        ShapeDrawer.setObjectName("ShapeDrawer")
        ShapeDrawer.resize(137, 317)
        self.gridLayout = QtWidgets.QGridLayout(ShapeDrawer)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.drawRectButton = QtWidgets.QPushButton(ShapeDrawer)
        self.drawRectButton.setObjectName("drawRectButton")
        self.verticalLayout.addWidget(self.drawRectButton)
        self.drawLineButton = QtWidgets.QPushButton(ShapeDrawer)
        self.drawLineButton.setObjectName("drawLineButton")
        self.verticalLayout.addWidget(self.drawLineButton)
        self.drawGridButton = QtWidgets.QPushButton(ShapeDrawer)
        self.drawGridButton.setObjectName("drawGridButton")
        self.verticalLayout.addWidget(self.drawGridButton)
        self.drawCircleButton = QtWidgets.QPushButton(ShapeDrawer)
        self.drawCircleButton.setObjectName("drawCircleButton")
        self.verticalLayout.addWidget(self.drawCircleButton)
        self.drawElipseButton = QtWidgets.QPushButton(ShapeDrawer)
        self.drawElipseButton.setObjectName("drawElipseButton")
        self.verticalLayout.addWidget(self.drawElipseButton)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(ShapeDrawer)
        QtCore.QMetaObject.connectSlotsByName(ShapeDrawer)

    def retranslateUi(self, ShapeDrawer):
        _translate = QtCore.QCoreApplication.translate
        ShapeDrawer.setWindowTitle(_translate("ShapeDrawer", "Form"))
        self.drawRectButton.setText(_translate("ShapeDrawer", "Rectangle"))
        self.drawLineButton.setText(_translate("ShapeDrawer", "Line"))
        self.drawGridButton.setText(_translate("ShapeDrawer", "Grid"))
        self.drawCircleButton.setText(_translate("ShapeDrawer", "Circle"))
        self.drawElipseButton.setText(_translate("ShapeDrawer", "Ellipse"))

