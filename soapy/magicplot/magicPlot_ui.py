# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'magicPlot.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MagicPlot(object):
    def setupUi(self, MagicPlot):
        MagicPlot.setObjectName("MagicPlot")
        MagicPlot.resize(658, 600)
        self.gridLayout = QtWidgets.QGridLayout(MagicPlot)
        self.gridLayout.setObjectName("gridLayout")
        self.analysisSplitter = QtWidgets.QSplitter(MagicPlot)
        self.analysisSplitter.setOrientation(QtCore.Qt.Vertical)
        self.analysisSplitter.setObjectName("analysisSplitter")
        self.drawSplitter = QtWidgets.QSplitter(self.analysisSplitter)
        self.drawSplitter.setOrientation(QtCore.Qt.Horizontal)
        self.drawSplitter.setHandleWidth(2)
        self.drawSplitter.setObjectName("drawSplitter")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.drawSplitter)
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.plotContainerLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.plotContainerLayout.setContentsMargins(0, 0, 0, 0)
        self.plotContainerLayout.setObjectName("plotContainerLayout")
        self.plotLayout = QtWidgets.QHBoxLayout()
        self.plotLayout.setObjectName("plotLayout")
        self.plotContainerLayout.addLayout(self.plotLayout)
        self.mousePosLabel = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.mousePosLabel.setText("")
        self.mousePosLabel.setObjectName("mousePosLabel")
        self.plotContainerLayout.addWidget(self.mousePosLabel)
        self.gridLayout.addWidget(self.analysisSplitter, 0, 0, 1, 1)

        self.retranslateUi(MagicPlot)
        QtCore.QMetaObject.connectSlotsByName(MagicPlot)

    def retranslateUi(self, MagicPlot):
        _translate = QtCore.QCoreApplication.translate
        MagicPlot.setWindowTitle(_translate("MagicPlot", "Form"))

