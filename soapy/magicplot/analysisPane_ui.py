# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/apr/CfAI/magicPlot/magicplot/analysisPane.ui'
#
# Created: Mon Sep 14 15:29:54 2015
#      by: PyQt4 UI code generator 4.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_AnalysisPane(object):
    def setupUi(self, AnalysisPane):
        AnalysisPane.setObjectName(_fromUtf8("AnalysisPane"))
        AnalysisPane.resize(461, 300)
        self.gradientTab = QtGui.QWidget()
        self.gradientTab.setObjectName(_fromUtf8("gradientTab"))
        self.gridLayoutWidget = QtGui.QWidget(self.gradientTab)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(2, 0, 451, 81))
        self.gridLayoutWidget.setObjectName(_fromUtf8("gridLayoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label = QtGui.QLabel(self.gridLayoutWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 2, 1, 1)
        self.x1Box = QtGui.QDoubleSpinBox(self.gridLayoutWidget)
        self.x1Box.setObjectName(_fromUtf8("x1Box"))
        self.gridLayout.addWidget(self.x1Box, 0, 1, 1, 1)
        self.x2Box = QtGui.QDoubleSpinBox(self.gridLayoutWidget)
        self.x2Box.setObjectName(_fromUtf8("x2Box"))
        self.gridLayout.addWidget(self.x2Box, 0, 3, 1, 1)
        self.gradientLabel = QtGui.QLabel(self.gridLayoutWidget)
        self.gradientLabel.setObjectName(_fromUtf8("gradientLabel"))
        self.gridLayout.addWidget(self.gradientLabel, 2, 0, 1, 1)
        self.checkRegion = QtGui.QCheckBox(self.gridLayoutWidget)
        self.checkRegion.setObjectName(_fromUtf8("checkRegion"))
        self.gridLayout.addWidget(self.checkRegion, 0, 0, 1, 1)
        self.gradientDisplay = QtGui.QLabel(self.gridLayoutWidget)
        self.gradientDisplay.setText(_fromUtf8(""))
        self.gradientDisplay.setObjectName(_fromUtf8("gradientDisplay"))
        self.gridLayout.addWidget(self.gradientDisplay, 2, 1, 1, 1)
        self.updateButton = QtGui.QPushButton(self.gridLayoutWidget)
        self.updateButton.setObjectName(_fromUtf8("updateButton"))
        self.gridLayout.addWidget(self.updateButton, 0, 4, 1, 1)
        AnalysisPane.addTab(self.gradientTab, _fromUtf8(""))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        AnalysisPane.addTab(self.tab, _fromUtf8(""))

        self.retranslateUi(AnalysisPane)
        AnalysisPane.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(AnalysisPane)

    def retranslateUi(self, AnalysisPane):
        AnalysisPane.setWindowTitle(_translate("AnalysisPane", "TabWidget", None))
        self.label.setText(_translate("AnalysisPane", "to", None))
        self.gradientLabel.setText(_translate("AnalysisPane", "Gradient:", None))
        self.checkRegion.setText(_translate("AnalysisPane", "Region:", None))
        self.updateButton.setText(_translate("AnalysisPane", "Update", None))
        AnalysisPane.setTabText(AnalysisPane.indexOf(self.gradientTab), _translate("AnalysisPane", "Tab 1", None))
        AnalysisPane.setTabText(AnalysisPane.indexOf(self.tab), _translate("AnalysisPane", "Tab 2", None))
