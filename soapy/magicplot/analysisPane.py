from __future__ import division
import sys
import os
import importlib
# SRC_PATH = os.path.dirname(os.path.abspath(__file__))
# os.system("pyuic4 {0}/analysisPane.ui > {0}/analysisPane_ui.py".format(SRC_PATH))
# Try importing PyQt5, if not fall back to PyQt4
try:
    from PyQt5 import QtCore, QtGui, QtWidgets, uic
    PYQTv = 5
except ImportError:
    from PyQt4 import QtCore, QtGui, uic
    QtWidgets = QtGui
    PyQTv = 4

from .analysisPlugins import AnalysisPlugin, Analyser
from . import pyqtgraph

import numpy
import magicplot
import logging

# Get this files path so we can get plugin
import os
PATH = os.path.dirname(os.path.abspath(__file__))

class AnalysisPane(QtWidgets.QWidget):
    
    runPluginSignal = QtCore.pyqtSignal(object)

    def __init__(self, parent=None, view=None, item=None):
        super(AnalysisPane, self).__init__(parent)
        self.getPluginList()
        self.setupUi()
        self.analyser = Analyser(self)
        self.analyser.pluginList = self.pluginList
        self.analyser.emitter.sigAnalysisFinished.connect(self.updateOutput)
        self.runPluginSignal.connect(self.updateData)
        QtCore.QThreadPool.globalInstance().start(self.analyser)
        self.data = None
        self.active = False

        # get a reference to the magicplot
        self.mplot = parent

    def setupUi(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.regionCheckbox = QtWidgets.QCheckBox('Region of Interest')
        self.regionCheckbox.toggled.connect(self.toggleRegion)
        self.activeCheckbox = QtWidgets.QCheckBox('Activate Analysis')
        self.activeCheckbox.toggled.connect(self.checkActiveHandler)
        self.tabWidget = QtWidgets.QTabWidget()
        for i in self.pluginList:
            i.generateUi()
            self.tabWidget.addTab(i, i.name)
        self.layout.addWidget(self.activeCheckbox)
        self.layout.addWidget(self.regionCheckbox)
        self.layout.addWidget(self.tabWidget)
        self.setLayout(self.layout)

    def checkActiveHandler(self, checked):
        if checked:
            self.active = True
        else:
            self.active = False

    def updateData(self, data):
        if self.active:
            if data is not None:
                self.data = data
            try:
                if self.region.isVisible():
                    pluginData = self.region.setData(self.data)
                    self.region.setBounds((self.data[0][0], self.data[0][-1]))
                else:
                    pluginData = self.data
            except AttributeError:
                pluginData = self.data
                logging.info('No 1D region - probably a 2d Plot')
            except RuntimeError as e:
                pluginData = self.data
                logging.info(e)

            for i in self.pluginList:
                i.setData(pluginData)
            self.runPlugins()

    def runPlugins(self):
        if self.active:
            self.analyser.data = self.data
            QtCore.QThreadPool.globalInstance().start(self.analyser)

    def updateOutput(self, outputDict):
        for i,j in zip(outputDict, self.pluginList):
            j.setOutput(i)

    def getPluginList(self):
        self.pluginList = []
        path = os.path.abspath(os.path.join(PATH, './plugins/analysis'))
        for i in os.listdir(path):
            fname = os.path.join(path, i)
            with open(fname, 'r') as f:
                exec(f.read(), globals())
                self.pluginList.append(Plugin())

    def toggleRegion(self, checked):
        # make sure we're in 1D plotMode!
        if self.mplot.plotMode == 1:
            if checked:
                self.region.setVisible(True)
                self.region.setRegion(
                        (self.data[0][0], self.data[0][-1]))
                self.region.setBounds(
                        (self.data[0][0], self.data[0][-1]))
                self.region.sigRegionChanged.connect(
                        lambda: self.updateData(None))
                self.updateData(None)
            else:
                self.region.setVisible(False)
                self.region.sigRegionChanged.disconnect()
                self.updateData(None)

        else:
            pass

    def initRegion(self, view):
        self.region = Region()
        self.region.setVisible(False)
        view.addItem(self.region)

class Region(pyqtgraph.LinearRegionItem):

    def __init__(self):
        super(Region, self).__init__()

    def setData(self, data):
        x1, x2 = self.getRegion()
        index1 = int((x1 - data[0][0])/data[0][-1] * len(data[0]))
        index2 = int((x2 - data[0][0])/data[0][-1] * len(data[0]))
        self.data = [data[0][index1:index2], data[1][index1:index2]]
        return self.data
