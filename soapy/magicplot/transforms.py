from __future__ import division
# Try importing PyQt5, if not fall back to PyQt4
try:
    from PyQt5 import QtCore, QtGui, QtWidgets, uic
    PYQTv = 5
except ImportError:
    from PyQt4 import QtCore, QtGui, uic
    QtWidgets = QtGui
    PyQTv = 4
# from PyQt4 import QtCore, QtGui
from . import transformPlugins

import copy

class Transformer(QtCore.QObject):
    """
    Controls transformation of data using MagicPlot TransformPlugins.

    Attributes:
        tList (transformPlugins.TransformList): List of available transforms
        aList (transformPlugins.TransformList): List of applied transforms
        active (bool): if True, then transforms are applied
        dialog (transformPlugins.TransformDialog): Dialog showing tList and aList,
            where the user can drag and drop from available to applied transforms
    """
    sigActiveToggle = QtCore.pyqtSignal(bool)
    def __init__(self):
        super(Transformer, self).__init__()
        self.dialog = transformPlugins.TransformDialog()
        self.active = False
        self.worker = Transformer_Worker()
        self.worker.plugin_dict = self.dialog.plugin_dict
        self.initContextMenu()


    def initContextMenu(self):
        """
        Right-click context menu shown under 'Transforms'
        """
        self.transMenu = QtGui.QMenu('Transforms')
        runTransforms = QtGui.QAction('Activate Transforms', self)
        runTransforms.setCheckable(True)
        runTransforms.toggled.connect(self.toggleRunning)
        self.transMenu.addAction(runTransforms)
        openDialog = QtGui.QAction('Open Transforms dialog', self)
        openDialog.triggered.connect(self.openDialog)
        self.transMenu.addAction(openDialog)
        self.transMenu.addSeparator()

        # add transforms to list below other options so they can be
        # quickly selected and applied
        self.quickTransforms = QtGui.QActionGroup(self)
        for row in range(self.dialog.tList.count()):
            name = self.dialog.tList.item(row).text()
            action = QtGui.QAction(name, self.quickTransforms)
            action.setData(name)
            self.transMenu.addAction(action)
        self.quickTransforms.triggered.connect(self.addFromContextMenu)

    def addFromContextMenu(self, action):
        """
        Use one of the quick transformations to transform data.
        This automatically enables transforms if they are not already
        active.
        """
        activeCheck = self.transMenu.actions()[0]
        if self.dialog.aList.count() != 0:
            self.dialog.aList.clear()
            activeCheck.setChecked(False)
        name = action.data()
        if type(name) is QtCore.QVariant:
            name = name.toString()
        self.dialog.aList.addItem(str(name))
        activeCheck.setChecked(True)

    def transform(self, data):
        if self.active:
            self.worker.data = data
            self.worker.aList = self.dialog.aList
            QtCore.QThreadPool.globalInstance().start(self.worker)
        else:
            pass
        
    def openDialog(self):
        """
        Open the transforms dialog
        """
        self.dialog.show()

    def toggleRunning(self, checked):
        """
        Handles the activate transforms menu option
        """
        self.active = checked
        self.sigActiveToggle.emit(checked)
        
class WorkerEmitter(QtCore.QObject):
    sigWorkerFinished = QtCore.pyqtSignal(object)

class Transformer_Worker(QtCore.QRunnable):

    def __init__(self, *args, **kwargs):
        super(Transformer_Worker, self).__init__(*args)
        self.data = None
        self.aList = None
        self.plugin_dict = None
        self.emitter = WorkerEmitter()
        self.setAutoDelete(False)

    def run(self):
        transformed_data = self.transform(self.data)
        self.emitter.sigWorkerFinished.emit(transformed_data)

    def transform(self, data):
        """
        Transform data by applying the transforms in the applied
        transforms list in order.

        Parameters:
            data (numpy.ndarray, tuple[numpy.ndarray]): the data to be
                transformed.

        Returns:
            numpy.ndarray: the transformed data
        """
        for row in range(self.aList.count()):
            plugin = self.plugin_dict[str(self.aList.item(row).text())]
            plugin.setData(data)
            data = plugin.transform()
        return data
