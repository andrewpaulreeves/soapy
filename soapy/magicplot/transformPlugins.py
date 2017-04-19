"""
=================
Transform Plugins
=================

Transform plugins act on data before it is plotted. To write a transform plugin,
create a class ``Plugin`` that inherits the base class ``TransformPlugin`` and
provide an ``__init__`` and a ``run()`` function.

In the ``__init__`` function call the ``__init__`` of the base class, and
provide parameters and default values as a dictionary, as well as a name for the
plugin::

    class Plugin(TransformPlugin):
        def __init__(self):
            TransformPlugin.__init__(
                params={'param1': param1, 'param2': param2},
                name ='MyTransform')

If any modules are needed (other than ``numpy``, which is already imported),
import them here. The ``params`` can be changed by the user through the
transforms dialog.

The ``run()`` method should take ``self.data`` as an input and return the
transformed data. Any parameters can be obtained from ``self.params``::

    def run(self):
        input = self.data
        param1, param2 = self.params['param1'], self.params['param2']
        output = transform(input, param1, param2)
        return output

Place the file containing the class into the ``MagicPlot/plugins/transforms``
directory and it will be autodetected by MagicPlot.

See the example plugins in the transforms directory for more information.
"""

import sys
# Try importing PyQt5, if not fall back to PyQt4
try:
    from PyQt5 import QtCore, QtGui, QtWidgets, uic
    PYQTv = 5
except ImportError:
    from PyQt4 import QtCore, QtGui, uic
    QtWidgets = QtGui
    PyQTv = 4
# from PyQt4 import QtGui, QtCore
import numpy
import os
import copy
PATH = os.path.dirname(os.path.abspath(__file__))

class TransformPlugin(QtCore.QObject):
    """
    Base class for transform plugins for MagicPlot
    """
    def __init__(self, params={}, name='Plugin'):
        super(TransformPlugin, self).__init__()
        self.params = params
        self.name = name

    def setData(self, data):
        self.data = data

    def setParams(self, params):
        for i in self.paramBoxList.keys():
            self.params[i] = self.paramBoxList[i].value()

    def transform(self):
        pass

    def generateUi(self):
        self.layout = QtGui.QGridLayout()
        self.paramBoxList = {}
        for i in self.params.keys():
            label = QtGui.QLabel(i)
            box = QtGui.QDoubleSpinBox()
            box.setValue(self.params[i])
            box.valueChanged.connect(self.setParams)
            self.paramBoxList[i] = box
            self.layout.addWidget(label)
            self.layout.addWidget(box)

class TransformDialog(QtGui.QDialog):
    """
    Dialog that shows available transforms in a QListView, and active
    transforms in another. Transforms are applied from top to bottom
    in the QListView.

    Transforms can be applied to data in any order by dragging and
    dropping.
    """

    def __init__(self, tList=None, aList=None):
        super(TransformDialog, self).__init__()
        self.setupUi(tList, aList)

    def setupUi(self, tList, aList):
        self.layout = QtGui.QGridLayout()
        self.tViewLabel = QtGui.QLabel('Available Transforms')
        self.activeViewLabel = QtGui.QLabel('Applied Transforms')
        self.layout.addWidget(self.tViewLabel, 0,0)
        self.layout.addWidget(self.activeViewLabel, 0, 1)
        self.tList = TransformListView(parent=self)
        self.aList = ActiveTransformListView(parent=self)
        self.layout.addWidget(self.tList, 1,0)
        self.layout.addWidget(self.aList, 1,1)
        self.setLayout(self.layout)
        self.getTransforms()

    def getTransforms(self):
        """
        Search the directory './plugins/transforms' for plugins and add them
        to the list
        """
        self.plugin_dict = {}
        path = os.path.abspath(os.path.join(PATH, './plugins/transforms'))
        for i in os.listdir(path):
            fname = os.path.join(path, i)
            with open(fname, 'r') as f:
                exec(f.read(), globals())
                plugin = Plugin()
                self.plugin_dict[plugin.name] = plugin
                self.tList.addItem(plugin.name)

class TransformListView(QtGui.QListWidget):
    """
    List view showing transforms, can be dragged and dropped to active
    transform ListView
    """
    def __init__(self, parent=None):
        super(TransformListView, self).__init__(parent)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)

    # def dragEnterEvent(self, event):
    #     if event.mimeData().hasFormat("text/plain"):
    #         event.setDropAction(QtCore.Qt.MoveAction)
    #         event.accept()
    #     else:
    #         event.ignore()

    # def startDrag(self, event):
    #     index = self.indexAt(event.pos())
    #     if not index.isValid():
    #         return

    #     mimeData = QtCore.QMimeData()
    #     mimeData.setText(str(index.row()))

    #     drag = QtGui.QDrag(self)
    #     drag.setMimeData(mimeData)

    #     result = drag.exec_(QtCore.Qt.MoveAction)

    # def mouseMoveEvent(self, event):
    #     self.startDrag(event)

class ActiveTransformListView(QtGui.QListWidget):
    """
    List view showing active transforms
    """
    def __init__(self, parent=None):
        super(ActiveTransformListView, self).__init__(parent)
        self.setDropIndicatorShown(True)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QtGui.QAbstractItemView.DragDrop)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        # connect double click to remove plugin, click
        # to bring up param dialog
        # self.itemClicked.connect(self.openParamDialog)
        self.customContextMenuRequested.connect(self.showContextMenu)

    def dropEvent(self, event):
        if event.source() != self:
            super(ActiveTransformListView, self).dropEvent(event)
        else:
            event.ignore()

    def removeItem(self):
        for i in self.selectedItems():
            self.takeItem(self.row(i))

    def openParamDialog(self):
        item = self.currentItem()
        plugin = self.parent().plugin_dict[item.text()]
        if plugin.params != {}:
            dialog = ParamDialog(plugin)
            dialog.exec_()

    def showContextMenu(self, pos):
        globalpos = self.mapToGlobal(pos)
        contextMenu = QtGui.QMenu()
        delete_action = QtGui.QAction('Delete', self)
        delete_action.triggered.connect(self.removeItem)
        contextMenu.addAction(delete_action)
        paramdialog_action = QtGui.QAction('Parameters', self)
        paramdialog_action.triggered.connect(self.openParamDialog)
        contextMenu.addAction(paramdialog_action)
        contextMenu.exec_(globalpos)

# class TransformList(QtCore.QAbstractListModel):
#     """
#     Model for TransformListView and ActiveTransformListView
#     """
#     def __init__(self, parent):
#         super(TransformList, self).__init__(parent)
#         self.parent = parent
#         self.tList = []

#     def rowCount(self, parent=QtCore.QModelIndex()):
#         if parent.isValid(): return 0
#         return len(self.tList)

#     def flags(self, index):
#         if index.isValid():
#             return QtCore.Qt.ItemIsSelectable| \
#                     QtCore.Qt.ItemIsDragEnabled| \
#                     QtCore.Qt.ItemIsEnabled
#         return QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsDragEnabled| \
#                 QtCore.Qt.ItemIsDropEnabled|QtCore.Qt.ItemIsEnabled

#     def data(self, index, role=QtCore.Qt.DisplayRole):
#         if not index.isValid(): return QtCore.QVariant()
#         if role == QtCore.Qt.DisplayRole: return self.tList[index.row()].name
#         elif role == QtCore.Qt.UserRole:
#             plugin = self.tList[index.row()]
#             return plugin
#         return QtCore.QVariant()

#     def setData(self, index, value, role=QtCore.Qt.EditRole):
#         if not index.isValid() or role!=QtCore.Qt.DisplayRole: return False

#         self.tList[index.row()]=value.name
#         self.dataChanged.emit(index,index)
#         return True

#     def append(self, transform):
#         self.beginInsertRows(QtCore.QModelIndex(), 0, 0)
#         self.tList.append(transform)
#         self.endInsertRows()

#     def insertRows(self, row, count, parent=QtCore.QModelIndex(), plugin=None):
#         if parent.isValid(): return False

#         beginRow=max(0,row)
#         endRow=min(row+count-1,len(self.tList))

#         self.beginInsertRows(parent, beginRow, endRow)

#         for i in xrange(beginRow, endRow+1): self.tList.insert(i,plugin)

#         self.endInsertRows()
#         return True

#     def removeRows(self, row, count, parent=QtCore.QModelIndex()):
#         if parent.isValid(): return False
#         if row >= len(self.tList) or row+count <=0: return False

#         beginRow = max(0,row)
#         endRow = min(row+(count-1), len(self.tList)-1)

#         self.beginRemoveRows(parent, beginRow, endRow)

#         for i in range(beginRow, endRow+1): self.tList.pop(i)

#         self.endRemoveRows()
#         return True

#     def clear(self):
#         count = self.rowCount()
#         self.removeRows(0,count)

#     def __getitem__(self, index):
#         return self.tList[index]

#     def getTransforms(self):
#         """
#         Search the directory './plugins/transforms' for plugins and add them
#         to the list
#         """
#         path = os.path.abspath(os.path.join(PATH, './plugins/transforms'))
#         for i in os.listdir(path):
#             fname = os.path.join(path, i)
#             with open(fname, 'r') as f:
#                 exec(f.read(), globals())
#                 plugin = Plugin()
#                 self.append(plugin)

#     def dropMimeData(self, data, action, row, column, parent):
#         tListRow = int(data.data("text/plain"))
#         plugin = copy.copy(self.parent[tListRow])
#         if action == QtCore.Qt.CopyAction:
#             if row == -1:
#                 self.append(plugin)
#             else:
#                 self.insertRows(row, 1, plugin=plugin)
#             return True
#         else: return False

class ParamDialog(QtGui.QDialog):
    """
    Dialog to get user defined parameters for a particular
    plugin in the active plugins list

    Parameters:
        plugin (TransformPlugin): The selected plugin whose parameters
            you want to change.
    """
    def __init__(self, plugin):
        super(ParamDialog, self).__init__()
        self.plugin=plugin
        self.setupUi()

    def setupUi(self):
        """
        Generates the Ui of the dialog by calling the generateUi()
        method of the plugin
        """
        self.plugin.generateUi()
        self.setLayout(self.plugin.layout)


if __name__ == '__main__':
    app = QtGui.QApplication([])
    tList = TransformList(None)
    aList = TransformList(None)
    Dialog = TransformDialog(tList=tList, aList=aList)
    Dialog.show()
    try:
        __IPYTHON__
    except NameError:
        __IPYTHON__=False

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        if not __IPYTHON__:
            QtGui.QApplication.instance().exec_()
