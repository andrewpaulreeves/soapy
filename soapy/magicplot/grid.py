"""
Contains a 'grid' object, that is drawn onto a magic plot canvas
"""
import os

# Try importing PyQt5, if not fall back to PyQt4
try:
    from PyQt5 import QtGui, QtWidgets, uic
    PYQTv = 5
except (ImportError, RuntimeError):
    from PyQt4 import QtGui, uic
    QtWidgets = QtGui
    PYQTv = 4

PATH = os.path.dirname(os.path.abspath(__file__))
Ui_ShapeDrawer= uic.loadUiType(os.path.join(PATH,"shapeDrawer.ui"))[0]

class Grid(QtWidgets.QGraphicsRectItem):
    def __init__(self, rect, nRows, nColumns):
        super(Grid, self).__init__()
        self.outRect = QtWidgets.QGraphicsRectItem(rect, self)
        self.nRows = nRows
        self.nColumns = nColumns
        self.update()

    def setRect(self, rect):
        self.outRect.setRect(rect)
        self.update()

    def update(self):
        self.vSpacing = self.outRect.rect().height() / self.nRows
        self.hSpacing = self.outRect.rect().width() / self.nColumns
        for i, line in enumerate(self.hLines):
            x1 = self.outRect.rect().left()
            y1 = self.outRect.rect().top() + (i+1)*self.vSpacing
            x2 = self.outRect.rect().right()
            y2 = y1
            line.setLine(x1, y1, x2, y2)
            # line.setPen(self.outRect.pen())
        for j, line in enumerate(self.vLines):
            x1 = self.outRect.rect().left() + (j+1)*self.hSpacing
            y1 = self.outRect.rect().top()
            x2 = x1
            y2 = self.outRect.rect().bottom()
            line.setLine(x1, y1, x2, y2)
            # line.setPen(self.outRect.pen())


    @property
    def shapes(self):
        return [self.outRect] + self.hLines + self.vLines

    # @property
    # def color(self):
    #     return self._color

    # @color.setter
    # def color(self, color):
    #     self._color = color
    #     self._pen = QtGui.QPen(color)
    #     self.outRect.setPen(self._pen)
    #     for i in self.vLines + self.hLines:
    #         i.setPen(QtGui.QPen(color))

    def setPen(self, pen):
        self.outRect.setPen(pen)
        for i in self.vLines + self.hLines:
            i.setPen(pen)
        super(Grid, self).setPen(pen)

    @property
    def nRows(self):
        return self._nRows

    @nRows.setter
    def nRows(self, nRows):
        self._nRows = nRows
        try:
            for i in self.hLines:
                i.setVisible(False)
        except AttributeError:
            pass
        self.hLines = []
        self.hLines = [QtGui.QGraphicsLineItem(self) for i in range(nRows-1)]

    @property
    def nColumns(self):
        return self._nColumns

    @nColumns.setter
    def nColumns(self, nColumns):
        self._nColumns = nColumns
        try:
            for i in self.vLines:
                i.setVisible(False)
        except AttributeError:
            pass
        self.vLines = []
        self.vLines = [QtGui.QGraphicsLineItem(self) for i in range(nColumns-1)]