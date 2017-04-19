"""
A class to draw shapes onto a QGraphicsView
"""
import os
# SRC_PATH = os.path.dirname(os.path.abspath(__file__))
# os.system("pyuic4 {0}/shapeDrawer.ui > {0}/shapeDrawer_ui.py".format(SRC_PATH))

# Try importing PyQt5, if not fall back to PyQt4
try:
    from PyQt5 import QtCore, QtGui, QtWidgets, uic
    PYQTv = 5
except (ImportError, RuntimeError):
    from PyQt4 import QtCore, QtGui, uic
    QtWidgets = QtGui
    PyQTv = 4

PATH = os.path.dirname(os.path.abspath(__file__))
Ui_ShapeDrawer= uic.loadUiType(os.path.join(PATH,"shapeDrawer.ui"))[0]

QPEN_WIDTH = 0

from . import shapeHolder
from .grid import Grid
from .pyqtgraph import RectROI, CircleROI, LineSegmentROI
import magicplot
import numpy
import logging

class ShapeDrawer(QtWidgets.QWidget, Ui_ShapeDrawer):

    """
    A Widget providing a list of shapes which can be drawn onto
    a QGraphicsView. The shapes are then added to a list and can
    be later removed or edited.

    Parameters:
        view (QGraphicsView): The Graphics View to draw onto
    """

    def __init__(self, view=None, item=None):
        # Run init on the QWidget class
        super(ShapeDrawer, self).__init__()
        self.setupUi(self)

        # Drawing buttons
        self.drawRectButton.clicked.connect(self.drawRect)
        self.drawLineButton.clicked.connect(self.drawLine)
        self.drawGridButton.clicked.connect(self.drawGrid)
        self.drawCircleButton.clicked.connect(self.drawCirc)
        self.drawElipseButton.clicked.connect(self.drawElipse)

        # Setup list to hold shapes
        self.shapes = shapeHolder.ShapeContainer(self)
        self.shapeList = ShapeList(self)
        self.verticalLayout.addWidget(self.shapeList)
        self.shapeList.setModel(self.shapes)

        # Connect double click to delete shape
        self.shapeList.doubleClicked.connect(self.openDialog)
        self.shapeList.delKeySig.connect(self.shapes.removeShape)

        # Button to plot selected RoI
        self.plotRoiButton = QtWidgets.QPushButton('Plot RoI')
        self.plotRoiButton.setEnabled(False)
        self.verticalLayout.addWidget(self.plotRoiButton)

        # Init roi
        self.roi = None

        # Create the defualt pen for shape drawing
        brush = QtGui.QBrush(QtGui.QColor('red'))
        self.pen = QtGui.QPen(brush, QPEN_WIDTH)

        self.setView(view, item)

    def getShapes(self):
        return self.shapes

    def setView(self, view, items):
        self.plotView = view
        self.plotItems = items
        try:
            self.viewBox = self.plotView.getViewBox()
        except AttributeError:
            pass
        self.clearShapes()
        # Get the scene object from the view.
        # pyqtgraph imageView is inconsistant, hence try/except
        if view!=None:
            try:
                self.scene = self.plotView.scene()
            except TypeError:
                self.scene = self.plotView.scene

    def clearShapes(self):
        self.shapes.clearShapes()

############ Dialog methods

    def openDialog(self, index):
        """
        Opens shape dialogs to edit shapes.

        Parameters:
            index (QtCore.QModelIndex): index of the shape clicked
        """
        shape = self.shapes[index.row()]
        if type(shape)==Grid:
            self.dialog = GridDialog(shape=shape, parent=self)
            self.dialog.applySig.connect(self.applyGridChanges)
            self.dialog.finished.connect(self.applyGridChanges)
        elif type(shape)==QtWidgets.QGraphicsRectItem:
            self.dialog = RectDialog(shape=shape, parent=self)
            self.dialog.applySig.connect(self.applyRectChanges)
            self.dialog.finished.connect(self.applyRectChanges)
        elif type(shape)==QtGui.QGraphicsLineItem:
            self.dialog = LineDialog(shape=shape, parent=self)
            self.dialog.applySig.connect(self.applyLineChanges)
            self.dialog.finished.connect(self.applyLineChanges)
        elif type(shape)==QtGui.QGraphicsEllipseItem:
            self.dialog = CircDialog(shape=shape, parent=self)
            self.dialog.applySig.connect(self.applyCircChanges)
            self.dialog.finished.connect(self.applyCircChanges)



    def applyGridChanges(self, *args):
        """
        Apply changes to a Grid when closing dialog or changing values
        in dialog.

        If the dialog is rejected, the shape will be returned to its
        initial state.
        """
        try:
            code = args[0]
        except IndexError:
            code = None

        if code == 1 or code == None:
            xPos, yPos, xSize, ySize, rows, columns, color, result = \
                self.dialog.getValues()
        elif code == 0:
            xPos, yPos, xSize, ySize, rows, columns, color, result = \
                self.dialog.initialValues
        else:
            raise('Result code not recognised')

        self.dialog.shape.nRows = rows
        self.dialog.shape.nColumns = columns
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        self.dialog.shape.setPen(pen)
        self.dialog.shape.setRect(QtCore.QRectF(xPos,yPos, xSize, ySize))
        self.dialog.shape.update()

    def applyRectChanges(self, *args):
        """
        Apply changes to a rect when closing dialog or changing values
        in dialog.

        If the dialog is rejected, the shape will be returned to its
        initial state.
        """
        try:
            code = args[0]
        except IndexError:
            code = None

        if code == None or code == 1:
            x, y, xSize, ySize, color, result = self.dialog.getValues()
        elif code == 0:
            x, y, xSize, ySize, color, result = self.dialog.initialValues
        else:
            raise('Result code not recognised')

        self.dialog.shape.setRect(x, y, xSize, ySize)
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        self.dialog.shape.setPen(pen)

    def applyLineChanges(self, *args):
        """
        Apply changes to a line when closing dialog or changing values
        in dialog.

        If the dialog is rejected, the shape will be returned to its
        initial state.
        """
        try:
            code = args[0]
        except IndexError:
            code = None

        if code == None or code == 1:
            x1, y1, x2, y2, color, result = self.dialog.getValues()
        elif code == 0:
            x1, y1, x2, y2, color, result = self.dialog.initialValues

        self.dialog.shape.setLine(x1, y1, x2, y2)
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        self.dialog.shape.setPen(pen)

    def applyCircChanges(self, *args):
        """
        Apply changes to a circle when closing dialog or changing values
        in dialog.

        If the dialog is rejected, the shape will be returned to its
        initial state.
        """
        try:
            code = args[0]
        except IndexError:
            code = None
        if code == None or code == 1:
            xPos, yPos, r, color, result = self.dialog.getValues()
        elif code == 0:
            xPos, yPos, r, color, result = self.dialog.initialValues

        self.dialog.shape.setRect(xPos-r, yPos-r, 2*r, 2*r)
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        self.dialog.shape.setPen(pen)

    def applyElipseChanges(self, *args):
        """
        Apply changes to an elipse when closing dialog or changing values
        in dialog.

        If the dialog is rejected, the shape will be returned to its
        initial state.
        """
        try:
            code = args[0]
        except IndexError:
            code = None
        if code == None or code == 1:
            xPos, yPos, rx, ry, color, result = self.dialog.getValues()
        elif code == 0:
            xPos, yPos, rx, ry, color, result = self.dialog.initialValues

        self.dialog.shape.setRect(xPos-rx, yPos-ry, 2*rx, 2*ry)
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        self.dialog.shape.setPen(pen)

# Rectangle drawing methods
#############################

    def addRect(self, x, y, xSize, ySize, color):
        """
        Used by MagicPlot API to draw rect from command line
        """
        self.shapes.append(QtGui.QGraphicsRectItem(
        				QtCore.QRectF(x,y,xSize,ySize)))
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        self.shapes[-1].setPen(pen)
        self.plotView.addItem(self.shapes[-1])
        return self.shapes[-1]

    def drawRect(self):
        self.dialog = RectDialog(parent=self)
        self.scene.sigMouseClicked.connect(self.mouseClicked_rect1)
        self.dialog.accepted.connect(self.drawRectFromValues)
        self.dialog.rejected.connect(self.cancelDrawRect)

    def cancelDrawRect(self):
        try:
            self.scene.sigMouseClicked.disconnect(self.mouseClicked_rect1)
        except TypeError:
            pass

    def drawRectFromValues(self):
        x, y, xSize, ySize, color, accepted = self.dialog.getValues()
        self.shapes.append(QtGui.QGraphicsRectItem(
                QtCore.QRectF(x,y,xSize,ySize)))
        self.shapes[-1].setPen(self.pen)
        self.plotView.addItem(self.shapes[-1])

    def updateRect(self, x, y, xSize, ySize):
        self.shapes[-1].setRect(QtCore.QRectF(x, y, xSize, ySize))
        self.dialog.setValuesFromShape()

    def mouseMoved_rect(self, pos):
        '''
        method attached to pyqtgraph image widget which gets the mouse position
        If the mouse is in the image, print both the mouse position and
        pixel value to the gui
        '''

        imgPos = self.viewBox.mapSceneToView(pos)
        scene = self.scene
        # Only update when mouse is in image
        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < scene.height()
                and pos.x() < scene.width()):


            self.mousePos = (imgPos.x(), imgPos.y())

            xSize = self.mousePos[0] - self.rectStartPos[0]
            ySize = self.mousePos[1] - self.rectStartPos[1]

            # xSize = pos.x() - self.rectStartPos[0]
            # ySize = pos.y() - self.rectStartPos[1]
            self.updateRect(
                    self.rectStartPos[0], self.rectStartPos[1],
                    xSize, ySize)

    def mouseClicked_rect1(self, event):
        logging.debug("Mouse clicked 1!")
        self.dialog.accepted.disconnect(self.drawRectFromValues)
        self.dialog.accepted.connect(self.applyRectChanges)
        self.dialog.applySig.connect(self.applyRectChanges)
        pos = event.scenePos()
        scene = self.scene
        imgPos = self.viewBox.mapSceneToView(pos)
        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < scene.height()
                and pos.x() < scene.width()):

            self.rectStartPos = (imgPos.x(), imgPos.y())
            self.shapes.append(QtGui.QGraphicsRectItem(
                    QtCore.QRectF(imgPos.x(),imgPos.y(),0,0)))
            pen = QtGui.QPen(QtGui.QBrush(self.dialog.color), QPEN_WIDTH)
            self.shapes[-1].setPen(pen)
            self.shapes[-1].setZValue(100)
            #self.shapes[-1].setBrush(QtGui.QBrush(QtCore.Qt.red))


            self.plotView.addItem(self.shapes[-1])
            self.dialog.setShape(self.shapes[-1])
            self.updateRect(imgPos.x(), imgPos.y(), 0,0)
            #self.updateRect(pos.x(), pos.y(), 0 ,0)
            self.scene.sigMouseMoved.connect(
                    self.mouseMoved_rect)
            self.scene.sigMouseClicked.disconnect(
                    self.mouseClicked_rect1)
            self.scene.sigMouseClicked.connect(
                    self.mouseClicked_rect2)

    def mouseClicked_rect2(self, event):
        logging.debug("Mouse clicked 2")
        pos = event.pos()
        scene = self.scene
        #imgPos = self.plotItem.mapFromScene(pos)
        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < scene.height()
                and pos.x() < scene.width()):

            self.scene.sigMouseMoved.disconnect(
                    self.mouseMoved_rect)
            self.scene.sigMouseClicked.disconnect(
                    self.mouseClicked_rect2)

            self.shapes.updateView()

# Line drawing methods
#############################

    def addLine(self, x1, y1, x2, y2, color):
        """
        Used by MagicPlot API to draw line from command line
        """
        self.shapes.append(QtGui.QGraphicsLineItem(x1,y1,x2,y2))
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        self.shapes[-1].setPen(pen)
        self.plotView.addItem(self.shapes[-1])
        return self.shapes[-1]

    def drawLine(self):
        self.dialog = LineDialog(parent=self)
        self.scene.sigMouseClicked.connect(self.mouseClicked_line1)
        self.dialog.accepted.connect(self.drawLineFromValues)
        self.dialog.rejected.connect(self.cancelDrawLine)

    def cancelDrawLine(self):
        try:
            self.scene.sigMouseClicked.disconnect(self.mouseClicked_line1)
        except TypeError:
            pass

    def drawLineFromValues(self):
        x1, y1, x2, y2, color, accepted = self.dialog.getValues()
        self.shapes.append(QtGui.QGraphicsLineItem(x1,y1,x2,y2))
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        self.shapes[-1].setPen(pen)
        self.plotView.addItem(self.shapes[-1])

    def updateLine(self, x1, x2, y1, y2):
        self.shapes[-1].setLine(QtCore.QLineF(x1, x2, y1, y2))
        self.dialog.setValuesFromShape()

    def mouseMoved_line(self, pos):
        '''
        method attached to pyqtgraph image widget which gets the mouse position
        If the mouse is in the image, print both the mouse position and
        pixel value to the gui
        '''

        imgPos = self.viewBox.mapSceneToView(pos)

        # Only update when mouse is in image
        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < self.scene.height()
                and pos.x() < self.scene.width()):

            self.mousePos = (imgPos.x(), imgPos.y())

            xSize = self.mousePos[0] - self.lineStartPos[0]
            ySize = self.mousePos[1] - self.lineStartPos[1]
            self.updateLine(
                    self.lineStartPos[0], self.lineStartPos[1],
                    self.mousePos[0], self.mousePos[1])

    def mouseClicked_line1(self, event):
        logging.debug("Line Mouse clicked 1")
        self.dialog.accepted.disconnect(self.drawLineFromValues)
        self.dialog.accepted.connect(self.applyLineChanges)
        self.dialog.applySig.connect(self.applyLineChanges)
        pos = event.scenePos()
        imgPos = self.viewBox.mapSceneToView(pos)
        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < self.scene.height()
                and pos.x() < self.scene.width()):

            self.lineStartPos = (imgPos.x(), imgPos.y())

            self.shapes.append(
                    QtGui.QGraphicsLineItem(QtCore.QLineF(
                            imgPos.x(),imgPos.y(),imgPos.x(),imgPos.y())))
            self.plotView.addItem(self.shapes[-1])
            pen = QtGui.QPen(QtGui.QBrush(self.dialog.color), QPEN_WIDTH)
            self.shapes[-1].setPen(pen)
            self.dialog.setShape(self.shapes[-1])

            #self.shapes[-1].setZValue(100)

            #imgPos = self.plotItem.mapFromScene(pos)

            self.updateLine(imgPos.x(), imgPos.y(), imgPos.x(),imgPos.y())
            self.scene.sigMouseMoved.connect(
                    self.mouseMoved_line)
            self.scene.sigMouseClicked.disconnect(
                    self.mouseClicked_line1)
            self.scene.sigMouseClicked.connect(
                    self.mouseClicked_line2)

    def mouseClicked_line2(self, event):
        logging.debug("Line Mouse clicked 2")
        pos = event.scenePos()
        imgPos = self.viewBox.mapSceneToView(pos)
        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < self.scene.height()
                and pos.x() < self.scene.width()):
            self.scene.sigMouseMoved.disconnect(
                    self.mouseMoved_line)
            self.scene.sigMouseClicked.disconnect(
                    self.mouseClicked_line2)

            self.shapes.updateView()

########## Grid Drawing ##############

    def addGrid(self, x, y, xSize, ySize, rows, cols, color):
        """
        Used by MagicPlot API to draw grid from command line
        """
        grid = Grid(QtCore.QRectF(
                x,y,xSize,ySize), rows, cols)
        self.shapes.append(grid)
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        self.shapes[-1].setPen(pen)
        self.plotView.addItem(grid)
        return grid

    # if size is 0, draw grid, else add grid
    def drawGrid(self):
        self.dialog = GridDialog(parent=self)
        self.scene.sigMouseClicked.connect(self.mouseClicked_grid1)
        self.dialog.accepted.connect(self.drawGridFromValues)
        self.dialog.rejected.connect(self.cancelDrawGrid)

    def cancelDrawGrid(self):
        try:
            self.scene.sigMouseClicked.disconnect(self.mouseClicked_grid1)
        except TypeError:
            pass

    def drawGridFromValues(self):
        xPos, yPos, xSize, ySize, rows, cols, color, result = \
            self.dialog.getValues()
        grid = Grid(QtCore.QRectF(
                xPos,yPos,xSize,ySize),rows,cols)
        self.shapes.append(grid)
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        self.shapes[-1].setPen(pen)
        self.plotView.addItem(grid)

    def updateGrid(self, x, y, xSize, ySize):
        self.shapes[-1].setRect(QtCore.QRectF(x, y, xSize, ySize))
        self.dialog.setValuesFromShape()

    def mouseMoved_grid(self, pos):
        imgPos = self.viewBox.mapSceneToView(pos)
        scene = self.scene
        # Only update when mouse is in image
        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < scene.height()
                and pos.x() < scene.width()):


            self.mousePos = (imgPos.x(), imgPos.y())

            xSize = self.mousePos[0] - self.gridStartPos[0]
            ySize = self.mousePos[1] - self.gridStartPos[1]

            # xSize = pos.x() - self.rectStartPos[0]
            # ySize = pos.y() - self.rectStartPos[1]
            self.updateGrid(
                    self.gridStartPos[0], self.gridStartPos[1],
                    xSize, ySize)

    def mouseClicked_grid1(self, event):
        logging.debug("Grid Mouse clicked 1")
        self.dialog.accepted.disconnect(self.drawGridFromValues)
        self.dialog.accepted.connect(self.applyGridChanges)
        self.dialog.applySig.connect(self.applyGridChanges)
        pos = event.scenePos()
        scene = self.scene
        imgPos = self.viewBox.mapSceneToView(pos)
        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < scene.height()
                and pos.x() < scene.width()):

            self.gridStartPos = (imgPos.x(), imgPos.y())
            rows, cols = self.dialog.getRowsCols()
            pen = QtGui.QPen(QtGui.QBrush(self.dialog.color), QPEN_WIDTH)
            grid =Grid(QtCore.QRectF(imgPos.x(), imgPos.y(), 0, 0), rows, cols)
            grid.setPen(pen)
            self.dialog.setShape(grid)
            self.shapes.append(grid)
            self.plotView.addItem(grid)
            # grid = Grid(QtCore.QRectF(
                # imgPos.x(),imgPos.y(),0,0),rows,cols)
            # self.shapes.append(grid)
            # self.plotView.addItem(grid)
            # self.shapes[-1].setPen(QtGui.QPen(QtCore.Qt.red))
            # self.shapes[-1].setZValue(100)
            #self.shapes[-1].setBrush(QtGui.QBrush(QtCore.Qt.red))
            self.updateGrid(imgPos.x(), imgPos.y(), 0,0)
            # gridShapes = self.grid.getShapes()
            # for i in gridShapes:
            #     self.plotView.addItem(i)
            #     i.setPen(QtGui.QPen(QtCore.Qt.red))
            # self.grid.setPen(QtGui.QPen(self.color))
            self.scene.sigMouseMoved.connect(
                    self.mouseMoved_grid)
            self.scene.sigMouseClicked.disconnect(
                    self.mouseClicked_grid1)
            self.scene.sigMouseClicked.connect(
                    self.mouseClicked_grid2)

    def mouseClicked_grid2(self, event):
        logging.debug("Grid Mouse clicked 2")
        pos = event.pos()
        scene = self.scene
        #imgPos = self.plotItem.mapFromScene(pos)
        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < scene.height()
                and pos.x() < scene.width()):

            self.scene.sigMouseMoved.disconnect(
                    self.mouseMoved_grid)
            self.scene.sigMouseClicked.disconnect(
                    self.mouseClicked_grid2)

            self.shapes.updateView()

########## Circle Drawing ###########

    def addCirc(self, x, y, r, color):
        circ = QtGui.QGraphicsEllipseItem(
                        QtCore.QRectF(x-r, y-r, 2*r, 2*r))
        self.shapes.append(circ)
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        circ.setPen(pen)
        self.plotView.addItem(circ)
        return circ

    def drawCirc(self):
        self.dialog = CircDialog(parent=self)
        self.scene.sigMouseClicked.connect(self.mouseClicked_circ1)
        self.dialog.accepted.connect(self.drawCircFromValues)
        self.dialog.rejected.connect(self.cancelDrawCirc)

    def cancelDrawCirc(self):
        try:
            self.scene.sigMouseClicked.disconnect(self.mouseClicked_circ1)
        except TypeError:
            pass

    def drawCircFromValues(self):
        x, y, r, color, accepted = self.dialog.getValues()
        self.shapes.append(QtGui.QGraphicsEllipseItem(
                        QtCore.QRectF(x-r,y-r,2*r,2*r)))
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        self.shapes[-1].setPen(pen)
        self.plotView.addItem(self.shapes[-1])

    def updateCirc(self, x, y, r):
        self.shapes[-1].setRect(QtCore.QRectF(x-r, y-r, 2*r, 2*r))
        self.dialog.setValuesFromShape()

    def mouseMoved_circ(self, pos):
        imgPos = self.viewBox.mapSceneToView(pos)
        scene = self.scene
        # Only update when mouse is in image
        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < scene.height()
                and pos.x() < scene.width()):

            self.mousePos = (imgPos.x(), imgPos.y())

            r = numpy.sqrt((self.mousePos[0]-self.circCenter[0])**2 +
                        (self.mousePos[1]-self.circCenter[1])**2)

            self.updateCirc(self.circCenter[0], self.circCenter[1], r)

    def mouseClicked_circ1(self, event):
        self.dialog.accepted.disconnect(self.drawCircFromValues)
        self.dialog.accepted.connect(self.applyCircChanges)
        self.dialog.applySig.connect(self.applyCircChanges)
        pos = event.scenePos()
        scene = self.scene
        imgPos = self.viewBox.mapSceneToView(pos)

        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < scene.height()
                and pos.x() < scene.width()):

            self.circCenter = (imgPos.x(), imgPos.y())
            self.shapes.append(QtGui.QGraphicsEllipseItem(
                    QtCore.QRectF(imgPos.x(), imgPos.y(), 0, 0)))
            pen = QtGui.QPen(QtGui.QBrush(self.dialog.color), QPEN_WIDTH)
            self.shapes[-1].setPen(pen)
            self.shapes[-1].setZValue(100)

            self.plotView.addItem(self.shapes[-1])
            self.dialog.setShape(self.shapes[-1])

            self.scene.sigMouseMoved.connect(
                    self.mouseMoved_circ)
            self.scene.sigMouseClicked.disconnect(
                    self.mouseClicked_circ1)
            self.scene.sigMouseClicked.connect(
                    self.mouseClicked_circ2)

    def mouseClicked_circ2(self, event):
        pos = event.pos()
        scene = self.scene

        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < scene.height()
                and pos.x() < scene.width()):

            self.scene.sigMouseMoved.disconnect(
                    self.mouseMoved_circ)
            self.scene.sigMouseClicked.disconnect(
                    self.mouseClicked_circ2)

            self.shapes.updateView()

########## Elipse Drawing ###########

    def addElipse(self, x, y, rx, ry, color):
        elipse = QtGui.QGraphicsEllipseItem(
                        QtCore.QRectF(x-rx, y-ry, 2*rx, 2*ry))
        self.shapes.append(elipse)
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        elipse.setPen(pen)
        self.plotView.addItem(elipse)
        return elipse

    def drawElipse(self):
        self.dialog = ElipseDialog(parent=self)
        self.scene.sigMouseClicked.connect(self.mouseClicked_elipse1)
        self.dialog.accepted.connect(self.drawElipseFromValues)
        self.dialog.rejected.connect(self.cancelDrawElipse)

    def cancelDrawElipse(self):
        try:
            self.scene.sigMouseClicked.disconnect(self.mouseClicked_elipse1)
        except TypeError:
            pass

    def drawElipseFromValues(self):
        x, y, rx, ry, color, accepted = self.dialog.getValues()
        self.shapes.append(QtGui.QGraphicsEllipseItem(
                        QtCore.QRectF(x-rx, y-ry, 2*rx, 2*ry)))
        pen = QtGui.QPen(QtGui.QBrush(color), QPEN_WIDTH)
        self.shapes[-1].setPen(pen)
        self.plotView.addItem(self.shapes[-1])

    def updateElipse(self, x, y, rx, ry):
        self.shapes[-1].setRect(QtCore.QRectF(x-rx, y-ry, 2*rx, 2*ry))
        self.dialog.setValuesFromShape()

    def mouseMoved_elipse(self, pos):
        imgPos = self.viewBox.mapSceneToView(pos)
        scene = self.scene
        # Only update when mouse is in image
        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < scene.height()
                and pos.x() < scene.width()):

            self.mousePos = (imgPos.x(), imgPos.y())

#            r = numpy.sqrt((self.mousePos[0]-self.elipseCenter[0])**2 +
#                        (self.mousePos[1]-self.elipseCenter[1])**2)
            rx = self.mousePos[0] - self.elipseCenter[0]
            ry = self.mousePos[1] - self.elipseCenter[1]

            self.updateElipse(self.elipseCenter[0], self.elipseCenter[1], rx, ry)

    def mouseClicked_elipse1(self, event):
        self.dialog.accepted.disconnect(self.drawElipseFromValues)
        self.dialog.accepted.connect(self.applyElipseChanges)
        self.dialog.applySig.connect(self.applyElipseChanges)
        pos = event.scenePos()
        scene = self.scene
        imgPos = self.viewBox.mapSceneToView(pos)

        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < scene.height()
                and pos.x() < scene.width()):

            self.elipseCenter = (imgPos.x(), imgPos.y())
            self.shapes.append(QtGui.QGraphicsEllipseItem(
                    QtCore.QRectF(imgPos.x(), imgPos.y(), 0, 0)))
            pen = QtGui.QPen(QtGui.QBrush(self.dialog.color), QPEN_WIDTH)
            self.shapes[-1].setPen(pen)
            self.shapes[-1].setZValue(100)

            self.plotView.addItem(self.shapes[-1])
            self.dialog.setShape(self.shapes[-1])

            self.scene.sigMouseMoved.connect(
                    self.mouseMoved_elipse)
            self.scene.sigMouseClicked.disconnect(
                    self.mouseClicked_elipse1)
            self.scene.sigMouseClicked.connect(
                    self.mouseClicked_elipse2)

    def mouseClicked_elipse2(self, event):
        pos = event.pos()
        scene = self.scene

        if      (pos.y() > 0 and pos.x() > 0
                and pos.y() < scene.height()
                and pos.x() < scene.width()):

            self.scene.sigMouseMoved.disconnect(
                    self.mouseMoved_elipse)
            self.scene.sigMouseClicked.disconnect(
                    self.mouseClicked_elipse2)

            self.shapes.updateView()

########### ROI tools #################################

    def setROI(self, checked):
        if checked:
            shape = self.dialog.shape
            if type(shape) in [QtGui.QGraphicsRectItem, Grid]:
                x, y, xSize, ySize  = self.dialog.getValues()[:4]
                try:
                    if type(self.roi) is RectROI:
                        self.roi.setPos([x,y])
                        self.roi.setSize([xSize,ySize])
                    else:
                        self.plotView.removeItem(self.roi)
                        self.roi = RectROI([x,y],[xSize,ySize], removable=True)
                        self.plotView.addItem(self.roi)
                except AttributeError:
                    self.roi = RectROI([x, y], [xSize, ySize], removable=True)
                    self.plotView.addItem(self.roi)
            elif type(shape) is QtGui.QGraphicsEllipseItem:
                x, y, r = self.dialog.getValues()[:3]
                cent_x, cent_y, diam = x-r, y-r, 2*r
                try:
                    if type(self.roi) is CircleROI:
                        self.roi.setPos([cent_x,cent_y])
                        self.roi.setSize([diam,diam])
                    else:
                        self.plotView.removeItem(self.roi)
                        self.roi = CircleROI([cent_x,cent_y],[diam,diam], removable=True)
                        self.plotView.addItem(self.roi)
                except AttributeError:
                    self.roi = CircleROI([cent_x,cent_y],[diam,diam], removable=True)
                    self.plotView.addItem(self.roi)
            elif type(shape) is QtGui.QGraphicsLineItem:
                x1, y1, x2, y2 = self.dialog.getValues()[:4]
                try:
                    if type(self.roi) is LineSegmentROI:
                        self.roi.setPos([(x1,y1),(x2,y2)])
                    else:
                        self.plotView.removeItem(self.roi)
                        self.roi = LineSegmentROI([(x1,y1),(x2,y2)], removable=True)
                        self.plotView.addItem(self.roi)
                except AttributeError:
                    self.roi = LineSegmentROI([(x1,y1),(x2,y2)], removable=True)
                    self.plotView.addItem(self.roi)
            self.roi.sigRemoveRequested.connect(self.removeRoi)
            # If there's a MagicPlotImageItem in the list, connect the region changed 
            # signal to the updateWindows function of that plotItem
            for i in self.plotItems:
                if type(i) is magicplot.MagicPlotImageItem:
                    self.roiPlotItem = i
                    self.plotRoiButton.setEnabled(True)
                    self.plotRoiButton.clicked.connect(self.plotROIHandler)
                    self.roi.sigRegionChanged.connect(i.updateWindows)
        else:
            try:
                self.plotView.removeItem(self.roi)
                self.removeRoi()
            except AttributeError as e:
                raise

    def plotROIHandler(self):
        self.roiPlotItem.plotROI(self.roi)

    def removeRoi(self):
        self.plotView.removeItem(self.roi)
        self.roi = None
        self.plotRoiButton.setEnabled(False)
        self.plotRoiButton.clicked.disconnect(self.plotROIHandler)
        if self.roiPlotItem:
            self.roiPlotItem = None

class ShapeDialog(QtWidgets.QDialog):
    """
    The base class of all shape dialogs.
    """

    # signal to emit to apply changes
    applySig = QtCore.pyqtSignal()

    def __init__(self, shape=None, parent=None, modal=False):
        super(ShapeDialog, self).__init__(parent)

        self.layout = QtWidgets.QGridLayout(self)
        self.colorButton = QtGui.QPushButton("Color")
        self.roiButton = QtWidgets.QCheckBox("Set RoI")
        # if the plot isn't an image plot, don't allow ROIs
        if type(self.parent().plotItems[0]) is magicplot.MagicPlotImageItem:
            self.roiButton.toggled.connect(self.parent().setROI)
        else:
            self.roiButton.setEnabled(False)
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self)
        self.buttons.addButton(self.roiButton, 3)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.colorButton.clicked.connect(self.getColor)
        self.setupUi()
        self.setShape(shape)

        if modal:
            self.exec_()
        else:
            self.show()

        # Put the dialog over the shape drawer to avoid it obscuring the plot
        shapeDrawerPosX = parent.window().pos().x() + parent.pos().x()
        shapeDrawerPosY = parent.window().pos().y() + parent.pos().y()
        self.move(shapeDrawerPosX, shapeDrawerPosY)

    def getColor(self):
        newColor = QtWidgets.QColorDialog().getColor(initial=self.color)
        if newColor.isValid():
            self.color = newColor
        # apply colour to shape
        self.applySig.emit()

    def apply(self):
        self.applySig.emit()

    def setShape(self, shape):
        if shape != None:
            self.shape = shape
            self.setValuesFromShape()
            self.color = self.shape.pen().color()
            self.setUpdateBoxes()
            self.initialValues = self.getValues()
        else:
            self.color = QtGui.QColor("red") # default
            logging.info('No shape')

    def setDefaultRange(self, spinboxes):
        doubleSpinBoxMin = -100000.0 # don't like setting spinbox limits like this
        doubleSpinBoxMax = 100000.0
        for i in spinboxes:
            i.setRange(doubleSpinBoxMin, doubleSpinBoxMax)

    def setValuesFromShape(self):
        pass

    def setupUi(self):
        pass

    def setUpdateBoxes(self):
        pass

class RectDialog(ShapeDialog):

    def __init__(self, shape=None, parent=None, modal=False):
        super(RectDialog, self).__init__(shape=shape, parent=parent,
                                        modal=modal)
        self.setWindowTitle("Draw Rectangle")

    def setupUi(self):
        self.posLabel = QtWidgets.QLabel("Pos (x,y)")
        self.xPosBox = QtWidgets.QDoubleSpinBox()
        self.yPosBox = QtWidgets.QDoubleSpinBox()
        self.sizeLabel = QtWidgets.QLabel("Size (width, height)")
        self.xSizeBox = QtWidgets.QDoubleSpinBox()
        self.ySizeBox = QtWidgets.QDoubleSpinBox()


        self.layout.addWidget(self.posLabel)
        self.layout.addWidget(self.xPosBox)
        self.layout.addWidget(self.yPosBox)
        self.layout.addWidget(self.sizeLabel)
        self.layout.addWidget(self.xSizeBox)
        self.layout.addWidget(self.ySizeBox)

        self.layout.addWidget(self.colorButton)
        self.layout.addWidget(self.buttons)
        self.setDefaultRange([self.xPosBox, self.yPosBox, self.xSizeBox,
                              self.ySizeBox])
        self.setLayout(self.layout)


    def setValuesFromShape(self):
        try:
            rect = self.shape.rect().normalized()
            x, y = rect.x(), rect.y()
            sizeX, sizeY = rect.width(), rect.height()
            self.xPosBox.setValue(x)
            self.yPosBox.setValue(y)
            self.xSizeBox.setValue(sizeX)
            self.ySizeBox.setValue(sizeY)
        except AttributeError as e:
            logging.info("no shape")

    def getValues(self):
        return (self.xPosBox.value(),
                self.yPosBox.value(),
                self.xSizeBox.value(),
                self.ySizeBox.value(),
                self.color,
                self.result())

    def setUpdateBoxes(self):
        # set boxes to update shape
        if self.shape != None:
            self.xPosBox.valueChanged.connect(self.apply)
            self.yPosBox.valueChanged.connect(self.apply)
            self.xSizeBox.valueChanged.connect(self.apply)
            self.ySizeBox.valueChanged.connect(self.apply)
        else:
            logging.info("No shape!")

class LineDialog(ShapeDialog):

    def __init__(self, shape=None, parent=None, modal=False):
        super(LineDialog, self).__init__(shape=shape, parent=parent,
                                         modal=modal)

        self.setWindowTitle("Draw Line")

    def setupUi(self):
        self.startLabel = QtWidgets.QLabel("Start Point")
        self.x1Box = QtWidgets.QDoubleSpinBox()
        self.y1Box = QtWidgets.QDoubleSpinBox()
        self.endLabel = QtWidgets.QLabel("End Point")
        self.x2Box = QtWidgets.QDoubleSpinBox()
        self.y2Box = QtWidgets.QDoubleSpinBox()
        self.layout.addWidget(self.startLabel)
        self.layout.addWidget(self.x1Box)
        self.layout.addWidget(self.y1Box)
        self.layout.addWidget(self.endLabel)
        self.layout.addWidget(self.x2Box)
        self.layout.addWidget(self.y2Box)

        self.layout.addWidget(self.colorButton)
        self.layout.addWidget(self.buttons)
        self.setDefaultRange([self.x1Box, self.y1Box, self.x2Box, self.y2Box])
        self.setLayout(self.layout)

    def setValuesFromShape(self):
        try:
            line = self.shape.line()
            self.x1Box.setValue(line.x1())
            self.y1Box.setValue(line.y1())
            self.x2Box.setValue(line.x2())
            self.y2Box.setValue(line.y2())
        except AttributeError:
            logging.info("No Shape")

    def getValues(self):
        return (self.x1Box.value(),
                self.y1Box.value(),
                self.x2Box.value(),
                self.y2Box.value(),
                self.color,
                self.result())

    def setUpdateBoxes(self):
        if self.shape != None:
            self.x1Box.valueChanged.connect(self.apply)
            self.y1Box.valueChanged.connect(self.apply)
            self.x2Box.valueChanged.connect(self.apply)
            self.y2Box.valueChanged.connect(self.apply)
        else:
            logging.info("No shape!")

class GridDialog(ShapeDialog):

    def __init__(self, shape=None, parent=None, modal=False):
        super(GridDialog, self).__init__(shape=shape, parent=parent,
                                         modal=modal)

        self.setWindowTitle("Draw Grid")

    def setupUi(self):
        self.posLabel = QtWidgets.QLabel("Pos (x,y)")
        self.xPosBox = QtWidgets.QDoubleSpinBox()
        self.yPosBox = QtWidgets.QDoubleSpinBox()
        self.sizeLabel = QtWidgets.QLabel("Size (width, height)")
        self.xSizeBox = QtWidgets.QDoubleSpinBox()
        self.ySizeBox = QtWidgets.QDoubleSpinBox()
        self.layout.addWidget(self.posLabel)
        self.layout.addWidget(self.xPosBox)
        self.layout.addWidget(self.yPosBox)
        self.layout.addWidget(self.sizeLabel)
        self.layout.addWidget(self.xSizeBox)
        self.layout.addWidget(self.ySizeBox)

        self.rowsLabel = QtWidgets.QLabel("# Rows")
        self.rowsBox = QtWidgets.QSpinBox()
        self.columnsLabel = QtWidgets.QLabel("# Columns")
        self.columnsBox = QtWidgets.QSpinBox()
        self.layout.addWidget(self.rowsLabel)
        self.layout.addWidget(self.rowsBox)
        self.layout.addWidget(self.columnsLabel)
        self.layout.addWidget(self.columnsBox)

        self.layout.addWidget(self.colorButton)
        self.layout.addWidget(self.buttons)
        self.setDefaultRange([self.xPosBox, self.yPosBox, self.xSizeBox,
                              self.ySizeBox])
        self.rowsBox.setRange(2,1000)
        self.columnsBox.setRange(2,1000)
        self.setLayout(self.layout)

    def getValues(self):
        return (self.xPosBox.value(),
                self.yPosBox.value(),
                self.xSizeBox.value(),
                self.ySizeBox.value(),
                self.rowsBox.value(),
                self.columnsBox.value(),
                self.color,
                self.result()
                )

    def getRowsCols(self):
        return self.rowsBox.value(), self.columnsBox.value()

    def setValuesFromShape(self):
        try:
            rect = self.shape.outRect.rect().normalized()
            self.xPosBox.setValue(rect.x())
            self.yPosBox.setValue(rect.y())
            self.xSizeBox.setValue(rect.width())
            self.ySizeBox.setValue(rect.height())
            self.rowsBox.setValue(self.shape.nRows)
            self.columnsBox.setValue(self.shape.nColumns)
        except AttributeError:
            logging.info('No shape')

    def setUpdateBoxes(self):
        if self.shape != None:
            self.xPosBox.valueChanged.connect(self.apply)
            self.yPosBox.valueChanged.connect(self.apply)
            self.xSizeBox.valueChanged.connect(self.apply)
            self.ySizeBox.valueChanged.connect(self.apply)
            self.rowsBox.valueChanged.connect(self.apply)
            self.columnsBox.valueChanged.connect(self.apply)
        else:
            logging.info("No shape!")

class CircDialog(ShapeDialog):

    def __init__(self, shape=None, parent=None, modal=False):
        super(CircDialog, self).__init__(shape=shape, parent=parent,
                                        modal = modal)

        self.setWindowTitle("Draw Circle")

    def setupUi(self):
        self.posLabel = QtWidgets.QLabel("Pos (x,y)")
        self.xPosBox = QtWidgets.QDoubleSpinBox()
        self.yPosBox = QtWidgets.QDoubleSpinBox()
        self.radiusLabel = QtWidgets.QLabel("Radius")
        self.radiusBox = QtWidgets.QDoubleSpinBox()

        self.layout.addWidget(self.posLabel)
        self.layout.addWidget(self.xPosBox)
        self.layout.addWidget(self.yPosBox)
        self.layout.addWidget(self.radiusLabel)
        self.layout.addWidget(self.radiusBox)

        self.layout.addWidget(self.colorButton)
        self.layout.addWidget(self.buttons)
        self.setDefaultRange([self.xPosBox, self.yPosBox, self.radiusBox])
        self.setLayout(self.layout)

    def setValuesFromShape(self):
        try:
            circ = self.shape.rect()
            r = circ.width()/2 # Better way of doing this?
            x, y = circ.x()+r, circ.y()+r
            self.xPosBox.setValue(x)
            self.yPosBox.setValue(y)
            self.radiusBox.setValue(r)
        except AttributeError:
            logging.info('No shape')

    def getValues(self):
        return (self.xPosBox.value(),
                self.yPosBox.value(),
                self.radiusBox.value(),
                self.color,
                self.result())

    def setUpdateBoxes(self):
        if self.shape != None:
            self.xPosBox.valueChanged.connect(self.apply)
            self.yPosBox.valueChanged.connect(self.apply)
            self.radiusBox.valueChanged.connect(self.apply)
        else:
            logging.info("No shape!")

class ElipseDialog(ShapeDialog):

    def __init__(self, shape=None, parent=None, modal=False):
        super(ElipseDialog, self).__init__(shape=shape, parent=parent,
                                        modal=modal)

        self.setWindowTitle("Draw Elipse")

    def setupUi(self):
        self.posLabel = QtGui.QLabel("Pos (x,y)")
        self.xPosBox = QtGui.QDoubleSpinBox()
        self.yPosBox = QtGui.QDoubleSpinBox()
        self.radiusLabel = QtGui.QLabel("Radius (x,y)")
        self.radiusXBox = QtGui.QDoubleSpinBox()
        self.radiusYBox = QtGui.QDoubleSpinBox()

        self.layout.addWidget(self.posLabel)
        self.layout.addWidget(self.xPosBox)
        self.layout.addWidget(self.yPosBox)
        self.layout.addWidget(self.radiusLabel)
        self.layout.addWidget(self.radiusXBox)
        self.layout.addWidget(self.radiusYBox)

        self.layout.addWidget(self.colorButton)
        self.layout.addWidget(self.buttons)
        self.setDefaultRange(
                [self.xPosBox, self.yPosBox, self.radiusXBox, self.radiusYBox])
        self.setLayout(self.layout)

    def setValuesFromShape(self):
        try:
            elipse = self.shape.rect()
            rx = elipse.width()/2. # Better way of doing this?
            ry = elipse.height()/2.
            x, y = elipse.x()+rx, elipse.y()+ry
            self.xPosBox.setValue(x)
            self.yPosBox.setValue(y)
            self.radiusXBox.setValue(rx)
            self.radiusYBox.setValue(ry)
        except AttributeError:
            logging.info('No shape')

    def getValues(self):
        return (self.xPosBox.value(),
                self.yPosBox.value(),
                self.radiusXBox.value(),
                self.radiusYBox.value(),
                self.color,
                self.result())

    def setUpdateBoxes(self):
        if self.shape != None:
            self.xPosBox.valueChanged.connect(self.apply)
            self.yPosBox.valueChanged.connect(self.apply)
            self.radiusXBox.valueChanged.connect(self.apply)
            self.radiusYBox.valueChanged.connect(self.apply)
        else:
            logging.info("No shape!")

class ShapeList(QtGui.QListView):
    """
    QListView to view shapes drawn on MagicPlot, reimplemented keyPressEvent
    to handle delete key (n.b. this breaks the arrow keys)
    """
    delKeySig = QtCore.pyqtSignal(object)

    def __init__(self, parent):
        super(ShapeList, self).__init__(parent)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete:
            self.delKeySig.emit(self.currentIndex())
        else:
            pass

    def contextMenuEvent(self, event):
        index = self.indexAt(event.pos())
        menu = QtGui.QMenu(self)
        delete = QtGui.QAction('Delete shape', self)
        delete.triggered.connect(
            lambda: self.parent().shapes.removeShape(index))
        menu.addAction(delete)
        action = menu.exec_(event.globalPos())
