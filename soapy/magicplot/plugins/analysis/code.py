class Plugin(AnalysisPlugin):

    def __init__(self):
        AnalysisPlugin.__init__(self, params={'code':''},
            name='Code')

    def generateUi(self):
        """
        overwrite this method as it's different to all the others
        """
        self.layout = QtGui.QGridLayout()
        self.codeBox = QtGui.QTextEdit('Put your code here!')
        self.codeBox.textChanged.connect(self.setParams)
        self.outputBox = QtGui.QTextEdit()
        self.layout.addWidget(self.codeBox, 0,0)
        self.layout.addWidget(self.outputBox, 0, 1)
        self.setLayout(self.layout)

    def setParams(self):
        """
        overwrite this one as well
        """
        self.params['code'] = str(self.codeBox.toPlainText())

    def run(self):
        exec(self.params['code'])
        return output 