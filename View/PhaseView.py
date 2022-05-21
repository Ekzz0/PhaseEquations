import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QWidget
from Utility.PhaseObserver import PhaseObserver
from Utility.PhaseMeta import PhaseMeta
from View.gui import Ui_MainWindow
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MyCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.fig, self.ax = plt.subplots()
        super(MyCanvas, self).__init__(self.fig)
        self.setParent(parent)

    def updateCanvas(self, X, Y):
        self.ax.plot(X,Y)
        self.draw()

    def clearCanvas(self):
        self.ax.cla()
        self.draw()


class PhaseView(QMainWindow, PhaseObserver, Ui_MainWindow, metaclass= PhaseMeta):
    """
    Класс отвечающий за визуальное представление PhaseModel.
    """
    def __init__(self, inController, inModel, parent = None):
        """
        Конструктор принимает ссылки на модель и контроллер.
        """
        super(QMainWindow, self).__init__(parent)
        self.mController = inController
        self.mModel = inModel

        # подключение визуального представления
        self.widget = MyCanvas(self, width=4, height=4, dpi=100)
        self.widget.setGeometry(457, 10, 421, 428)
        self.setupUi(self)

        # Представление регистрируется как наблюдатель
        self.mModel.addObserver(self)

        # Связка события и метода контроллера
        self.browse.clicked.connect(self.browseFiles)
        self.Temperature.valueChanged.connect(self.mController.setTemperature)
        self.Pmin.valueChanged.connect(self.mController.setPmin)
        self.Pmax.valueChanged.connect(self.mController.setPmax)
        self.Pstep.valueChanged.connect(self.mController.setPstep)
        self.start.clicked.connect(self.mController.start)

    def browseFiles(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '*.xlsx')
        self.filename.setText(fname[0])
        self.mController.extractData(fname[0])


    def modelIsChanged(self):
        """
        Метод вызывается при изменении модели.
        """
        pass


