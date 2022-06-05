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

    def updateCanvas(self, X, Y, legend):
        self.ax.plot(X, Y, label=legend)
        self.ax.set_xlabel('Pressure [MPa]')
        self.ax.set_ylabel('Percentage of liquid phase [%]')
        self.ax.legend(loc='lower right')
        self.ax.grid(True, color='black')
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
        self.widget.setGeometry(29, 159, 831, 381)
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

        self.setFixedSize(self.size())
    def browseFiles(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '*.xlsx')
        self.filename.setText(fname[0])
        self.mController.extractData(fname[0])


    def modelIsChanged(self):
        """
        Метод вызывается при изменении модели.
        """
        pass


