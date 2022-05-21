from View.PhaseView import PhaseView
import pandas as pd
import numpy as np


class PhaseController():
    def __init__(self, inModel):
        """
        Конструктор принимает ссылку на модель.
        Конструктор создаёт и отображает представление.
        """
        self.mModel = inModel
        self.mView = PhaseView(self, self.mModel)

        self.mView.show()

    def setTemperature(self):
        Temperature = self.mView.Temperature.value()
        self.mModel.currentT = Temperature

    def setPmin(self):
        Pmin = self.mView.Pmin.value()
        self.mModel.minimalPressure = Pmin

    def setPmax(self):
        Pmax = self.mView.Pmax.value()
        self.mModel.maximumPressure = Pmax

    def setPstep(self):
        Pstep = self.mView.Pstep.value()
        self.mModel.Pstep = Pstep


    def extractData(self, pth):
        '''
        Функция извлекающая данные из файла
        '''
        data = pd.read_excel(pth)
        dict = {"CH4": 0,
                "C2H6": 1,
                "C3H8": 2,
                "C4H10": 3,
                "C5H12": 4,
                "C6H14": 5,
                "C7H16": 6,
                "С10Н22": 7,
                "N2": 8,
                "CO2": 9,
                "H2S": 10}

        ind = 0
        for name in data['name'].values[:]:
            self.mModel.massFractions[dict[name]] = data['z'].values[ind]
            ind += 1

    def start(self):
        '''
        Функция начинающая рассчет
        '''
        X, Y_srk, Y_brus = self.mModel.graph(self.mModel.massFractions, self.mModel.acentricFactor, self.mModel.criticalPressure,
                                               self.mModel.criticalTemperature, self.mModel.Pressure, self.mModel.currentT, self.mModel.c_ij)
        print(X)
        print(Y_srk)
        print(Y_brus)
        self.mView.widget.clearCanvas()
        self.mView.widget.updateCanvas(X, Y_srk)
        self.mView.widget.updateCanvas(X, Y_brus)
        # self.mView.widget.fig.savefig("FilteredSignal")
        '''
        NoiseLen= len(self.mModel.NoiseSignal)
        XN = np.arange(0, 15*NoiseLen, 15)
        if self.mView.midArifmRunning.isChecked():
            self.mModel.ClearSignal = self.mModel.runningMidArifm(self.mModel.Num, self.mModel.NoiseSignal)
            ClearLen = len(self.mModel.ClearSignal)
            XC = np.linspace(0, 15*NoiseLen, ClearLen)
        elif self.mView.midArifm.isChecked():
            self.mModel.ClearSignal = self.mModel.midArifm(self.mModel.Num, self.mModel.NoiseSignal)
            ClearLen = len(self.mModel.ClearSignal)
            XC = np.linspace(0, 15 * NoiseLen, ClearLen)
        self.mView.widget.clearCanvas()
        self.mView.widget.updateCanvas(XN, self.mModel.NoiseSignal)
        self.mView.widget.updateCanvas(XC, self.mModel.ClearSignal)
        self.mView.widget.fig.savefig("FilteredSignal")
        with open("FilteredSignal.txt", 'w') as file:
            for value in self.mModel.ClearSignal:
                file.write(str(value) + '\n')
        '''

