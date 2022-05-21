import sys
from PyQt5.QtWidgets import QApplication

from Model.PhaseModel import PhaseModel
from Controller.PhaseController import PhaseController

def main():
    app = QApplication(sys.argv)

    # Создание модели
    model = PhaseModel()

    # Создание контроллера и передача ему ссылки на модель
    controller = PhaseController(model)

    app.exec()



if __name__ == '__main__':
    sys.exit(  main() )