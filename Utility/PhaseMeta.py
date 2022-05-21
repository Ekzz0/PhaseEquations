"""
Модуль реализации метакласса, необходимого для работы представления.

pyqtWrapperType - метакласс общий для оконных компонентов Qt.
ABCMeta - метакласс для реализации абстрактных суперклассов.

FilterDMeta - метакласс для представления.
"""

from PyQt5.QtCore import QObject
from abc import ABCMeta

pyqtWrapperType = type(QObject)


class PhaseMeta(pyqtWrapperType, ABCMeta):
    pass
