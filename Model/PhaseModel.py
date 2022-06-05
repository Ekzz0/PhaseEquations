import numpy as np
import math
from math import log as ln


class PhaseModel(object):
    """
    Класс PhaseModel представляет собой реализацию модели данных

    В модели хранятся массив значений снятых с датчика
    Модель представляет интерфейс, через который можно работать с хранимыми значениями

    Модель реализует методы регистрации, удаления и оповещения наблюдателей
    """

    def __init__(self):
        self._mZ = [0 for i in range(11)]   # Массив массовых долей
        self._mT = 1  # Текущее значение температуры
        self._mPmin = 1  # Начальное давление
        self._mP = self._mPmin  # Текущее значение давления
        self._mPmax = 1  # Конечное давление
        self._mPstep = 1  # Шаг изменения давления

        self._mObservers = []  # Массив наблюдателей

    # Массив коэффициентов парного взаимодействия
    c_ij = [[0.000, 0.005, 0.010, 0.010, 0.030, 0.030, 0.035, 0.045, 0.025, 0.105, 0.07],
        [0.005, 0.000, 0.005, 0.010, 0.010, 0.020, 0.020, 0.02, 0.010, 0.13, 0.085],
        [0.010, 0.005, 0.000, 0.000, 0.020, 0.005, 0.005, 0.005, 0.090, 0.125, 0.08],
        [0.010, 0.010, 0.000, 0.000, 0.005, 0.005, 0.005, 0.005, 0.095, 0.115, 0.075],
        [0.030, 0.010, 0.020, 0.005, 0.000, 0.000, 0.000, 0.000, 0.100, 0.115, 0.07],
        [0.030, 0.020, 0.005, 0.005, 0.000, 0.000, 0.000, 0.000, 0.110, 0.115, 0.07],
        [0.035, 0.020, 0.005, 0.005, 0.000, 0.000, 0.000, 0.000, 0.115, 0.115, 0.06],
        [0.045, 0.020, 0.005, 0.005, 0.000, 0.000, 0.000, 0.000, 0.125, 0.115, 0.055],
        [0.025, 0.010, 0.090, 0.095, 0.100, 0.110, 0.115, 0.125, 0, 000, 0.000, 0.13],
        [0.105, 0.13, 0.125, 0.115, 0.115, 0.115, 0.115, 0.115, 0.000, 0.000, 0.135],
        [0.07, 0.085, 0.08, 0.075, 0.07, 0.070, 0.060, 0.055, 0.130, 0.135, 0.000]]

    acentricFactor = [0.01142, 0.0995, 0.1521, 0.184, 0.201, 0.251, 0.349, 0.4884, 0.0372, 0.22394, 0.1005]

    criticalPressure = [4.5992, 4.8722, 4.2512, 3.629, 3.796, 3.370, 2.736, 2.103, 3.3958, 7.3773, 9.00]

    criticalTemperature = [190.56, 305.32, 369.89, 407.81, 425.13, 469.6, 540.13, 617.7, 126.19, 304.13, 373.10]

    # Убрать потом
    @property
    def Pressure(self):
        return self._mP

    @property
    def Pstep(self):
        return self._mPstep

    @property
    def massFractions(self):
        return self._mZ

    @property
    def currentT(self):
        return self._mT

    @property
    def minimalPressure(self):
        return self._mPmin

    @property
    def maximumPressure(self):
        return self._mPmax

    # Убрать потом
    @Pressure.setter
    def Pressure(self, value):
        self._mP = value
        self.notifyObservers()

    @Pstep.setter
    def Pstep(self, value):
        self._mPstep = value
        self.notifyObservers()

    @massFractions.setter
    def massFractions(self, value):
        self._mZ = value
        self.notifyObservers()

    @currentT.setter
    def currentT(self, value):
        self._mT = value + 273
        self.notifyObservers()

    @minimalPressure.setter
    def minimalPressure(self, value):
        self._mPmin = value
        self.notifyObservers()

    @maximumPressure.setter
    def maximumPressure(self, value):
        self._mPmax = value
        self.notifyObservers()

    def addObserver(self, inObserver):
        self._mObservers.append(inObserver)

    def removeObserver(self, inObserver):
        self._mObservers.remove(inObserver)

    def notifyObservers(self):
        for x in self._mObservers:
            x.modelIsChanged()

    # уравнение Рашфорда – Райса
    @staticmethod
    def f(x, z, K):
        res = 0
        for i in range(len(z)):
            res += z[i] * (K[i] - 1) / (1 + x * (K[i] - 1))
        return res


    # Отбрасывание комплексный корней
    @staticmethod
    def no_compl(array):
        ans = []
        for found_root in array:
            if found_root == found_root.real:
                ans.append(found_root.real)
        return ans

    # Метод биссекции
    def bisection(self, a, b, eps, z, K):
        iter_b = 0
        root = None
        while abs(self.f(b, z, K) - self.f(a, z, K)) > eps:
            mid = (a + b) / 2
            if self.f(mid, z, K) == 0 or abs(self.f(mid, z, K)) < eps:
                root = mid
                break
            elif self.f(a, z, K) * self.f(mid, z, K) < 0:
                b = mid
            else:
                a = mid
            iter_b += 1
        if root is None:
            print('Корень не найден')

        return root


    def brusilovski_method(self,massFractions, acentricFactor, criticalPressure, criticalTemperature, Pressure, currentT, c_ij):
        K = []
        x = []
        y = []
        Bi = []
        Ci = []
        Di = []
        alpha = []
        beta = []
        sigma = []
        delta = []
        a_m_y = 0
        b_m_y = 0
        c_m_y = 0
        d_m_y = 0
        a_m_x = 0
        b_m_x = 0
        c_m_x = 0
        d_m_x = 0
        k1 = []
        k2 = []
        k3 = []
        k4 = []
        F_v = []
        F_l = []
        K_0 = []
        K_1 = []
        check = []
        eps = 0.0001

        R = 8.314462  # Дж/(моль·K)  если Дж/(кг К), то R*1000/94.288
        N = len(massFractions)
        '''Определим параметры Z_c, Omega_c, Psi'''
        Z_c = [0.33294, 0.31274, 0.31508, 0.30663, 0.31232, 0.0, 0.0, 0.0, 0.34626, 0.31933, 0.30418]
        omega_c = [0.75630, 0.77698, 0.76974, 0.78017, 0.76921, 0.75001, 0.75001, 0.75001, 0.75001, 0.75282, 0.78524]
        psi = [0.37447, 0.49550, 0.53248, 0.63875, 0.57594, 0.0, 0.0, 0.0, 0.37182, 0.74212, 0.38203]

        # Заполнение недостающих элементов массивов Z_c и Psi
        num = 0
        for i in Z_c:
            if i == 0:
                Z_c[num] = 0.3357 - 0.0294 * acentricFactor[num]
            num += 1
        num = 0
        for i in psi:
            if i == 0:
                if acentricFactor[num] < 0.4489:
                    psi[num] = 1.050 + 0.105 * acentricFactor[num] + 0.482 * acentricFactor[num] ** 2
                else:
                    psi[num] = 0.429 + 1.004 * acentricFactor[num] + 1.561 * acentricFactor[num] ** 2
            num += 1
        del num

        '''Вычислим коэф-ы альфа, бета, сигма, дельта'''

        for i in range(N):
            alpha.append(omega_c[i] ** 3)
            beta.append(Z_c[i] + omega_c[i] - 1)
            sigma.append(-Z_c[i] + omega_c[i] * (0.5 + (omega_c[i] - 0.75) ** 0.5))
            delta.append(-Z_c[i] + omega_c[i] * (0.5 - (omega_c[i] - 0.75) ** 0.5))

        '''Далее вычислим коэффициенты a, b, c, d для каждого компонента связи'''
        a = []
        b = []
        c = []
        d = []
        for i in range(N):
            a_c = alpha[i] * (R ** 2) * (criticalTemperature[i] ** 2) / criticalPressure[i]
            al_T_w = (1 + psi[i] * (1 - (currentT / criticalTemperature[i]) ** 0.5)) ** 2
            a.append(a_c * al_T_w)
            b.append(beta[i] * R * criticalTemperature[i] / criticalPressure[i])
            c.append(sigma[i] * R * criticalTemperature[i] / criticalPressure[i])
            d.append(delta[i] * R * criticalTemperature[i] / criticalPressure[i])
        # print(a)
        # print(b)
        # print(c)
        # print(d)

        '''Расчитаем начальное приближение K'''
        for i in range(N):
            K.append(
                (criticalPressure[i] / Pressure) * math.exp(5.37 * (1 + acentricFactor[i]) * (1 - (criticalTemperature[i] / currentT))))  # Нач. знач. K по корреляции Вильсона
        # print(K)

        '''Проведем одну итерацию вручную'''
        # Решим уравнение Рашфорда – Райса
        V = self.bisection(1 / (1 - max(K)) + 0.0001, 1 / (1 - min(K)) - 0.0001, 0.0001, massFractions, K)
        # print("V: ", V)
        # Вычислим y_i и x_i и сразу же проверим условие их сумм = 1
        sum_x = 0
        sum_y = 0
        sum_z = 0
        for i in range(N):
            y.append(massFractions[i] * K[i] / (V * (K[i] - 1) + 1))
            x.append(y[i] / K[i])
            sum_x += x[i]
            sum_y += y[i]
            sum_z += massFractions[i]
        for i in range(N):
            if y[i] == 0:
                y[i] = y[i] + eps
            if x[i] == 0:
                x[i] = x[i] + eps
        # print(y)
        # print(x)
        # print(sum_x, sum_y)
        '''Вычислим все коэф-ы a_m_x, a_m_y, b_m_x, b_m_y, c_m_x, c_m_y, d_m_x, d_m_y'''
        for i in range(N):
            c_m_y += y[i] * c[i]
            c_m_x += x[i] * c[i]
            d_m_y += y[i] * d[i]
            d_m_x += x[i] * d[i]
            for j in range(N):
                a_m_y += (y[i] * y[j] * (1 - c_ij[i][j]) * math.sqrt(a[i] * a[j]))
                b_m_y += (y[i] * y[j] * (0.5 * (b[i] + b[j])))
                a_m_x += (x[i] * x[j] * (1 - c_ij[i][j]) * math.sqrt(a[i] * a[j]))
                b_m_x += (x[i] * x[j] * (0.5 * (b[i] + b[j])))

        # Обозначим Am_x, Bm_x, Cm_x, Dm_x, Bi, Ci, Di
        Am_y = a_m_y * Pressure / (R * R * currentT * currentT)
        Am_x = a_m_x * Pressure / (R * R * currentT * currentT)
        Bm_y = b_m_y * Pressure / (R * currentT)
        Bm_x = b_m_x * Pressure / (R * currentT)
        Cm_y = c_m_y * Pressure / (R * currentT)
        Cm_x = c_m_x * Pressure / (R * currentT)
        Dm_y = d_m_y * Pressure / (R * currentT)
        Dm_x = d_m_x * Pressure / (R * currentT)
        for i in range(N):
            Bi.append(b[i] * Pressure / (R * currentT))
            Ci.append(c[i] * Pressure / (R * currentT))
            Di.append(d[i] * Pressure / (R * currentT))

        # Ищем решение кубического уравнения относительно Z_v и Z_l
        pol_K_y = [1, 0, 0, 0]
        pol_K_y[1] = Cm_y + Dm_y - Bm_y - 1
        pol_K_y[2] = Am_y - Bm_y * Cm_y + Cm_y * Dm_y - Bm_y * Dm_y - Dm_y - Cm_y
        pol_K_y[3] = -(Bm_y * Cm_y * Dm_y + Cm_y * Dm_y + Am_y * Bm_y)
        z_fact_y = np.roots(pol_K_y)
        z_fact_y = self.no_compl(z_fact_y)
        z_v_max = round(max(z_fact_y), 6)  # Корень ур-я Z_v

        pol_K_x = [1, 0, 0, 0]
        pol_K_x[1] = Cm_x + Dm_x - Bm_x - 1
        pol_K_x[2] = Am_x - Bm_x * Cm_x + Cm_x * Dm_x - Bm_x * Dm_x - Dm_x - Cm_x
        pol_K_x[3] = -(Bm_x * Cm_x * Dm_x + Cm_x * Dm_x + Am_x * Bm_x)
        z_fact_x = np.roots(pol_K_x)
        z_fact_x = self.no_compl(z_fact_x)
        z_l_min = round(min(z_fact_x), 6)  # Корень ур-я Z_l

        K_0 = K.copy()  # Начальное значение, вычисленное по корреляции Вильсона

        for i in range(N):
            sum5 = 0
            for j in range(N):
                sum5 += y[j] * (1 - c_ij[i][j]) * math.sqrt(a[i] * a[j])
            # print(y[i]*Pressure )
            k1.append(ln(y[i] * Pressure))
            k2.append(ln(z_v_max - Bm_y))
            k3.append((Am_y / (Cm_y - Dm_y)) * (2 * sum5 / a_m_y - (c[i] - d[i]) / (c_m_y - d_m_y)) * ln(
                (z_v_max + Cm_y) / (z_v_max + Dm_y)))
            k4.append(
                Bi[i] / (z_v_max - Bm_y) - Am_y / (Cm_y - Dm_y) * (Ci[i] / (z_v_max + Cm_y) - Di[i] / (z_v_max + Dm_y)))
            F_v.append(math.exp(k1[i] - k2[i] - k3[i] + k4[i]))

            sum5 = 0
            for j in range(N):
                sum5 += x[j] * (1 - c_ij[i][j]) * math.sqrt(a[i] * a[j])
            k1[i] = (ln(x[i] * Pressure))
            k2[i] = (ln(z_l_min - Bm_x))
            k3[i] = ((Am_x / (Cm_x - Dm_x)) * (2 * sum5 / a_m_x - (c[i] - d[i]) / (c_m_x - d_m_x)) * ln(
                (z_l_min + Cm_x) / (z_l_min + Dm_x)))
            k4[i] = (Bi[i] / (z_l_min - Bm_x) - Am_x / (Cm_x - Dm_x) * (
                        Ci[i] / (z_l_min + Cm_x) - Di[i] / (z_l_min + Dm_x)))
            F_l.append(math.exp(k1[i] - k2[i] - k3[i] + k4[i]))

            K_1.append(K_0[i] * F_l[i] / F_v[i])
            check.append(abs(K_1[i] - K_0[i]) / abs(K_0[i]))

        '''На этом шаге мы получили значения K_1 и K_0, с которых мы далее начнем проверять условие сходимости '''
        # print(K_1)
        # print(K_0)

        iter = 1
        '''Непосредственно сам расчетный цикл'''

        while max(check) >= eps:  # пока максимальная ошибка > eps
            a_m_x = 0
            a_m_y = 0
            b_m_x = 0
            b_m_y = 0
            c_m_x = 0
            c_m_y = 0
            d_m_x = 0
            d_m_y = 0
            # Решим уравнение Рашфорда – Райса
            V = self.bisection(1 / (1 - max(K_1)) + 0.0001, 1 / (1 - min(K_1)) - 0.0001, 0.0001, massFractions, K_1)
            # print("a: ", 1 / (1 - max(K_1)) + 0.0001)
            # print("b: ", 1 / (1 - min(K_1)) + 0.0001)
            # print("V: ", V)
            # Вычислим y_i и x_i и сразу же проверим условие их сумм = 1
            sum_x = 0
            sum_y = 0
            sum_z = 0
            for i in range(N):
                y[i] = (massFractions[i] * K_1[i] / (V * (K_1[i] - 1) + 1))
                x[i] = (y[i] / K_1[i])
                sum_x += x[i]
                sum_y += y[i]
                sum_z += massFractions[i]
            for i in range(N):
                if y[i] == 0:
                    y[i] = y[i] + eps
                if x[i] == 0:
                    x[i] = x[i] + eps
            # print(y)
            # print(x)
            # print(sum_x, sum_y)
            # Вычислим все коэф-ы a_m_x, a_m_y, b_m_x, b_m_y, c_m_x, c_m_y, d_m_x, d_m_y

            for i in range(N):
                c_m_y += y[i] * c[i]
                c_m_x += x[i] * c[i]
                d_m_y += y[i] * d[i]
                d_m_x += x[i] * d[i]
                for j in range(N):
                    a_m_y += (y[i] * y[j] * (1 - c_ij[i][j]) * math.sqrt(a[i] * a[j]))
                    b_m_y += (y[i] * y[j] * (0.5 * (b[i] + b[j])))
                    a_m_x += (x[i] * x[j] * (1 - c_ij[i][j]) * math.sqrt(a[i] * a[j]))
                    b_m_x += (x[i] * x[j] * (0.5 * (b[i] + b[j])))

            # Обозначим Am_x, Bm_x, Cm_x, Dm_x, Bi, Ci, Di
            Am_y = a_m_y * Pressure / (R * R * currentT * currentT)
            Am_x = a_m_x * Pressure / (R * R * currentT * currentT)
            Bm_y = b_m_y * Pressure / (R * currentT)
            Bm_x = b_m_x * Pressure / (R * currentT)
            Cm_y = c_m_y * Pressure / (R * currentT)
            Cm_x = c_m_x * Pressure / (R * currentT)
            Dm_y = d_m_y * Pressure / (R * currentT)
            Dm_x = d_m_x * Pressure / (R * currentT)
            for i in range(N):
                Bi[i] = (b[i] * Pressure / (R * currentT))
                Ci[i] = (c[i] * Pressure / (R * currentT))
                Di[i] = (d[i] * Pressure / (R * currentT))

            # Ищем решение кубического уравнения относительно Z_v и Z_l
            pol_K_y = [1, 0, 0, 0]
            pol_K_y[1] = Cm_y + Dm_y - Bm_y - 1
            pol_K_y[2] = Am_y - Bm_y * Cm_y + Cm_y * Dm_y - Bm_y * Dm_y - Dm_y - Cm_y
            pol_K_y[3] = -(Bm_y * Cm_y * Dm_y + Cm_y * Dm_y + Am_y * Bm_y)
            z_fact_y = np.roots(pol_K_y)
            z_fact_y = self.no_compl(z_fact_y)
            z_v_max = round(max(z_fact_y), 6)  # Корень ур-я Z_v

            pol_K_x = [1, 0, 0, 0]
            pol_K_x[1] = Cm_x + Dm_x - Bm_x - 1
            pol_K_x[2] = Am_x - Bm_x * Cm_x + Cm_x * Dm_x - Bm_x * Dm_x - Dm_x - Cm_x
            pol_K_x[3] = -(Bm_x * Cm_x * Dm_x + Cm_x * Dm_x + Am_x * Bm_x)
            z_fact_x = np.roots(pol_K_x)
            z_fact_x = self.no_compl(z_fact_x)
            z_l_min = round(min(z_fact_x), 6)  # Корень ур-я Z_l

            K_0 = K_1.copy()  # Далее буду считать следующее значение K -> нужно зафиксировать предыдущее

            for i in range(N):
                sum5 = 0
                for j in range(N):
                    sum5 += y[j] * (1 - c_ij[i][j]) * math.sqrt(a[i] * a[j])
                k1[i] = (ln(y[i] * Pressure))
                k2[i] = (ln(z_v_max - Bm_y))
                k3[i] = ((Am_y / (Cm_y - Dm_y)) * (2 * sum5 / a_m_y - (c[i] - d[i]) / (c_m_y - d_m_y)) * ln(
                    (z_v_max + Cm_y) / (z_v_max + Dm_y)))
                k4[i] = (Bi[i] / (z_v_max - Bm_y) - Am_y / (Cm_y - Dm_y) * (
                            Ci[i] / (z_v_max + Cm_y) - Di[i] / (z_v_max + Dm_y)))
                F_v[i] = (math.exp(k1[i] - k2[i] - k3[i] + k4[i]))

                sum5 = 0
                for j in range(N):
                    sum5 += x[j] * (1 - c_ij[i][j]) * math.sqrt(a[i] * a[j])
                k1[i] = (ln(x[i] * Pressure))
                k2[i] = (ln(z_l_min - Bm_x))
                k3[i] = ((Am_x / (Cm_x - Dm_x)) * (2 * sum5 / a_m_x - (c[i] - d[i]) / (c_m_x - d_m_x)) * ln(
                    (z_l_min + Cm_x) / (z_l_min + Dm_x)))
                k4[i] = (Bi[i] / (z_l_min - Bm_x) - Am_x / (Cm_x - Dm_x) * (
                        Ci[i] / (z_l_min + Cm_x) - Di[i] / (z_l_min + Dm_x)))
                F_l[i] = (math.exp(k1[i] - k2[i] - k3[i] + k4[i]))

                K_1[i] = (K_0[i] * F_l[i] / F_v[i])
                check[i] = abs(K_1[i] - K_0[i]) / abs(K_0[i])
                #check[i] = abs(F_l[i] / F_v[i] - 1)
            iter += 1

        '''Проверка некоторых выражений'''
        # print("Метод Брусиловского:")
        # print("Pressure: ", Pressure)
        # print("Iter:", iter)
        # print("Summ_z: ", sum_z, "Summ_y: ", sum_y, "Summ_x: ", sum_x)
        # root_r = 0
        # for i in range(N):
        #     root_r += massFractions[i] * (K_1[i] - 1) / (1 + V * (K_1[i] - 1))
        # print("Подставим найденное V в уравнение Рашфорда – Райса:", root_r)
        L = (sum_z - sum_y * V) / sum_x
        L = 1 - V
        # print("L: ", L, " -> следовательно жидкая фаза существует!")

        return L, K_1, iter


    def SRK_method(self, massFractions, acentricFactor, criticalPressure, criticalTemperature, Pressure, currentT, c_ij):
        '''
        Объявим все нужные массивы и создадим их нужного размера
        '''
        N = len(massFractions)
        K = []; K_1 = []; K_0 = []
        alpha = []; m = []; y = []; x = []
        B_v = 0.0; A_v = 0.0; B_l = 0.0; A_l = 0.0
        F_v = []; F_l = []; check = []
        T_r = []; P_r = []
        A_p = []; B_p = []
        eps = 0.00001
        A_v = 0; B_v = 0; A_l = 0; B_l = 0
        k1 = []; k2 = []; k3 = []

        '''Вычислим значение коэф-ов m, alpha, K_0'''
        for i in range(N):
            T_r.append(currentT / criticalTemperature[i])
            P_r.append(Pressure / criticalPressure[i])
            m.append(0.48 + 1.574 * acentricFactor[i] - 0.176 * acentricFactor[i] ** 2)
            alpha.append((1 + m[i] * (1 - math.sqrt(T_r[i]))) ** 2)
            K.append((criticalPressure[i] / Pressure) * math.exp(5.37 * (1 + acentricFactor[i]) * (1 - (criticalTemperature[i] / currentT))))  # Начальное значение K по корреляции Вильсона

        '''Далее проведем одну итерацию вручную, чтобы получить 2 значения K1 и K0'''
        # Решим уравнение Рашфорда – Райса
        V = self.bisection(1 / (1 - max(K)) + 0.0001, 1 / (1 - min(K)) - 0.0001, 0.0001, massFractions, K)

        # Вычислим y_i и x_i и сразу же проверим условие их сумм = 1
        sum_x = 0; sum_y = 0; sum_z = 0
        for i in range(N):
            y.append(massFractions[i] * K[i] / (V * (K[i] - 1) + 1))
            x.append(y[i] / K[i])
            sum_x += x[i]; sum_y += y[i]; sum_z += massFractions[i]

        # Вычислим коэффициенты A_p, B_p и сразу же B_v и B_l
        for i in range(N):
            A_p.append(0.42747 * alpha[i] * P_r[i] / T_r[i] ** 2)
            B_p.append(0.08664 * P_r[i] / T_r[i])
            B_v += y[i] * B_p[i]
            B_l += x[i] * B_p[i]

        for i in range(N):
            for j in range(N):
                A_v += y[i] * y[j] * (1-c_ij[i][j]) * math.sqrt(A_p[i] * A_p[j])
                A_l += x[i] * x[j] * (1-c_ij[i][j]) * math.sqrt(A_p[i] * A_p[j])

        # Ищем решение кубического уравнения относительно Z_v и Z_l

        pol_K_y = [1, -1, 0, 0]
        pol_K_y[2] = A_v - B_v - B_v ** 2
        pol_K_y[3] = -A_v * B_v
        z_fact_y = np.roots(pol_K_y)
        z_fact_y = self.no_compl(z_fact_y)
        Z_v_max = round(max(z_fact_y), 6)  # Корень ур-я Z_v
        pol_K_x = [1, -1, 0, 0]
        pol_K_x[2] = A_l - B_l - B_l ** 2
        pol_K_x[3] = -A_l * B_l
        z_fact_x = np.roots(pol_K_x)
        z_fact_x = self.no_compl(z_fact_x)
        Z_l_min = round(min(z_fact_x), 6)  # Корень ур-я Z_l

        K_0 = K.copy()  # Начальное значение, вычисленное по корреляции Вильсона

        for i in range(N):
            k1.append(((Z_v_max - 1) * B_p[i] / B_v))
            k2.append((ln(Z_v_max - B_v)))
            k3.append(((A_v / B_v) * (2 * math.sqrt(A_p[i] / A_v) - B_p[i] / B_v) * ln((Z_v_max + B_v) / Z_v_max)))
            F_v.append(math.exp(k1[i] - k2[i] - k3[i]))

            k1[i] = ((Z_l_min - 1) * B_p[i] / B_l)
            k2[i] = (ln(Z_l_min - B_l))
            k3[i] = ((A_l / B_l) * (2 * math.sqrt(A_p[i] / A_l) - B_p[i] / B_l) * ln((Z_l_min + B_l) / Z_l_min))
            F_l.append(math.exp(k1[i] - k2[i] - k3[i]))

            K_1.append(F_l[i] / F_v[i])
            check.append(abs(K_1[i] - K_0[i]) / abs(K_0[i]))
        '''На этом шаге мы получили значения K_1 и K_0, с которых мы далее начнем проверять условие сходимости '''
        # print(K_0)
        # print(K_1)
        iter = 0
        '''Непосредственно сам расчетный цикл'''
        while max(check) >= eps:  # пока максимальная ошибка > eps
            B_v = 0; B_l = 0; A_v = 0; A_l = 0
            '''K_1 - текущее значение, K_0 - предыдущее значение'''
            # Решим уравнение Рашфорда – Райса
            V = self.bisection(1 / (1 - max(K_1)) + 0.00001, 1 / (1 - min(K_1)) - 0.00001, 0.0001, massFractions, K_1)

            # Вычислим y_i и x_i и сразу же проверим условие их сумм = 1
            sum_x = 0; sum_y = 0; sum_z = 0
            for i in range(N):
                y[i] = massFractions[i] * K_1[i] / (V * (K_1[i] - 1) + 1)
                x[i] = y[i] / K_1[i]  # massFractions[i]/(V * (K[i] - 1) + 1)
                sum_x += x[i]; sum_y += y[i]; sum_z += massFractions[i]
            # print("Суммы x, y", sum_x, sum_y)
            # Вычислим коэффициенты A_p, B_p и сразу же B_v и B_l
            for i in range(N):
                A_p[i] = 0.42747 * alpha[i] * P_r[i] / T_r[i] ** 2
                B_p[i] = 0.08664 * P_r[i] / T_r[i]
                B_v += y[i] * B_p[i]
                B_l += x[i] * B_p[i]

            for i in range(N):
                for j in range(N):
                    A_v += y[i] * y[j] * math.sqrt(A_p[i] * A_p[j])
                    A_l += x[i] * x[j] * math.sqrt(A_p[i] * A_p[j])

            # Ищем решение кубического уравнения относительно Z_v и Z_l
            pol_K_y = [1, -1, 0, 0]
            pol_K_y[2] = A_v - B_v - B_v ** 2
            pol_K_y[3] = -A_v * B_v
            z_fact_y = np.roots(pol_K_y)
            z_fact_y = self.no_compl(z_fact_y)
            Z_v_max = round(max(z_fact_y), 6)  # Корень ур-я Z_v

            pol_K_x = [1, -1, 0, 0]
            pol_K_x[2] = A_l - B_l - B_l ** 2
            pol_K_x[3] = -A_l * B_l
            z_fact_x = np.roots(pol_K_x)
            z_fact_x = self.no_compl(z_fact_x)
            Z_l_min = round(min(z_fact_x), 6)  # Корень ур-я Z_l

            # Тут я буду пересчитывать K -> мне нужно запомнить предыдущий шаг -> K_0_i = K_1_i
            K_0 = K_1.copy()
            for i in range(N):
                # Нахождение F_v
                k1[i] = ((Z_v_max - 1) * B_p[i] / B_v)
                k2[i] = (ln(Z_v_max - B_v))
                k3[i] = ((A_v / B_v) * (2 * math.sqrt(A_p[i] / A_v) - B_p[i] / B_v) * ln((Z_v_max + B_v) / Z_v_max))
                F_v[i] = math.exp(k1[i] - k2[i] - k3[i])
                # Нахождение F_l
                k1[i] = ((Z_l_min - 1) * B_p[i] / B_l)
                k2[i] = (ln(Z_l_min - B_l))
                k3[i] = ((A_l / B_l) * (2 * math.sqrt(A_p[i] / A_l) - B_p[i] / B_l) * ln((Z_l_min + B_l) / Z_l_min))
                F_l[i] = math.exp(k1[i] - k2[i] - k3[i])
                # Нахождение K
                K_1[i] = F_l[i] / F_v[i]
                check[i] = abs(K_1[i] - K_0[i]) / abs(K_0[i])  # Норма
            iter += 1

        '''Проверка некоторых выражений'''
        # print("Метод СРК:")
        # print("Pressure: ", Pressure)
        # print("Iter:", iter)
        # print("Summ_z: ", sum_z, "Summ_y: ", sum_y, "Summ_x: ", sum_x)
        # root_r = 0
        # for i in range(N):
        #     root_r += massFractions[i] * (K_1[i] - 1) / (1 + V * (K_1[i] - 1))
        # print("Подставим найденное V в уравнение Рашфорда – Райса:", root_r)
        L = (sum_z - sum_y * V) / sum_x
        L = 1 - V
        # print("L: ",L , " -> следовательно жидкая фаза существует!")


        return L, K_1


    def graph(self, massFractions, acentricFactor, criticalPressure, criticalTemperature, Pressure, currentT, c_ij):
        result_SRK = []
        result_brusilovski = []
        P_arr = []
        for P in np.arange(self.minimalPressure, self.maximumPressure + self.Pstep , self.Pstep):
            tmp = self.SRK_method(massFractions, acentricFactor, criticalPressure, criticalTemperature, P, currentT, c_ij)[0]
            # print("СРК","Pressure=", Pressure, "L =", tmp)
            tmp1 = self.brusilovski_method(massFractions, acentricFactor, criticalPressure, criticalTemperature, P, currentT, c_ij)[0]
            # print("Б","P=", P, "L =", tmp1)
            result_SRK.append(tmp * 100)
            result_brusilovski.append(tmp1 * 100)
            P_arr.append(P)
            del tmp
            del tmp1

        return P_arr, result_SRK, result_brusilovski









