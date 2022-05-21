import pandas as pd
from Model.PhaseModel import PhaseModel
import matplotlib.pyplot as plt



pth = r"D:\Программирование\ForPolitech\PhaseEquations\Test.xlsx"
model = PhaseModel()
data = pd.read_excel(pth)
dict ={"CH4":0,
    "C2H6":1,
    "C3H8":2,
    "C4H10":3,
    "C5H12":4,
    "C6H14":5,
    "C7H16":6,
    "С10Н22":7,
    "N2":8,
    "CO2":9,
    "H2S":10}

ind = 0
for name in data['name'].values[:]:
    model.massFractions[dict[name]] = data['z'].values[ind]
    ind+=1

model.Pressure = 3
model.currentT =10
model.maximumPressure =4
model.minimalPressure =3
model.Pstep = 0.1

X,Y1,Y2 = model.graph(model.massFractions, model.acentricFactor, model.criticalPressure,
                                               model.criticalTemperature, model.Pressure, model.currentT, model.c_ij)
fig = plt.figure()
plt.xlabel(u'Давление [МПа]', fontsize=12)
plt.ylabel(u'Процент жидкой фазы [%]', fontsize=12)
plt.title(u'Зависимость процента жидкой фазы от давления L(P)', fontsize=12)

plt.plot(X, Y1, label=u'СРК Метод')
plt.plot(X,Y2,  label = u'Метод Брусиловского')
fig.legend(loc='center right')
plt.savefig('L(P)')
plt.grid(True, color='black')
plt.show()

