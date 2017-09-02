#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import mpmath
import matplotlib.pyplot as plot
index = []
thickness = []

numberOfLayers = 3
coupledPrismRefractiveIndex = 1.515
coupledPrismThickness = 200
index.append(coupledPrismRefractiveIndex)
thickness.append(coupledPrismThickness)

highRefractiveIndexMedium = 2.0373
highRefractiveIndexMediumThickness = 150
index.append(highRefractiveIndexMedium)
thickness.append(highRefractiveIndexMediumThickness)

lowerRefractiveIndexMedium = 1.457
lowerRefractiveIndexMediumThickness = 420
index.append(lowerRefractiveIndexMedium)
thickness.append(lowerRefractiveIndexMediumThickness)

AuMetalRefractiveIndex = 0.16172 + 3.21182j
AgMetalRefractiveIndex = 0.05625 + 4.27603j
metalThickness = 50
AuMetalThickness = 50
AgMetalThickness = 50
index.append(AuMetalRefractiveIndex)
index.append(AgMetalRefractiveIndex)
thickness.append(AuMetalThickness)
thickness.append(AgMetalThickness)


aqueousSolutionRefractiveIndex = 1.3321
aqueousSolutionThickness = 4 * 10 ** 2
index.append(aqueousSolutionRefractiveIndex)
thickness.append(aqueousSolutionThickness)

wavelength = 632.8
aqueousSolutionRefractiveIndex_1 = 1.33211
incidentAngle = 61.55
i=5

for n in range(50,90,1):
    angle = n /180*np.pi
    if(thickness[i]*index[i]*np.sin(angle) > wavelength):
        print("index:"+str(index[i])+"thickness:"+str(thickness[i])+"indent angle:"+str(angle)+"wavelength:"+str(wavelength))

#测试后发现如果d取值太大比如会不符合 n*d*sin(theta)<lamda

"""


for i in range(9, -1, -1):
    if i == 9:
        print("****** i == 9 :"+str(i))
    else:
        print("i = " +str(i))

reflectedLight = []
for i in range(0,10+1):
    reflectedLight.append(i)
    print(reflectedLight[-1])

"""
"""
print(mpmath.fabs(-1))
intensity = []
x = []
for i in np.arange(50.0, 90.0, 0.1):
    incidentAngle = i
    result = mpmath.sin(i)
    if mpmath.isnan(result):
        continue
    intensity.append(result)
    x.append(i)
"""
"""
x = [1, 2, 3, 3,6,8]
y = [4, 17, 24, 37, 43, 53]
z = [5,9,16,24,33,55]




def fun(*x):
    if len(x) == 0:
        return None
    else:
        lenght = len(x)
        array = []
        for i in x:
            j = np.array(i)
            array.append(j)


            print(type(j))
            print(j)
        for i in range(len(array)):
            plot.plot(array[0],array[i])
        plot.ylabel("Intensity")
        plot.xlabel("Angle(Degree)")
        plot.title("SPR")
        plot.show()


#不在一个class中，必须先定义，后面才能使用
fun()
fun(x, y,z)

"""


"""
y = np.array(intensity)
x = np.array(x)
print(y)
print(x)
plot.plot(x, y)
plot.ylabel("Intensity")
plot.xlabel("Angle(Degree)")
plot.title("SPR")
plot.show()

print((0.00017494456400 - 0.014841550958) ** (1 / 2))

print(0 ** (1 / 2))

from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer

data = np.arange(6).reshape(2, 3)
print(data)


# normalize ndarray
def normalize_func(minVal, maxVal):
    def normalizeFunc(x):
        r = (x - minVal) / (maxVal - minVal)
        return r

    return np.frompyfunc(normalizeFunc, 1, 1)


minVal = np.amin(data)
maxVal = np.amax(data)
outufuncXArray = normalize_func(minVal, maxVal)(data)  # the result is a ufunc object
dataXArray = outufuncXArray.astype(float)  # cast ufunc object ndarray to float ndarray

print(dataXArray)
"""

"""
fig = figure()
x=linspace(0,1,10)
y1=x
y2=x**2

subplot(211)
plot.plot(, label="test1")
plot.plot(, label="test2")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)
subplot(223)
plot.plot(, label="test1")
plot.plot(, label="test2")
plot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plot.show()
"""

