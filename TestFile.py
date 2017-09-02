import math
import matplotlib.pyplot as plot
import numpy as np
import cmath
z = 3 + 2j
x = cmath.phase(z)
print(x)
mm = z.imag+z.real
y = []
for i in range(-50,50,1):
    y.append(i**2)
x = np.arange(-50, 50,1)
intensity = np.array(y)

intensity = []
# intensity = np.array(intensity)
x = np.arange(50, 90, 1)
for i in range(50, 90, 1):
    theta = i
    intensity.append(i**2)
minVal = np.amin(intensity)
maxVal = np.amax(intensity)
intensity = np.array(intensity)
outufuncXArray = normalize_func(minVal, maxVal)(intensity)  # the result is a ufunc object
normalizedIntensity = outufuncXArray.astype(float)  # cast ufunc object ndarray to float ndarray

y = np.array(normalizedIntensity)
print(y)
plot.plot(x, y)
plot.ylabel("Intensity")
plot.xlabel("Angle(Degree)")
plot.title("SPR")
plot.show()


def normalize_func(minVal, maxVal, newMinValue=0, newMaxValue=1):
    def normalizeFunc(x):
        r = (x - minVal) * newMaxValue / (maxVal - minVal) + newMinValue
        return r

    return np.frompyfunc(normalizeFunc, 1, 1)

print("intensity:\n" )
print(intensity)
plot.plot(x, intensity)
plot.ylabel("Intensity")
plot.xlabel("Angle")
plot.title("SPR")
plot.show()


def a(**x):print(x)

a(x = 3,y = 8,z = 3)

print("-----------------------")
a(x = 3,u = 8)


print("-----------------------")

a = 1 + 3j
d = abs(a)
dd = math.e**1j
print("|dd|:"+str(abs(dd)))

print("dd"+str(dd))
#m = cmath.exp(2a)


b = cmath.polar(a)
c = b[0]**2+b[1]**2
#c = math.pow(b,2)
print(b)
print(c)
print(d)
print(d**2)
print(abs(m))