#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/3 10:27
# @Author  : Colin
# @Site    : https://github.com/DreamsFuture
# @File    : PredecessorsModel.py
# @Software: PyCharm Community Edition


# -*- encoding: utf-8 -*-
#金属部分为Au，多层结构方法效果

from sympy import *
import mpmath

class PredecessorsModel(object):
    def __init__(self, numberOfLayers, coupledPrismRefractiveIndex, coupledPrismThickness, highRefractiveIndexMedium,
                 highRefractiveIndexMediumThickness,mediRefractiveIndex,mediRefractiveIndexMediumThickness,
                 lowerRefractiveIndexMedium, lowerRefractiveIndexMediumThickness, metalRefractiveIndex, metalThickness,
                 AuMetalRefractiveIndex,AgMetalRefractiveIndex,AuMetalThickness,AgMetalThickness,wavelength, incidentAngle,
                 aqueousSolutionRefractiveIndex, aqueousSolutionThickness):
        self.numberOfLayers = numberOfLayers

        self.coupledPrismRefractiveIndex = coupledPrismRefractiveIndex
        self.coupledPrismThickness = coupledPrismThickness

        self.highRefractiveIndexMedium = highRefractiveIndexMedium
        self.highRefractiveIndexMediumThickness = highRefractiveIndexMediumThickness

        self.mediRefractiveIndex = mediRefractiveIndex
        self.mediRefractiveIndexMediumThickness = mediRefractiveIndexMediumThickness

        self.lowerRefractiveIndexMedium = lowerRefractiveIndexMedium
        self.lowerRefractiveIndexMediumThickness = lowerRefractiveIndexMediumThickness

        self.metalRefractiveIndex = metalRefractiveIndex
        self.metalThickness = metalThickness
        self.AuMetalRefractiveIndex = AuMetalRefractiveIndex
        self.AgMetalRefractiveIndex = AgMetalRefractiveIndex
        self.AuMetalThickness = AuMetalThickness
        self.AgMetalThickness = AgMetalThickness

        self.wavelength = wavelength
        self.incidentAngle = incidentAngle

        self.aqueousSolutionRefractiveIndex = aqueousSolutionRefractiveIndex
        self.aqueousSolutionThickness = aqueousSolutionThickness

        self.reflectedLight = []


    def ReflectedLightIntensity(self):

        for i in range(self.numberOfLayers+1):
            try:
                self.reflectedLight.append(0)
            except IndexError:
                print("IndexError is i =" + str(i))
        self.StructuralReflectionCoefficient(self.numberOfLayers)
        R = mpmath.fabs(self.reflectedLight[1])

        #for i in range(len(self.reflectedLight)):
            #self.reflectedLight[i] = mpmath.fabs(self.reflectedLight[i])**2
            #print("self.reflectedLight[%s] = %s"%(i,self.reflectedLight[i]))
        dep =  mpmath.fabs(self.DetectionDepth(self.numberOfLayers))
        return Float(R ** 2,15),dep

    def StructuralReflectionCoefficient(self, N):
        for i in range(N-1,-1,-1):
            if i == N - 1:
                self.reflectedLight[i] = self.AdjacentReflectionCoefficient(i)
            elif i != 0:
                self.reflectedLight[i] = self.NewModel(i)
            else:
                break
        # 防止出现0作为指数底数

    def NewModel(self, i):
        structuralReflectionCoefficient = (self.AdjacentReflectionCoefficient(i) + self.reflectedLight[i+1]*
            mpmath.e ** ( 2j * self.MediumThickness(i + 1) * self.DetectionDepth(i + 1) )) / \
                (1 + self.AdjacentReflectionCoefficient(i) * self.reflectedLight[i+1] * mpmath.e ** ( 2j *
                    self.MediumThickness(i + 1) * self.DetectionDepth(i + 1)))
        return structuralReflectionCoefficient

    # ri,i+1 = [(ni+1**2/kz,i+1)-(ni**2/kz,i)]/[(ni+1**2/kz,i+1)+(ni**2/kz,i)]
    def AdjacentReflectionCoefficient(self, i):
        adjacentReflectionCoefficient = 0
        refractiveIndex_0 = self.RefractiveIndex(i)
        refractiveIndex_1 = self.RefractiveIndex(i + 1)
        detectionDepth_0 = self.DetectionDepth(i)
        detectionDepth_1 = self.DetectionDepth(i + 1)
        #print(refractiveIndex_0,refractiveIndex_1,detectionDepth_0,detectionDepth_1)
        try:
            adjacentReflectionCoefficient = (refractiveIndex_1 ** 2 / detectionDepth_1 - refractiveIndex_0 ** 2 / detectionDepth_0) \
                                        / (refractiveIndex_1 ** 2 / detectionDepth_1 + refractiveIndex_0 ** 2 / detectionDepth_0)
        except ZeroDivisionError:
            print("adjacentReflectionCoefficient has some problem with ZeroDivisionError")
            print("self.DetectionDepth(i + 1) = " + str(detectionDepth_1))
            print("self.DetectionDepth(i) = " + str(detectionDepth_0))
            print("self.RefractiveIndex(i) = " + str(refractiveIndex_0))
            print("self.RefractiveIndex(i + 1) = " + str(refractiveIndex_1))
        return adjacentReflectionCoefficient

    def DetectionDepth(self, i):
        detectionDepth = 0
        try:
            detectionDepth = ((2 * mpmath.pi * self.RefractiveIndex(i)/ self.wavelength ) ** 2 - self.HorizontalWaveVector() ** 2) ** (1 / 2)
        except ZeroDivisionError:
            print("detectionDepth has some problem with ZeroDivisionError")
            print("i = "+str(i) + "self.MediumThickness(i) = " + str(self.MediumThickness(i)) )
            print("self.HorizontalWaveVector() = " + str(self.HorizontalWaveVector()))
        return detectionDepth

    def HorizontalWaveVector(self):
        horizontalWaveVector = 0
        try:
            horizontalWaveVector = 2 * mpmath.pi * self.coupledPrismRefractiveIndex * mpmath.sin(self.incidentAngle) / self.wavelength
        except ZeroDivisionError:
            print("horizontalWaveVector has some problem with ZeroDivisionError\n" )
            print("coupledPrismRefractiveIndex = :"+str(self.coupledPrismRefractiveIndex))
        finally:
            return horizontalWaveVector

    def RefractiveIndex(self,i):
        if (i == 1):
            #print("i/2 != 0  ,i = %s" % str(i))
            return self.coupledPrismRefractiveIndex
        elif(i == self.numberOfLayers):
            return self.aqueousSolutionRefractiveIndex
        elif (i == self.numberOfLayers - 1):
            return self.metalRefractiveIndex
        elif (i%2 == 0):
            #print("i/2 != 0  ,i = %s"%str(i))
            return self.lowerRefractiveIndexMedium
        elif(i%2== 1):
            #print("i/2 == 0  ,i = %s" % str(i))
            return self.highRefractiveIndexMedium
        else:
            print("the index i is "+str(i))
            return 0
        #除法中有/和%，所以之前的问题就是在这里，好了一点
    def MediumThickness(self,i):
        if (i == 1):
            return self.coupledPrismThickness
        elif(i == self.numberOfLayers):
            return self.aqueousSolutionThickness
        elif (i == self.numberOfLayers - 1 ):
            return self.metalThickness
        elif (i%2 == 0):
            return self.lowerRefractiveIndexMediumThickness
        elif(i%2 == 1):
            return self.highRefractiveIndexMediumThickness
        else:
            print("The index i is "+str(i))
            return 0