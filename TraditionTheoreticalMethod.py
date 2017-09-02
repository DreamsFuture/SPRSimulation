# -*- encoding: utf-8 -*-

import numpy
#from InherentAttribute import InherentAttribute

class TraditionTheoreticalMethod(object):
    def __init__(self,numberOfLayers,coupledPrismRefractiveIndex,coupledPrismThickness,highRefractiveIndexMedium,highRefractiveIndexMediumThickness,
                 lowerRefractiveIndexMedium, lowerRefractiveIndexMediumThickness,metalRefractiveIndex,metalThickness,wavelength,incidentAngle,
                 aqueousSolutionRefractiveIndex,aqueousSolutionThickness):
        self.numberOfLayers = numberOfLayers

        self.coupledPrismRefractiveIndex = coupledPrismRefractiveIndex
        self.coupledPrismThickness = coupledPrismThickness

        self.highRefractiveIndexMedium = highRefractiveIndexMedium
        self.highRefractiveIndexMediumThickness = highRefractiveIndexMediumThickness

        self.lowerRefractiveIndexMedium = lowerRefractiveIndexMedium
        self.lowerRefractiveIndexMediumThickness = lowerRefractiveIndexMediumThickness

        self.metalRefractiveIndex = metalRefractiveIndex
        self.metalThickness = metalThickness

        self.wavelength = wavelength
        self.incidentAngle = incidentAngle

        self.aqueousSolutionRefractiveIndex = aqueousSolutionRefractiveIndex
        self.aqueousSolutionThickness = aqueousSolutionThickness

        self.reflectedLight = [numberOfLayers]

    def ReflectedLightIntensity(self):
        R = numpy.abs(self.StructuralReflectionCoefficient(1,3))
        return R**2

    def StructuralReflectionCoefficient(self,i,N):
        #防止出现0作为指数底数
        if(i == N-1):
            return self.AdjacentReflectionCoefficient(i)
       




        traditionalModel = self.TraditionalModel(i,N)
        return traditionalModel


    def TraditionalModel(self,i=1,N=3):

        structuralReflectionCoefficient = (self.AdjacentReflectionCoefficient(i) + self.StructuralReflectionCoefficient(i + 1, N)
            * numpy.e ** ((self.TraditionalModelThickness(i + 1) * self.DetectionDepth(i + 1)) * 2j)) / \
            (1 + self.AdjacentReflectionCoefficient(i) * self.StructuralReflectionCoefficient(i + 1, N) * numpy.e ** (
            (2j) * self.TraditionalModelThickness(i + 1) * self.DetectionDepth(i + 1)))
        return structuralReflectionCoefficient



    # ri,i+1 = [(ni+1**2/kz,i+1)-(ni**2/kz,i)]/[(ni+1**2/kz,i+1)+(ni**2/kz,i)]
    def AdjacentReflectionCoefficient(self, i):

        refractiveIndex_0 = self.TraditionalModelRefractiveIndex(i)
        refractiveIndex_1 = self.TraditionalModelRefractiveIndex(i + 1)
        detectionDepth_0 = self.DetectionDepth(i)
        detectionDepth_1 = self.DetectionDepth(i + 1)
        adjacentReflectionCoefficient = (refractiveIndex_1**2/detectionDepth_1 - refractiveIndex_0**2/detectionDepth_0)\
               /(refractiveIndex_1**2/detectionDepth_1 + refractiveIndex_0**2 / detectionDepth_0)
        return adjacentReflectionCoefficient

    def DetectionDepth(self,i):

        detectionDepth = (((2*numpy.pi/self.wavelength * self.TraditionalModelRefractiveIndex(i))**2 - self.HorizontalWaveVector()**2)**(1/2))
        return detectionDepth

    def HorizontalWaveVector(self):
        horizontalWaveVector = 2 * numpy.pi / self.wavelength * self.coupledPrismRefractiveIndex * numpy.sin(self.incidentAngle)
        return horizontalWaveVector

    def TraditionalModelRefractiveIndex(self,i):
        if i == 1 :
            return self.coupledPrismRefractiveIndex
        elif(i == 2):
            return self.metalRefractiveIndex
        elif(i == 3):
            return self.aqueousSolutionRefractiveIndex
        else:
            return 1

    def TraditionalModelThickness(self,i):
        if(i == 1 ):
            return self.coupledPrismThickness
        elif(i == 2):
            return self.metalThickness
        elif(i == 3):
            return self.aqueousSolutionThickness
        else:
            return 1
