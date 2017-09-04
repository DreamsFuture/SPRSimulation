# -*- encoding: utf-8 -*-

import numpy
import mpmath


# from InherentAttribute import InherentAttribute

class TraditionTheoreticalMethodNonIterate(object):
    def __init__(self, numberOfLayers, coupledPrismRefractiveIndex, coupledPrismThickness, highRefractiveIndexMedium,
                 highRefractiveIndexMediumThickness,
                 lowerRefractiveIndexMedium, lowerRefractiveIndexMediumThickness, metalRefractiveIndex, metalThickness,
                 AuMetalRefractiveIndex,AgMetalRefractiveIndex,wavelength, incidentAngle,
                 aqueousSolutionRefractiveIndex, aqueousSolutionThickness):
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

        self.reflectedLight = [numberOfLayers+1]
        for i in range(0,numberOfLayers):
            self.reflectedLight.append(0)

    def ReflectedLightIntensity(self):
        self.StructuralReflectionCoefficient(1, 3)
        R = mpmath.fabs(self.reflectedLight[1])
        dep = mpmath.fabs(self.DetectionDepth(self.numberOfLayers))
        return R ** 2,dep

    def StructuralReflectionCoefficient(self, i, N):
        for i in range(N-1,-1,-1):
            if i == N - 1:

                self.reflectedLight[i] = self.AdjacentReflectionCoefficient(i)
            elif(i != 0):
                self.reflectedLight[i] = self.TraditionalModel(i)
        # 防止出现0作为指数底数



    def TraditionalModel(self, i=1):

        structuralReflectionCoefficient = (self.AdjacentReflectionCoefficient(i) + self.reflectedLight[i+1]*
            mpmath.e ** ((self.TraditionalModelThickness(i + 1) * self.DetectionDepth(i + 1)) * 2j)) / \
                (1 + self.AdjacentReflectionCoefficient(i) * self.reflectedLight[i+1] * mpmath.e ** ((2j) *
                    self.TraditionalModelThickness(i + 1) * self.DetectionDepth(i + 1)))
        return structuralReflectionCoefficient

    # ri,i+1 = [(ni+1**2/kz,i+1)-(ni**2/kz,i)]/[(ni+1**2/kz,i+1)+(ni**2/kz,i)]
    def AdjacentReflectionCoefficient(self, i):

        refractiveIndex_0 = self.TraditionalModelRefractiveIndex(i)
        refractiveIndex_1 = self.TraditionalModelRefractiveIndex(i + 1)
        detectionDepth_0 = self.DetectionDepth(i)
        detectionDepth_1 = self.DetectionDepth(i + 1)
        #print(refractiveIndex_0,refractiveIndex_1,detectionDepth_0,detectionDepth_1)

        adjacentReflectionCoefficient = (
                                        refractiveIndex_1 ** 2 / detectionDepth_1 - refractiveIndex_0 ** 2 / detectionDepth_0) \
                                        / (
                                        refractiveIndex_1 ** 2 / detectionDepth_1 + refractiveIndex_0 ** 2 / detectionDepth_0)
        return adjacentReflectionCoefficient

    def DetectionDepth(self, i):

        detectionDepth = (((2 * mpmath.pi / self.wavelength * self.TraditionalModelRefractiveIndex(
            i)) ** 2 - self.HorizontalWaveVector() ** 2) ** (1 / 2))
        #print(self.wavelength,self.HorizontalWaveVector())
        return detectionDepth

    def HorizontalWaveVector(self):
        horizontalWaveVector = 2 * mpmath.pi / self.wavelength * self.coupledPrismRefractiveIndex * mpmath.sin(
            self.incidentAngle)
        return horizontalWaveVector

    def TraditionalModelRefractiveIndex(self, i):
        if i == 1:
            return self.coupledPrismRefractiveIndex
        elif (i == 2):
            return self.metalRefractiveIndex
        elif (i == 3):
            return self.aqueousSolutionRefractiveIndex
        else:
            print("TraditionalModelRefractiveIndex: i is %i"%i)

            return 1

    def TraditionalModelThickness(self, i):
        if (i == 1):
            return self.coupledPrismThickness
        elif (i == 2):
            return self.metalThickness
        elif (i == 3):
            return self.aqueousSolutionThickness
        else:
            print(" TraditionalModelThickness:  i is %i"%i)
            return 1
