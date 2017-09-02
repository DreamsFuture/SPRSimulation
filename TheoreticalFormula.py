# -*- encoding: utf-8 -*-
#金属部分中间使用银(Ag),外面镀金，防止被氧化
import numpy
import mpmath


# from InherentAttribute import InherentAttribute

class TheorecticalFormula(object):
    def __init__(self, numberOfLayers, coupledPrismRefractiveIndex, coupledPrismThickness, highRefractiveIndexMedium,
                 highRefractiveIndexMediumThickness,
                 lowerRefractiveIndexMedium, lowerRefractiveIndexMediumThickness, metalRefractiveIndex, metalThickness,
                 AuMetalRefractiveIndex,AgMetalRefractiveIndex,AuMetalThickness,AgMetalThickness,wavelength, incidentAngle,
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
        self.AuMetalRefractiveIndex = AuMetalRefractiveIndex
        self.AgMetalRefractiveIndex = AgMetalRefractiveIndex
        self.AuMetalThickness = AuMetalThickness
        self.AgMetalThickness = AgMetalThickness

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
        return R ** 2

    def StructuralReflectionCoefficient(self, i, N):
        for i in range(N-1,-1,-1):
            if i == N - 1:
                self.reflectedLight[i] = self.AdjacentReflectionCoefficient(i)
            else:
                self.reflectedLight[i] = self.TraditionalModel(i)
        # 防止出现0作为指数底数



    def TraditionalModel(self, i=1, N=3):

        structuralReflectionCoefficient = (self.AdjacentReflectionCoefficient(i) + self.reflectedLight[i+1]*
            mpmath.e ** ((self.MediumThickness(i + 1) * self.DetectionDepth(i + 1)) * 2j)) / \
                (1 + self.AdjacentReflectionCoefficient(i) * self.reflectedLight[i+1] * mpmath.e ** ((2j) *
                    self.MediumThickness(i + 1) * self.DetectionDepth(i + 1)))
        return structuralReflectionCoefficient

    # ri,i+1 = [(ni+1**2/kz,i+1)-(ni**2/kz,i)]/[(ni+1**2/kz,i+1)+(ni**2/kz,i)]
    def AdjacentReflectionCoefficient(self, i):

        refractiveIndex_0 = self.MediumThickness(i)
        refractiveIndex_1 = self.MediumThickness(i + 1)
        detectionDepth_0 = self.DetectionDepth(i)
        detectionDepth_1 = self.DetectionDepth(i + 1)
        adjacentReflectionCoefficient = (
                                        refractiveIndex_1 ** 2 / detectionDepth_1 - refractiveIndex_0 ** 2 / detectionDepth_0) \
                                        / (
                                        refractiveIndex_1 ** 2 / detectionDepth_1 + refractiveIndex_0 ** 2 / detectionDepth_0)
        return adjacentReflectionCoefficient

    def DetectionDepth(self, i):
        detectionDepth = (((2 * mpmath.pi / self.wavelength * self.MediumThickness(
            i)) ** 2 - self.HorizontalWaveVector() ** 2) ** (1 / 2))
        return detectionDepth

    def HorizontalWaveVector(self):
        horizontalWaveVector = 2 * mpmath.pi / self.wavelength * self.coupledPrismRefractiveIndex * mpmath.sin(
            self.incidentAngle)
        return horizontalWaveVector



    def RefractiveIndex(self,i):
        if (i == 1):
            return self.coupledPrismRefractiveIndex
        elif(i == self.numberOfLayers):
            return self.aqueousSolutionRefractiveIndex
        elif (i == self.numberOfLayers - 1 or i == self.numberOfLayers - 3):
            return self.AuMetalRefractiveIndex
        elif (i == self.numberOfLayers - 2):
            return self.AgMetalRefractiveIndex

        elif (i/2 != 0):
            return self.lowerRefractiveIndexMedium
        elif(i/2 == 0):
            return self.highRefractiveIndexMedium
        else:
            return 1

    def MediumThickness(self,i):
        if (i == 1):
            return self.coupledPrismThickness
        elif(i == self.numberOfLayers):
            return self.aqueousSolutionThickness
        elif (i == self.numberOfLayers - 1 or i == self.numberOfLayers - 3):
            return self.AuMetalThickness
        elif(i == self.numberOfLayers - 2):
            return self.AgMetalThickness
        elif (i/2 != 0):
            return self.lowerRefractiveIndexMediumThickness
        elif(i/2==0):
            return self.highRefractiveIndexMediumThickness
        else:
            return 1