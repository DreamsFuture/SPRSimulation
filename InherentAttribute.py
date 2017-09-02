# -*- encoding: utf-8 -*-

class InherentAttribute(object):
    def __init__(self,numberOfLayers,coupledPrismRefractiveIndex,coupledPrismThickness,highRefractiveIndexMedium,highRefractiveIndexMediumThickness,
                 lowerRefractiveIndexMedium, lowerRefractiveIndexMediumThickness,metalRefractiveIndex,metalThickness):
        self.numberOfLayers = numberOfLayers
        self.coupledPrismRefractiveIndex =coupledPrismRefractiveIndex
        self.coupledPrismThickness = coupledPrismThickness

        self.highRefractiveIndexMedium = highRefractiveIndexMedium
        self.highRefractiveIndexMediumThickness = highRefractiveIndexMediumThickness
        self.lowerRefractiveIndexMedium = lowerRefractiveIndexMedium
        self.lowerRefractiveIndexMediumThickness = lowerRefractiveIndexMediumThickness
        self.metalRefractiveIndex = metalRefractiveIndex
        self.metalThickness = metalThickness

    def RefractiveIndex(self,i):
        if (i == 0):
            return self.coupledPrismRefractiveIndex
        elif(i == self.numberOfLayers ):
            return self.metalRefractiveIndex
        elif (i/2 != 0):
            return self.lowerRefractiveIndexMedium
        else:
            return self.highRefractiveIndexMedium

    def MediumThickness(self,i):
        if (i == 0):
            return self.coupledPrismThickness
        elif(i == self.numberOfLayers ):
            return self.metalThickness
        elif (i/2 != 0):
            return self.lowerRefractiveIndexMediumThickness
        else:
            return self.highRefractiveIndexMediumThickness