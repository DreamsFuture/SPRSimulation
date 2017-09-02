# -*- encoding: utf-8 -*-
#为什么遇到bug总是很难察觉那里出问题了，自己在代码处理bug的问题上没有做什么工作，也就是raise方面没有做try...catch的工作
from __future__ import division

from random import random

from TheoreticalFormula import TheorecticalFormula
from TraditionTheoreticalMethod import TraditionTheoreticalMethod
from TraditionTheoreticalMethodNonIterate import TraditionTheoreticalMethodNonIterate
import matplotlib.pyplot as plot
from NewMethod import NewWayForModule
from MultilayerStructureMethod import MultilayerStructureMethod
import mpmath
import numpy as np
import math




class ParameterInitialization(object):  # class继承object类

    #初始化的时候numberOfLayers传过去是3，所以在另外一边会出现 out of index 的Error
    def __init__(self, numberOfLayers=9,
                 coupledPrismRefractiveIndex=1.515,
                 coupledPrismThickness=500,
                 highRefractiveIndexMedium=2.0373,
                 highRefractiveIndexMediumThickness=120,
                 mediRefractiveIndex = 1.8234,
                 mediRefractiveIndexMediumThickness = 250,
                 lowerRefractiveIndexMedium=1.457,
                 lowerRefractiveIndexMediumThickness=450,
                 AuMetalRefractiveIndex=0.16172 + 3.21182j,
                 AgMetalRefractiveIndex=0.05625 + 4.27603j,
                 metalThickness=50,
                 AuMetalThickness = 50,
                 AgMetalThickness =50,
                 aqueousSolutionRefractiveIndex=1.3321,
                 aqueousSolutionThickness=4 * 10 ** 2,
                 wavelength=632.8,
                 aqueousSolutionRefractiveIndex_1=1.33211,
                 incidentAngle=61.55):

        self.numberOfLayers = numberOfLayers

        self.coupledPrismRefractiveIndex = coupledPrismRefractiveIndex
        self.coupledPrismThickness = coupledPrismThickness

        self.highRefractiveIndexMedium = highRefractiveIndexMedium
        self.highRefractiveIndexMediumThickness = highRefractiveIndexMediumThickness

        self.mediRefractiveIndex = mediRefractiveIndex
        self.mediRefractiveIndexMediumThickness = mediRefractiveIndexMediumThickness

        self.lowerRefractiveIndexMedium = lowerRefractiveIndexMedium
        self.lowerRefractiveIndexMediumThickness = lowerRefractiveIndexMediumThickness

        self.AuMetalRefractiveIndex = AuMetalRefractiveIndex
        self.AgMetalRefractiveIndex = AgMetalRefractiveIndex
        self.metalThickness = metalThickness
        self.AuMetalThickness = AuMetalThickness
        self.AgMetalThickness = AgMetalThickness

        self.aqueousSolutionRefractiveIndex = aqueousSolutionRefractiveIndex
        self.aqueousSolutionThickness = aqueousSolutionThickness
        self.aqueousSolutionRefractiveIndex_1 = aqueousSolutionRefractiveIndex_1

        self.wavelength = wavelength

        self.incidentAngle = incidentAngle

        self.newWayForModule = NewWayForModule(numberOfLayers=numberOfLayers,
                                                    coupledPrismRefractiveIndex=coupledPrismRefractiveIndex,
                                                    coupledPrismThickness=coupledPrismThickness,
                                                    highRefractiveIndexMedium=highRefractiveIndexMedium,
                                                    highRefractiveIndexMediumThickness=highRefractiveIndexMediumThickness,
                                                    mediRefractiveIndex = mediRefractiveIndex,
                                                    mediRefractiveIndexMediumThickness = mediRefractiveIndexMediumThickness,
                                                    lowerRefractiveIndexMedium=lowerRefractiveIndexMedium,
                                                    lowerRefractiveIndexMediumThickness=lowerRefractiveIndexMediumThickness,
                                                    metalRefractiveIndex=AuMetalRefractiveIndex,
                                                    metalThickness=metalThickness,
                                                    AuMetalRefractiveIndex=AuMetalRefractiveIndex,
                                                    AgMetalRefractiveIndex=AgMetalRefractiveIndex,
                                                    AuMetalThickness = AuMetalThickness,
                                                    AgMetalThickness = AgMetalThickness,
                                                    wavelength=wavelength,
                                                    incidentAngle=incidentAngle,
                                                    aqueousSolutionRefractiveIndex=aqueousSolutionRefractiveIndex,
                                                    aqueousSolutionThickness=aqueousSolutionThickness)
        self.multilayerStructureMethod = MultilayerStructureMethod(numberOfLayers=numberOfLayers,
                                                    coupledPrismRefractiveIndex=coupledPrismRefractiveIndex,
                                                    coupledPrismThickness=coupledPrismThickness,
                                                    highRefractiveIndexMedium=highRefractiveIndexMedium,
                                                    highRefractiveIndexMediumThickness=highRefractiveIndexMediumThickness,
                                                    mediRefractiveIndex = mediRefractiveIndex,
                                                    mediRefractiveIndexMediumThickness = mediRefractiveIndexMediumThickness,
                                                    lowerRefractiveIndexMedium=lowerRefractiveIndexMedium,
                                                    lowerRefractiveIndexMediumThickness=lowerRefractiveIndexMediumThickness,
                                                    metalRefractiveIndex=AuMetalRefractiveIndex,
                                                    metalThickness=metalThickness,
                                                    AuMetalRefractiveIndex=AuMetalRefractiveIndex,
                                                    AgMetalRefractiveIndex=AgMetalRefractiveIndex,
                                                    AuMetalThickness = AuMetalThickness,
                                                    AgMetalThickness = AgMetalThickness,
                                                    wavelength=wavelength,
                                                    incidentAngle=incidentAngle,
                                                    aqueousSolutionRefractiveIndex=aqueousSolutionRefractiveIndex,
                                                    aqueousSolutionThickness=aqueousSolutionThickness)
        self.traditionTheoreticalMethod = TraditionTheoreticalMethod(numberOfLayers=numberOfLayers,
                                                    coupledPrismRefractiveIndex=coupledPrismRefractiveIndex,
                                                    coupledPrismThickness=coupledPrismThickness,
                                                    highRefractiveIndexMedium=highRefractiveIndexMedium,
                                                    highRefractiveIndexMediumThickness=highRefractiveIndexMediumThickness,
                                                    lowerRefractiveIndexMedium=lowerRefractiveIndexMedium,
                                                    lowerRefractiveIndexMediumThickness=lowerRefractiveIndexMediumThickness,
                                                    metalRefractiveIndex=AuMetalRefractiveIndex,
                                                    metalThickness=metalThickness,
                                                    wavelength=wavelength, incidentAngle=incidentAngle,
                                                    aqueousSolutionRefractiveIndex=aqueousSolutionRefractiveIndex,
                                                    aqueousSolutionThickness=aqueousSolutionThickness)

        self.traditionTheoreticalMethodNonIterate = TraditionTheoreticalMethodNonIterate(numberOfLayers=numberOfLayers,
                                                    coupledPrismRefractiveIndex=coupledPrismRefractiveIndex,
                                                    coupledPrismThickness=coupledPrismThickness,
                                                    highRefractiveIndexMedium=highRefractiveIndexMedium,
                                                    highRefractiveIndexMediumThickness=highRefractiveIndexMediumThickness,
                                                    lowerRefractiveIndexMedium=lowerRefractiveIndexMedium,
                                                    lowerRefractiveIndexMediumThickness=lowerRefractiveIndexMediumThickness,
                                                    metalRefractiveIndex=AuMetalRefractiveIndex,
                                                    metalThickness=metalThickness,
                                                    AuMetalRefractiveIndex=AuMetalRefractiveIndex,
                                                    AgMetalRefractiveIndex=AgMetalRefractiveIndex,
                                                    wavelength=wavelength,
                                                    incidentAngle=incidentAngle,
                                                    aqueousSolutionRefractiveIndex=aqueousSolutionRefractiveIndex,
                                                    aqueousSolutionThickness=aqueousSolutionThickness)














    def PlotGraph(self):
        new = False
        multilayers =not False
        tradition =  False
        metalProperties =not False
        metalThickness =False
        lightWavelengthProperties =  False
        sensitivityIntensity =False
        old_settings = np.seterr(all='warn', over='raise')
        #传统方法层数为3层，总共耦合层、金属层、检测物质层
        if tradition:
            print("None Iteration and Tradition way for stimulation:\n")
            #金属折射率不同Ag=0.05625 + 4.27603j,Au=0.16172 + 3.21182j
            if metalProperties:
                xylabel = ["Angle(Degree)","Intensity","SPR Curve"]
                legendLabel = []
                halfFullWidth = []
                maximumSlopeRate = []

                self.traditionTheoreticalMethodNonIterate.numberOfLayers = 3
                self.traditionTheoreticalMethodNonIterate.metalThickness = self.metalThickness
                self.traditionTheoreticalMethodNonIterate.metalRefractiveIndex = self.AuMetalRefractiveIndex
                intensity_1_y, intensity_1_x = self.Result(self.traditionTheoreticalMethodNonIterate)
                halfFullWidth.append(self.HalfFullWidth(intensity_1_x, intensity_1_y))
                maximumSlopeRate.append(self.MaximumSlopeRate(intensity_1_x, intensity_1_y))

                legendLabel.append("Ag:%s" % self.AgMetalThickness)

                self.traditionTheoreticalMethodNonIterate.metalThickness = self.metalThickness
                self.traditionTheoreticalMethodNonIterate.metalRefractiveIndex = self.AuMetalRefractiveIndex
                intensity_2_y, intensity_2_x = self.Result(self.traditionTheoreticalMethodNonIterate)
                halfFullWidth.append(self.HalfFullWidth(intensity_2_x, intensity_2_y))
                maximumSlopeRate.append(self.MaximumSlopeRate(intensity_2_x, intensity_2_y))

                legendLabel.append("Au:%s" % self.AuMetalThickness)

                print("maximumSlopeRate:\n")
                print(maximumSlopeRate)
                print("halfFullWidth:\n")
                print(halfFullWidth)
                #intensity_1_y, intensity_2_y = self.NormlizedArray(intensity_1_y,intensity_2_y)

                self.PlotGraphs(xylabel,legendLabel,intensity_1_x,intensity_1_y,  intensity_2_y)
            #同一个金属，但是厚度不同导致的影响
            elif metalThickness:
                legendLabel = []
                halfFullWidth = []
                maximumSlopeRate = []
                SI = []

                self.traditionTheoreticalMethodNonIterate.metalThickness = 30
                intensity_1_y, intensity_1_x = self.Result(self.traditionTheoreticalMethodNonIterate)
                halfFullWidth.append(self.HalfFullWidth(intensity_1_x, intensity_1_y))
                maximumSlopeRate.append(self.MaximumSlopeRate(intensity_1_x, intensity_1_y))
                maximumSensitivityAngle = maximumSlopeRate[-1][1]
                SI_30 = self.SensitivityIntensity(maximumSensitivityAngle,self.traditionTheoreticalMethodNonIterate)
                legendLabel.append("MetalThickness = 30")
                SI.append("SI_30 = %s"%str(SI_30))

                self.traditionTheoreticalMethodNonIterate.metalThickness = 40
                intensity_2_y, intensity_2_x = self.Result(self.traditionTheoreticalMethodNonIterate)
                halfFullWidth.append(self.HalfFullWidth(intensity_2_x, intensity_2_y))
                maximumSlopeRate.append(self.MaximumSlopeRate(intensity_2_x, intensity_2_y))
                maximumSensitivityAngle = maximumSlopeRate[-1][1]
                SI_40 = self.SensitivityIntensity(maximumSensitivityAngle,self.traditionTheoreticalMethodNonIterate)
                legendLabel.append("MetalThickness = 40")
                SI.append("SI_40 = %s" % str(SI_40))

                self.traditionTheoreticalMethodNonIterate.metalThickness = 50
                intensity_3_y, intensity_3_x = self.Result(self.traditionTheoreticalMethodNonIterate)
                halfFullWidth.append(self.HalfFullWidth(intensity_3_x, intensity_3_y))
                maximumSlopeRate.append(self.MaximumSlopeRate(intensity_3_x, intensity_3_y))
                maximumSensitivityAngle = maximumSlopeRate[-1][1]
                SI_50 = self.SensitivityIntensity(maximumSensitivityAngle,self.traditionTheoreticalMethodNonIterate)
                legendLabel.append("MetalThickness = 50")
                SI.append("SI_50 = %s" % str(SI_50))

                self.traditionTheoreticalMethodNonIterate.metalThickness = 60
                intensity_4_y, intensity_4_x = self.Result(self.traditionTheoreticalMethodNonIterate)
                halfFullWidth.append(self.HalfFullWidth(intensity_4_x, intensity_4_y))
                maximumSlopeRate.append(self.MaximumSlopeRate(intensity_4_x, intensity_4_y))
                maximumSensitivityAngle = maximumSlopeRate[-1][1]
                SI_60 = self.SensitivityIntensity(maximumSensitivityAngle)
                legendLabel.append("MetalThickness = 60")
                SI.append("SI_60 = %s" % str(SI_60))

                self.traditionTheoreticalMethodNonIterate.metalThickness = 70
                intensity_5_y, intensity_5_x = self.Result(self.traditionTheoreticalMethodNonIterate)
                halfFullWidth.append(self.HalfFullWidth(intensity_5_x, intensity_5_y))
                maximumSlopeRate.append(self.MaximumSlopeRate(intensity_5_x, intensity_5_y))
                maximumSensitivityAngle = maximumSlopeRate[-1][1]
                SI_70 = self.SensitivityIntensity(maximumSensitivityAngle,self.traditionTheoreticalMethodNonIterate)
                legendLabel.append("MetalThickness = 70")
                SI.append("SI_70 = %s" % str(SI_70))
                """
                self.traditionTheoreticalMethodNonIterate.metalThickness = 80
                intensity_6_y, intensity_6_x = self.Result(self.traditionTheoreticalMethodNonIterate)
                halfFullWidth.append(self.HalfFullWidth(intensity_6_x, intensity_6_y))
                maximumSlopeRate.append(self.MaximumSlopeRate(intensity_6_x, intensity_6_y))
                maximumSensitivityAngle = maximumSlopeRate[-1][1]
                SI_80 = self.SensitivityIntensity(maximumSensitivityAngle)
                legendLabel.append("MetalThickness = 80")
                SI.append("SI_80 = %s" % str(SI_80))
                """
                print("maximumSlopeRate:\n")
                print(maximumSlopeRate)
                print("halfFullWidth:\n")
                print(halfFullWidth)
                print(SI)
                xylabel = ["Angle(Degree)","Intensity","SPR Curve"]

                intensity_1_y, intensity_2_y, intensity_3_y, intensity_4_y, intensity_5_y = self.NormlizedArray(intensity_1_y, intensity_2_y,intensity_3_y,intensity_4_y,intensity_5_y )
                self.PlotGraphs(xylabel,legendLabel, intensity_1_x, intensity_1_y, intensity_2_y,intensity_3_y,intensity_4_y,intensity_5_y)
            #入射光波长不同导致反射光强度的变化
            elif lightWavelengthProperties:
                legendLabel = []
                halfFullWidth = []
                maximumSlopeRate = []

                self.traditionTheoreticalMethodNonIterate.wavelength = 632.8
                intensity_1_y, intensity_1_x = self.Result(self.traditionTheoreticalMethodNonIterate)
                halfFullWidth.append(self.HalfFullWidth(intensity_1_x,intensity_1_y))
                maximumSlopeRate.append(self.MaximumSlopeRate(intensity_1_x,intensity_1_y))
                legendLabel.append("Wavelength = 632.8nm")

                self.traditionTheoreticalMethodNonIterate.wavelength = 850
                intensity_2_y, intensity_2_x = self.Result(self.traditionTheoreticalMethodNonIterate)
                halfFullWidth.append(self.HalfFullWidth(intensity_2_x, intensity_2_y))
                maximumSlopeRate.append(self.MaximumSlopeRate(intensity_2_x, intensity_2_y))
                legendLabel.append("Wavelength = 850nm")

                self.traditionTheoreticalMethodNonIterate.wavelength = 980
                intensity_3_y, intensity_3_x = self.Result(self.traditionTheoreticalMethodNonIterate)
                halfFullWidth.append(self.HalfFullWidth(intensity_3_x, intensity_3_y))
                maximumSlopeRate.append(self.MaximumSlopeRate(intensity_3_x, intensity_3_y))
                legendLabel.append("Wavelength = 980nm")

                print("maximumSlopeRate:\n")
                print(maximumSlopeRate)
                print("halfFullWidth:\n")
                print(halfFullWidth)
                xylabel = ["Angle(Degree)","Intensity",  "SPR Curve"]
                intensity_1_y, intensity_2_y, intensity_3_y= self.NormlizedArray(intensity_1_y, intensity_2_y, intensity_3_y)

                self.PlotGraphs(xylabel,legendLabel, intensity_1_x, intensity_1_y, intensity_2_y, intensity_3_y)

            elif sensitivityIntensity:

                legendLabel = []
                halfFullWidth = []
                maximumSlopeRate = []
                SI = []
                x =[]
                xylabel = ["Metal thickness","Sensitivity intensity","SI Information"]
                for i in np.arange(50,60,0.1):

                    self.traditionTheoreticalMethodNonIterate.metalThickness = i
                    intensity_1_y, intensity_1_x = self.Result(self.traditionTheoreticalMethodNonIterate)
                    halfFullWidth.append(self.HalfFullWidth(intensity_1_x, intensity_1_y))
                    maximumSlopeRate.append(self.MaximumSlopeRate(intensity_1_x, intensity_1_y))
                    maximumSensitivityAngle = maximumSlopeRate[-1][1]
                    SIValue = self.SensitivityIntensity(maximumSensitivityAngle,self.traditionTheoreticalMethodNonIterate)
                    legendLabel.append("Metal thickness = %snm"%i)
                    x.append(i)
                    SI.append(SIValue)

                self.PlotGraphs(xylabel, legendLabel,x,SI)

        elif (new):
            print("New way for stimulation:\n")
            legendLabel = []
            halfFullWidth = []
            maximumSlopeRate = []
            xylabel = ["Angle(Degree)","Intensity","SPR Curve"]
            xylabelfordep = ["Angle(Degree)", "Detection depth", "Detection Curve"]

            #新方法：银表面镀金
            self.newWayForModule.numberOfLayers = 8
            self.newWayForModule.AuMetalThickness = self.AuMetalThickness - 20
            self.newWayForModule.AgMetalThickness = self.AgMetalThickness - 40
            intensity_1_y, intensity_1_x,dep_1 = self.Result(self.newWayForModule)
            halfFullWidth.append(self.HalfFullWidth(intensity_1_x, intensity_1_y))
            maximumSlopeRate.append(self.MaximumSlopeRate(intensity_1_x, intensity_1_y))
            SI_1 = self.SensitivityIntensity(maximumSlopeRate[0][1],self.newWayForModule)
            print("新方法：银表面镀金——灵敏度值SI:" + str(SI_1))
            self.PlotGraphDep(xylabelfordep, intensity_1_x, dep_1)

            legendLabel.append("Au:"+str(self.AuMetalThickness)+"  Ag:"+str(self.AgMetalThickness)+"  Au:"+str(self.AuMetalThickness))

            #金膜和银膜传统方法
            self.traditionTheoreticalMethodNonIterate.numberOfLayers = 3
            self.traditionTheoreticalMethodNonIterate.metalThickness = self.AgMetalThickness
            self.traditionTheoreticalMethodNonIterate.metalRefractiveIndex = self.AgMetalRefractiveIndex
            intensity_2_y, intensity_2_x,dep_2 = self.Result(self.traditionTheoreticalMethodNonIterate)
            halfFullWidth.append(self.HalfFullWidth(intensity_2_x, intensity_2_y))
            maximumSlopeRate.append(self.MaximumSlopeRate(intensity_2_x, intensity_2_y))
            SI_2 = self.SensitivityIntensity(maximumSlopeRate[0][1],self.traditionTheoreticalMethodNonIterate)
            print("传统方法：银薄膜——灵敏度值SI:" + str(SI_2))
            self.PlotGraphDep(xylabelfordep, intensity_1_x, dep_2)


            legendLabel.append("Ag:%snm"%self.AgMetalThickness)

            self.traditionTheoreticalMethodNonIterate.metalThickness = self.AuMetalThickness
            self.traditionTheoreticalMethodNonIterate.metalRefractiveIndex = self.AuMetalRefractiveIndex
            intensity_3_y, intensity_3_x,dep_3 = self.Result(self.traditionTheoreticalMethodNonIterate)
            halfFullWidth.append(self.HalfFullWidth(intensity_3_x, intensity_3_y))
            maximumSlopeRate.append(self.MaximumSlopeRate(intensity_3_x, intensity_3_y))
            SI_3 = self.SensitivityIntensity(maximumSlopeRate[0][1],self.traditionTheoreticalMethodNonIterate)
            print("传统方法：金薄膜——灵敏度值SI:" + str(SI_3))
            self.PlotGraphDep(xylabelfordep, intensity_1_x, dep_3)


            legendLabel.append("Au:%snm"%self.AuMetalThickness)

            print("maximumSlopeRate:\n")
            print(maximumSlopeRate)
            print("halfFullWidth:\n")
            print(halfFullWidth)

            intensity_1_y, intensity_2_y, intensity_3_y = self.NormlizedArray( intensity_1_y, intensity_2_y, intensity_3_y)
            self.PlotGraphs(xylabel, legendLabel, intensity_1_x, intensity_1_y, intensity_2_y, intensity_3_y)
        elif (multilayers):

            domain = [442.56419411949844, 279.86968114973445, 117.90222515851033, 33.64959923508323, 11, 632.8]
            #报错，显示的结果值大于1
            print("Multilayer way for stimulation:\n")
            legendLabel = []
            halfFullWidth = []
            maximumSlopeRate = []

            xylabel = ["Angle(Degree)","Intensity","SPR Curve"]

            #新方法：多层结构
            self.multilayerStructureMethod.numberOfLayers = domain[4]
            self.multilayerStructureMethod.metalThickness = domain[3]
            self.multilayerStructureMethod.metalRefractiveIndex = self.AuMetalRefractiveIndex
            self.multilayerStructureMethod.lowerRefractiveIndexMediumThickness = domain[0]
            self.multilayerStructureMethod.mediRefractiveIndexMediumThickness = domain[1]
            self.multilayerStructureMethod.highRefractiveIndexMediumThickness = domain[2]
            intensity_1_y, intensity_1_x,dep = self.Result(self.multilayerStructureMethod)

            #65°对应的数组索引为array_x.index(65),截取数组的一部分，也即切片
            x65 = 0
            for i in intensity_1_x:
                if (math.fabs(i - 65.00 )< 0.5):
                    x65 = i
                    break

            intensity_y = intensity_1_y[intensity_1_x.index(x65):]
            intensity_x = intensity_1_x[intensity_1_x.index(x65):]
            # print(intensity_1_x)

            halfFullWidth.append(self.HalfFullWidth(intensity_x, intensity_y))
            maximumSlopeRate.append(self.MaximumSlopeRate(intensity_x, intensity_y))
            SI = self.SensitivityIntensity(maximumSlopeRate[0][1],self.multilayerStructureMethod)
            print("灵敏度值SI:"+str(SI))
            legendLabel.append("Multilayer Au:%snm"%str(self.metalThickness))

            print("maximumSlopeRate:\n")
            print(maximumSlopeRate)
            print("halfFullWidth:\n")
            print(halfFullWidth)
            #for i in intensity_1_y:
                #print(i)
            #intensity_1_y, intensity_2_y= self.NormlizedArray(intensity_1_y, intensity_2_y)
            self.PlotGraphs(xylabel, legendLabel, intensity_x, intensity_y)
            xylabelfordep = ["Angle(Degree)","Detection depth","Detection Curve"]
            self.PlotGraphDep(xylabelfordep,intensity_1_x,dep)
        else:
            print("else")

    def PlotGraphs(self,xylabel,legendLabel, *array):
        if len(array) == 0:
            print("There is no arrays input!")
        else:
            array1 = []
            #提取输入的X，Y轴曲线
            for i in array:
                y = np.array(i)
                array1.append(y)

            #把所有的计算曲线信息都加入到同一个坐标轴里面
            for i in range(len(array1)):
                j = i+1
                if(j <= len(array1)-1):
                    plot.plot(array1[0], array1[j],label='%s'%legendLabel[i])
            #把这个绘制好的坐标轴显示出来

            plot.legend()
            #plot.legend(loc='lower left', bbox_to_anchor=(65, 0.4))
            plot.xlabel(xylabel[0])  # 默认为"Intensity"
            plot.ylabel(xylabel[1])#默认为"Angle(Degree)"
            plot.title(xylabel[2])#默认为"SPR Curve"
            plot.show()

    def PlotGraphDep(self,label,x,y):
                plot.plot(x,y)
                plot.xlabel(label[0])  # 默认为"Intensity"
                plot.ylabel(label[1])  # 默认为"Angle(Degree)"
                plot.title(label[2])  # 默认为"SPR Curve"
                plot.show()

    def Result(self,classVariable):
        x = []
        intensity = []
        dep = []
        for i in np.arange(55, 80, 0.05):
            #入射角转化为弧度
            #classVariable.incidentAngle = i
            classVariable.incidentAngle = i * mpmath.pi / 180
            #计算模型的折射率信息
            result,depResult = classVariable.ReflectedLightIntensity()
            #返回的信息不能是Nan，否则一整个数组都是Nan
            if math.isnan(result):
                continue
            intensity.append(result)
            x.append(i)
            dep.append(depResult)
        #归一化处理
        #intensity = self.NormlizedArray(intensity) 暂时不做归一化，目的就是把所有的内容合在一起归一化

        return intensity, x,dep



    #选取的半峰宽的基底为0-1，所以表示为0.5附近，但是得出Y-0.5<=0.01
    def HalfFullWidth(self,array_x,array_y):
        #求中间值，半峰就是1/2的值
        min = np.amin(array_y)
        minIndex = array_y.index(min)
        max = np.amax(array_y)
        first = []
        arraylen = len(array_y)
        for i in range(arraylen):
            # 求解半峰位置的值，第一个精度是一个问题，<0.001表示半峰值为一半的时候，最近的一个array_y值对于的x，和另外一边离半峰值最后一个x，两者相减就是半峰宽
            if(array_y[i]-(min+max)/2<0.001 and array_x[i]> array_x[minIndex] - 8 and array_x[i] < array_x[minIndex] + 8):
                first.append(array_x[i])
        lenth = len(first)
        if(lenth<2):
            return 0
        else:
            return np.abs(first[0] - first[lenth-1]),first[0],first[lenth-1]

    def MaximumSlopeRate(self,array_x,array_y):
        minIndex = array_y.index(np.amin(array_y))
        maxRate = 0
        x_axis = 0
        lenth = len(array_y)
        for i in range(lenth - 2):
            rate =np.abs((array_y[i] - array_y[i+1])/(array_x[i] - array_x[i+1]))
            if(rate>maxRate and array_x[i]>65 and array_x[i] < array_x[minIndex] + 8):
                maxRate = rate
                x_axis = array_x[i]
        return maxRate,x_axis

    def SensitivityIntensity(self,angle,classVariable):

        classVariable.incidentAngle = angle * mpmath.pi / 180
        classVariable.aqueousSolutionRefractiveIndex = self.aqueousSolutionRefractiveIndex
        nonChangeLightIntensity, _ = classVariable.ReflectedLightIntensity()
        classVariable.aqueousSolutionRefractiveIndex = self.aqueousSolutionRefractiveIndex_1
        changeLightIntensity, _ = classVariable.ReflectedLightIntensity()
        SI = (nonChangeLightIntensity - changeLightIntensity) / (
        self.aqueousSolutionRefractiveIndex - self.aqueousSolutionRefractiveIndex_1)
        return np.abs(SI)


    def NormlizedArray(self, *intensity):
        minVal = 100000
        maxVal = -1
        intensityArray = []
        if len(intensity) == 0:
            print("There is no arrays input!")
        else:
            #提取输入的X，Y轴曲线
            for i in intensity:
                y = np.array(i)
                intensityArray.append(y)
                if(minVal>np.amin(y)):
                    minVal = np.amin(y)
                if(maxVal<np.amax(y)):
                    maxVal = np.amax(y)

        for i in range(len(intensityArray)):
            outufuncXArray = self.normalize_func(minVal, maxVal)(intensityArray[i])  # the result is a ufunc object
            intensityArray[i] = outufuncXArray.astype(float)  # cast ufunc object ndarray to float ndarray
            yield intensityArray[i]

    # normalize ndarray
    def normalize_func(self, minVal, maxVal):
        def normalizeFunc(x):
            r = (x - minVal) / (maxVal - minVal)
            return r

        return np.frompyfunc(normalizeFunc, 1, 1)

def main():
    parameterInitialization = ParameterInitialization()
    parameterInitialization.PlotGraph()

if __name__ == '__main__':
    main()
