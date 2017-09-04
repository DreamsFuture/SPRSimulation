#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/4 14:24
# @Author  : Colin
# @Site    : https://github.com/DreamsFuture
# @File    : SimulationWithAnnealingAndGeneticAlgorithm.py
# @Software: PyCharm Community Edition

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
import time

plot.rcParams['font.sans-serif']=['SimHei']
plot.rcParams['axes.unicode_minus']=False
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

        multilayers =not False
        tradition =  False
        metalProperties =False


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
                self.traditionTheoreticalMethodNonIterate.metalThickness = 56.8
                self.traditionTheoreticalMethodNonIterate.metalRefractiveIndex = self.AgMetalRefractiveIndex
                intensity_1_y, intensity_1_x,dep = self.Result(self.traditionTheoreticalMethodNonIterate)
                x65 = 0
                for i in intensity_1_x:
                    if (math.fabs(i - 65.00) < 0.5):
                        x65 = i
                        break
                x65 = intensity_1_x.index(x65)
                intensity_1_y = intensity_1_y[x65:]
                intensity_1_x = intensity_1_x[x65:]
                halfFullWidth.append(self.HalfFullWidth(intensity_1_x, intensity_1_y))
                maximumSlopeRate.append(self.MaximumSlopeRate(intensity_1_x, intensity_1_y))
                legendLabel.append("Ag:%s" % self.AgMetalThickness)
                self.traditionTheoreticalMethodNonIterate.metalThickness = 56.8
                self.traditionTheoreticalMethodNonIterate.metalRefractiveIndex = self.AuMetalRefractiveIndex
                intensity_2_y, intensity_2_x,dep = self.Result(self.traditionTheoreticalMethodNonIterate)
                intensity_2_y = intensity_2_y[x65:]
                intensity_2_x = intensity_2_x[x65:]
                halfFullWidth.append(self.HalfFullWidth(intensity_2_x, intensity_2_y))
                maximumSlopeRate.append(self.MaximumSlopeRate(intensity_2_x, intensity_2_y))
                legendLabel.append("Au:%s" % self.AuMetalThickness)
                #pop函数是先进后出的模式，类似于栈的数据结构
                print("\nmaximumSlopeRate(Au最大斜率，最大斜率对应的角度):    ")
                print(maximumSlopeRate.pop())
                print("\nmaximumSlopeRate(Ag最大斜率，最大斜率对应的角度):    ")
                print(maximumSlopeRate.pop())
                print("\nhalfFullWidth(Au半峰宽值，半峰宽第一个值对应的横坐标角度值，半峰宽最后一个值对应的横坐标角度值,SPR角度):")
                print(halfFullWidth.pop())
                print("\nhalfFullWidth(Ag半峰宽值，半峰宽第一个值对应的横坐标角度值，半峰宽最后一个值对应的横坐标角度值，SPR角度):")
                print(halfFullWidth.pop())
                #intensity_1_y, intensity_2_y = self.NormlizedArray(intensity_1_y,intensity_2_y)
                self.PlotGraphs(xylabel,legendLabel,intensity_1_x,intensity_1_y,  intensity_2_y)
            #同一个金属，但是厚度不同导致的影响
        elif (multilayers):
            domain = [436.0877338447084, 249.89697184904634, 119.69219094148396, 37.46770096226627, 11, 632.8]
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

            # pop函数是先进后出的模式，类似于栈的数据结构
            print("\nmaximumSlopeRate(Au最大斜率，最大斜率对应的角度):    ")
            print(maximumSlopeRate.pop())
            print("\nhalfFullWidth(Au半峰宽值，半峰宽第一个值对应的横坐标角度值，半峰宽最后一个值对应的横坐标角度值,SPR角度):")
            print(halfFullWidth.pop())
            #for i in intensity_1_y:
                #print(i)
            #intensity_1_y, intensity_2_y= self.NormlizedArray(intensity_1_y, intensity_2_y)
            self.PlotGraphs(xylabel, legendLabel, intensity_1_x, intensity_1_y)
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
    def PlotGraphsWithReflection(self,xylabel,legendLabel, *array):
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

    def ReflectionResult(self,SPRAngle,classVariable):
        x = []
        intensity = []
        dep = []
        classVariable.incidentAngle = SPRAngle * mpmath.pi / 180
        for i in np.arange(1.0, 4.0, 0.1):
            #入射角转化为弧度
            #classVariable.incidentAngle = i
            classVariable.aqueousSolutionRefractiveIndex = i
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

        return intensity, x

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
            return np.abs(first[0] - first[lenth-1]),first[0],first[lenth-1],array_x[minIndex]

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
    start = time.clock()
    parameterInitialization = ParameterInitialization()
    parameterInitialization.PlotGraph()
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)

if __name__ == '__main__':
    main()
