#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/3 10:33
# @Author  : Colin
# @Site    : https://github.com/DreamsFuture
# @File    : PredecessorsModelWithAnnealingOptium.py
# @Software: PyCharm Community Edition


# -*- encoding: utf-8 -*-
#为什么遇到bug总是很难察觉那里出问题了，自己在代码处理bug的问题上没有做什么工作，也就是raise方面没有做try...catch的工作
from __future__ import division
import random
from PredecessorsModel import PredecessorsModel
import mpmath
import numpy as np
import math
import time

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
        self.predecessorsModel = PredecessorsModel(numberOfLayers=numberOfLayers,
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

    def annealingoptimize(self,  T=10000.0, cool=0.98, step=1):
        # 随机初始化值
        #低折射率介质，中折射率介质，高折射介质，金属介质，波长
        domain = [403,107,440,22, 7, 632.8]
        temp = domain

        print("低， 中， 高， 金属， 波长；代价：SPR角，      半峰宽，      灵敏度\n")
        # 循环
        while T > 0.1:
            # 选择一个索引值，而且这个索引是要随着T的减少变小的
            step = math.exp(T/10000)
            # 选择一个改变索引值的方向

            for i  in range(len(temp)-1):
                # 构造新的解
                if(i != len(temp)-2):
                    tmp = temp[i] + random.uniform( - step*3,step*3)
                elif(i == len(temp) - 2):
                    tmp = temp[i] + random.uniform(- step, step)# -1 or 0 or 1


                if(tmp < 500 and tmp>20):
                    temp[i] = tmp
                else:
                    i = i-1
            #
            # tmp = temp[4] + random.choice(nRandom)
            # if(tmp<20 and tmp>=5):
            #     temp[4] = tmp

                # 判断越界情况

            # 计算当前成本和新的成本
            cost1 = self.CostFunction(temp)
            cost2 = self.CostFunction(domain)

            # 判断新的解是否优于原始解 或者 算法将以一定概率接受较差的解
            #半峰宽要变得越小，灵敏度要变得越大，这个代价才好
            if cost2[1] > cost1[1] and cost2[2] < cost1[2]  or random.random() < math.exp(-(cost2[1] - cost1[1]) / T):
                domain = temp
                #print(np.around(temp, decimals=2), "代价:", self.CostFunction(temp)[:])
                print(temp[:], "代价:", self.CostFunction(temp)[:])
            T = T * cool  # 温度冷却


        print("模拟退火算法得到的最小代价是：", self.CostFunction(domain)[:])
        return domain


    def CostFunction(self,domain):
        result = []
        multilayers = True
        if (multilayers):

            maximumSlopeRate = []
            #新方法：多层结构

            self.predecessorsModel.lowerRefractiveIndexMediumThickness = domain[0]
            self.predecessorsModel.highRefractiveIndexMediumThickness = domain[1]
            self.predecessorsModel.coupledPrismThickness = domain[2]
            self.predecessorsModel.metalThickness = domain[3]
            self.predecessorsModel.numberOfLayers = int(domain[4])
            self.predecessorsModel.wavelength = domain[5]
            self.predecessorsModel.metalRefractiveIndex = self.AuMetalRefractiveIndex
            self.predecessorsModel.aqueousSolutionThickness = 100
            intensity_1_y, intensity_1_x,dep = self.Result(self.predecessorsModel)

            # 65°对应的数组索引为array_x.index(65),截取数组的一部分，也即切片
            x65 = 0
            for i in intensity_1_x:
                if(math.fabs(i - 65.00 ) < 0.01):
                    x65 = i
                    break
            intensity_1_y = intensity_1_y[intensity_1_x.index(x65):]
            intensity_1_x = intensity_1_x[intensity_1_x.index(x65):]

            result.append(self.SPRAngle(intensity_1_x,intensity_1_y))
            result.append(self.HalfFullWidth(intensity_1_x, intensity_1_y))
            maximumSlopeRate.append(self.MaximumSlopeRate(intensity_1_x, intensity_1_y))
            result.append(self.SensitivityIntensity(maximumSlopeRate[0][1],self.predecessorsModel))
            result = np.array(result)

            # for i in range(len(result)):
            #     np.rint(result[i],3)

            return result
        else:
            print("else")

    def SPRAngle(self,x,y):
        return x[y.index(min(y))]

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

    #选取的半峰宽的基底为mediValue = (maxValue+minValue)/2，但是得出Y-mediValue<=0.01
    def HalfFullWidth(self,array_x,array_y):
        first = []
        min = np.amin(array_y)
        minIndex = array_y.index(min)
        max = np.amax(array_y)
        arraylen = len(array_y)
        for i in range(arraylen):
            #最小值正负8的横坐标为其可能的SPR激发曲线范围
            if(array_y[i]-(min+max)/2<0.01 and array_x[i]>array_x[minIndex] - 8 and array_x[i] < array_x[minIndex] + 8):
                first.append(array_x[i])
        lenth = len(first)
        if(lenth<2):
            return 0
        else:
            return np.abs(first[0] - first[lenth-1])

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

        classVariable.incidentAngle = angle*mpmath.pi/180
        classVariable.aqueousSolutionRefractiveIndex = self.aqueousSolutionRefractiveIndex
        nonChangeLightIntensity,_ = classVariable.ReflectedLightIntensity()
        classVariable.aqueousSolutionRefractiveIndex = self.aqueousSolutionRefractiveIndex_1
        changeLightIntensity,_ = classVariable.ReflectedLightIntensity()
        SI = (nonChangeLightIntensity - changeLightIntensity)/(self.aqueousSolutionRefractiveIndex - self.aqueousSolutionRefractiveIndex_1)
        return np.abs(SI)

def main():
    start = time.clock()
    annealing =not  False
    genetic =False
    parameterInitialization = ParameterInitialization()
    if(annealing):
        parameterInitialization.annealingoptimize()
    elif(genetic):
        parameterInitialization.geneticoptimize()
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)

if __name__ == '__main__':
    main()
