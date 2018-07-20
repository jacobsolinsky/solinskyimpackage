
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 11:20:35 2018

@author: Lab
"""

import pandas as pd
import numpy as np
from . import solinskyim as solim
from rpy2.robjects import pandas2ri
import rpy2.rinterface as rinterface
rinterface.initr()
import rpy2.robjects as robjects
pandas2ri.activate()
def cellcyclemodel(data, datarange, bins):
    global g1modelr
    global g2modelr
    histogram = np.histogram(data, range=datarange, bins=bins)
    histogramx = solim._middler(histogram[1])
    histogramy = histogram[0]
    xdis = histogramx[1] -histogramx[0]
#find G1 max
    g1maxy = np.max(histogramy)
    g1maxset = np.where(histogramy > (0.8*g1maxy))
    g1maxleft = g1maxset[0][0]
    g1maxright = g1maxset[0][-1]
    g1maxxin = np.argmax(histogramy)
    g1maxx = histogramx[g1maxxin]
    g1mean = (g1maxleft + 1 + (g1maxright - g1maxleft) / 2) * xdis
    robjects.r(f'g1maxy = {g1maxy}')
    robjects.r(f'g1mean = {g1mean}')
#pick valleydivide
    valleydivide = float(input('type the x-coordinate of what looks to be the low point between the two peaks: '))
    valleydividein = np.abs(histogramx-valleydivide).argmin()
    valleydivide = histogramx[valleydividein]
    valleyheight = histogramy[valleydividein]
    robjects.r(f'valleyheight = {valleyheight}')
    robjects.r(f'valleydivide = {valleydivide}')
#find G2 max
    g2maxy = np.max(histogramy[histogramx > valleydivide])
    g2maxxin = int(2 * (g1maxleft + 1 + (g1maxright - g1maxleft) / 2))
    g2maxx = histogramx[g2maxxin]
    robjects.r(f'g2maxy = {g2maxy}')
    robjects.r(f'g2mean = {g2maxx}')
#pick bigdivide
    bigdivide = float(input('type the x-coordinate of where you think the upper boundary of the G2 peak is: '))
    bigdividein = np.abs(histogramx-bigdivide).argmin()
    bigdivide = histogramx[bigdividein]
#pick debrisdivide
    debrisdivide = float(input('type the x-coordinate of where you think the boundary between the G1 peak and debris is: '))
    debrisdividein = np.abs(histogramx-debrisdivide).argmin()
    debrisdivide = histogramx[debrisdividein]
#find g1 modeldata
    g1startsd = np.std(data[np.logical_and(debrisdivide<data, data<(debrisdivide + (g1maxx-debrisdivide)*2))])
    robjects.r(f'g1startsd = {g1startsd}')
    g1modelx = histogramx[debrisdividein:g1maxright + (g1maxleft - debrisdividein) + 1]
    g1modely = histogramy[debrisdividein:g1maxright + 1]
    g1modelyright = np.array(list(reversed(histogramy[debrisdividein:g1maxleft])))
    g1modely = np.concatenate((g1modely, g1modelyright))
    g1model = pd.DataFrame({'x':g1modelx, 'y':g1modely})
    g1modelr = pandas2ri.py2ri(g1model)
#find g2 modeldata
    g2startsd = np.std(data[np.logical_and(valleydivide<data, data<(valleydivide + (g2maxx-valleydivide)*2))])
    robjects.r(f'g2startsd = {g2startsd}')
    g2modelx = histogramx[g2maxxin - (bigdividein-g2maxxin):bigdividein + 1]
    g2modely = histogramy[g2maxxin:bigdividein+1]
    g2modely = np.concatenate([np.array(list(reversed(g2modely[1:]))),g2modely])
    g2model = pd.DataFrame({'x':g2modelx, 'y':g2modely})
    g2modelr = pandas2ri.py2ri(g2model)
    fulldata = pd.DataFrame({'x':histogramx, 'y':histogramy})
    fulldatar = pandas2ri.py2ri(fulldata)
    modelwindow = pd.DataFrame({'x':histogramx[g1maxxin:g2maxxin+1], 'y':histogramy[g1maxxin:g2maxxin+1]})
    modelwindowr = pandas2ri.py2ri(modelwindow)
    rinterface.globalenv.do_slot_assign('modelwindowr', modelwindowr)
    robjects.r('modelwindowr = attr(globalenv(), "modelwindowr")')
    rinterface.globalenv.do_slot_assign('fulldatar', fulldatar)
    robjects.r('fulldatar = attr(globalenv(), "fulldatar")')
    rinterface.globalenv.do_slot_assign('g1modelr', g1modelr)
    robjects.r('g1modelr = attr(globalenv(), "g1modelr")')
    rinterface.globalenv.do_slot_assign('g2modelr', g2modelr)
    robjects.r('g2modelr = attr(globalenv(), "g2modelr")')
    robjects.r("""
    library(purrr)
    dnormmodel = function(data, params, mean){
      return(params['height']*dnorm(x=data$x, sd=params['sd'], mean=mean))
    }
    g1model = partial(dnormmodel, mean=g1mean)
    g2model = partial(dnormmodel, mean=g2mean)
    measure_distance = function(params, data, model){
      diff = data$y - abs(model(data, params))
      return(sqrt(mean(diff^2)))
    }
    g1startheight = g1maxy / dnorm(0, sd=g1startsd)
    g2startheight = g2maxy / dnorm(0, sd=g2startsd)
    g1startparams = c(height=g1startheight, sd=g1startsd)
    g2startparams = c(height=g1startheight, sd=g2startsd)
    g1res = optim(g1startparams, measure_distance, data = g1modelr, model=g1model)
    g2res = optim(g1startparams, measure_distance, data = g2modelr, model=g2model)
    g1est = g1model(data=fulldatar, params=g1res$par)
    g2est = g2model(data=fulldatar, params=g2res$par)
    sest = fulldatar$y - (g1est + g2est)
        """)
    return(robjects.r('g1res$par'), robjects.r('g2res$par'), robjects.r('g1est'), robjects.r('g2est'), robjects.r('sest'),histogramy, histogramx)
    