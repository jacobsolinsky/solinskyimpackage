#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:49:44 2018

@author: Lab
"""
import bokeh.plotting
from bokeh.io import show
import numpy as np
from solinskyimpackage.solinskyim import _middler as _middler
location = "/Users/Lab/improject/.tempplot/tempplot.html"
bokeh.plotting.output_file(location)
def bokehhist(data, bins = 10, range = None):
    if range is None:
        range = (np.min(data), np.max(data))
    hist = np.histogram(data, bins=bins, range = range)
    xhist = _middler(hist[1])
    yhist = hist[0]
    p = bokeh.plotting.figure(plot_width=1000, plot_height=500)
    p.line(xhist, yhist, line_color='blue', line_alpha=0.5)
    show(p)