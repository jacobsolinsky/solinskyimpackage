#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:46:42 2018

@author: Jacob Solinsky
Stitches nikon im images
"""
import os
os.chdir('/Users/Lab/improject/implace/nd2-2')
from skimage import io, morphology as morph, filters
name1 = 'NRVM-sensory vectorXY'
name2 = '.tif'
import numpy as np
import cv2 as cv
disk = morph.disk(25)
for time in range(1, 66):
    h2bim = np.zeros((6912, 8192)).astype('uint16')
    dhbim = np.zeros((6912, 8192)).astype('uint16')
    for xy in range(1, 50):
        h2bimlet = io.imread(name1 + str(xy).zfill(2) + 'T'+str(time).zfill(2) +'C1.tif')
        dhbimlet = io.imread(name1 + str(xy).zfill(2) + 'T'+str(time).zfill(2) +'C2.tif')
        rowstart = (xy-1) // 7 * 972
        rowend = rowstart + 1080 
        colstart = 1152*((xy-1) % 7)
        colend = colstart+1280
        h2bim[rowstart:rowend, colstart:colend] = h2bimlet
        dhbim[rowstart:rowend, colstart:colend] = dhbimlet
    h2bim = cv.morphologyEx(h2bim, cv.MORPH_TOPHAT, disk)
    dhbim = cv.morphologyEx(dhbim, cv.MORPH_TOPHAT, disk)
    io.imsave('h2b' + str(time) + '.tif', h2bim)
    io.imsave('dhb' + str(time) + '.tif', dhbim)
    print(time)

        