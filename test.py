#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:20:27 2018

@author: Lab
"""
imagefolder = '/Users/Lab/improject/DNAquantification/images/180620_191225_CD1Rab8OEHighGlucosepHH3'
platename  = '180620_191225_CD1Rab8OEHighGlucosepHH3'
import numpy as np
import itertools as it
from solinskyimpackage.dapigfprfp import Imageset 
import cv2
from timeit import default_timer as timer
from joblib import Parallel, delayed
from skimage.filters import gaussian





