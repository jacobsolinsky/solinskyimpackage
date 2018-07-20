# -*- coding: utf-8 -*-
#48 minutes on lab computer
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import filters
from skimage import io
from skimage import measure
from skimage import segmentation as seg
from skimage import morphology as morph
from scipy import ndimage
from scipy.ndimage import morphology as ndmorph
from math import floor, sqrt
from string import ascii_uppercase
import os
import sys
import random
import cv2
from ..multiotsu import threshold_multiotsu as threshold_multiotsu


sys.path.append('/Users/Lab/improject')
vals = np.linspace(0, 1, 10000)
np.random.shuffle(vals)
randomcmap = plt.cm.colors.ListedColormap(plt.cm.jet(vals))

def objectsmallfilter(map, size):
    label_img, cc_num = ndimage.label(map)
    cc_areas = ndimage.sum(map, label_img, range(cc_num+1))
    area_mask = (cc_areas < size)
    label_img[area_mask[label_img]] = 0
    return label_img
def objectlargefilter(map, size):
    label_img, cc_num = ndimage.label(map)
    cc_areas = ndimage.sum(map, label_img, range(cc_num+1))
    area_mask = (cc_areas > size)
    label_img[area_mask[label_img]] = 0
    return label_img
def analyze(platename, analyzeedu = True, rowstart = 2, rowend = 7, colstart =2, colend = 11, rowinspect = None, colinspect = None, filterradius = 25):
    platefullname = platename
    platenumname = platefullname[0:13] #This is the 6 digit date _ those other 6 numbers that identifies a set of images taken on the automated microscope here. I use these 2 numbers to name the scripts and data associated with that image set
    #This automatically generates the filenames our microscope creates based on stain, wellrow, wellcolumn, and site. The code expects that there be 56 sites per well, 8 rows, 7 columns, with  site 1 starting in the top left corner and the sites being ordered in row-major order. The code also expects that only rows B-G and columns 2-11 are in the set.
    # 1  2  3  4  5  6  7
    # 8  9  10 11 12 13 14
    # 15 16 ...
    sys.path.append(os.path.expanduser('~/improject/'))
    os.chdir(os.path.expanduser('~/improject/implace'))
    os.chdir(platefullname)
    dapin1 = '_01_1_'
    gfpn1 = '_01_2_'
    rfpn1 = '_01_3_'
    dapin2 = '_DAPI_001.tif'
    gfpn2 = '_GFP_001.tif'
    rfpn2 = '_RFP_001.tif'
    #This takes the list of rows and columns that have populated wells in them and selects a random one to bring to the end.
    collist = list(range(colstart, colend+1))
    rowlist = list(range(rowstart-1,rowend))
    if not rowinspect:
        rowinspect = random.choice(rowlist)
    if not colinspect:
        colinspect = random.choice(collist)
    rowlist.append(rowlist.pop(rowlist.index(rowinspect)))
    collist.append(collist.pop(collist.index(colinspect)))
    rowletters = [ascii_uppercase[i] for i in rowlist]
    #Initializes constants used in each iteration of the well forloops
    nuctablelist = []
    gfparealist = []
    rowno = 903*np.array(list(range(1, 8)))
    maskrowno = np.append(rowno, rowno+1)
    horzmark = np.zeros((904*8, 1224*7), dtype = 'bool_')
    horzmark[maskrowno,:] = True
    dapisitecell = np.zeros((904*8, 1224*7), dtype = 'uint16')
    gfpsitecell = np.zeros((904*8, 1224*7), dtype = 'uint16')
    rfpsitecell = np.zeros((904*8, 1224*7), dtype = 'uint16')
    disk = morph.disk(filterradius)
    for row in rowletters:
        for col in collist:
            #Assembles the stitched image for each well from the 56 site images. The  images from the microscope each have 904 rows and 1224 columns of pixels.
            for site in range(1,57):
                curdapi = row + str(col) + dapin1 + str(site) + dapin2
                curgfp = row + str(col) + gfpn1 + str(site) + gfpn2
                rstart = 904*floor((site-1)/7)
                rend = 904*(floor((site-1)/7) + 1)
                cstart = 1224*((site-1) % 7)
                cend = 1224*(((site-1) % 7) + 1)
                try:
                    dapisitecell[rstart:rend, cstart:cend] =io.imread(curdapi)
                except:
                    dapisitecell[rstart:rend, cstart:cend] = io.imread(row + str(col) + dapin1 + str(site-1) + dapin2)
                gfpsitecell[rstart:rend, cstart:cend] = io.imread(curgfp)
                if analyzeedu:
                    currfp = row + str(col) + rfpn1 + str(site) + rfpn2
                    rfpsitecell[rstart:rend, cstart:cend] = io.imread(currfp)
            #Identifies DAPI+ and GFP+ areas. I found the triangle method made
            #satisfactory thresholds for the troponin GFP stained images.
            #Troponin binary thresholded images do not have their borders cleared
            #as the cardiomyocytes often form large confluent masses that contact
            #the border of the image. Holes larger than 3 times the size of the average nucleus are not filled, as the cardiomyocytes often form rings which enclose large areas. Some hole filling is necessary though, as the cardiomyocyte nuclei tend to sit in a hole in the cell with very little troponin staining and would otherwise be classified as troponin negative cardiomyocyte nuclei. Hole filling is carried out after, not before, troponin positive area is quantified.
            if analyzeedu:
                rfpsitecell = cv2.morphologyEx(rfpsitecell, cv2.MORPH_TOPHAT, disk)
            gfpbin = gfpsitecell > threshold_multiotsu(gfpsitecell,4)[0][0]
            gfparea = np.sum(gfpbin)
            gfpbin = np.invert(objectsmallfilter(np.invert(gfpbin), 1000) >0)
            #I found that the Otsu method made satisfactory thresholds for the dapi images.
            #30 seconds
            dapibin = ndmorph.binary_fill_holes(dapisitecell > filters.threshold_otsu(dapisitecell))
            #1 minute
            dapilabel = measure.label(dapibin)
            #This normalized the RFP images with respect to background
            rfpsitecell = rfpsitecell.astype('uint16')
            #Classifies as troponin negative nuclei that overlap troponin+ and troponin- regions.
            #I've made the assumption that cardiomyocyte nuclei will appear to be entirely surrounded
            #by troponin positive cytoplasm.
            #Integrates RFP Intensity
            rfpprop = measure.regionprops(dapilabel, rfpsitecell)
            if analyzeedu:
                rfpint = np.array([np.sum(prop.intensity_image) for prop in rfpprop ])
            dapiarea = np.array([prop.area for prop in rfpprop])
            circularity = np.array([prop.perimeter/sqrt(prop.area) for prop in rfpprop])
            global lookcircle
            lookcircle = circularity
            tropoprop = measure.regionprops(dapilabel, gfpbin)
            tropopos = np.array([prop.min_intensity for prop in tropoprop])
            cutprop = measure.regionprops(dapilabel, horzmark)
            cutpos = np.array([prop.max_intensity for prop in cutprop])
            rowarray = np.full((len(rfpint),), row)
            colarray = np.full((len(rfpint),),col)
            if analyzeedu:
                nuctablelist.append(pd.DataFrame({'Row':rowarray, 'Column':colarray, 'IntegratedEdu':rfpint, 'TroponinPositive':tropopos, 'Area':dapiarea, 'Circularity':circularity,  'Cutoff':cutpos}))
            else:
                nuctablelist.append(pd.DataFrame({'Row':rowarray, 'Column':colarray, 'TroponinPositive':tropopos, 'Area':dapiarea, 'Circularity':circularity, 'Cutoff':cutpos}))
            print(row+str(col))
            gfparealist.append(pd.DataFrame({'Row':[row], 'Column':[col], 'GfpArea':[gfparea]}))
    nuctabler = pd.concat(nuctablelist).reset_index()
    nucfilter = (nuctabler['Cutoff'] == False) & (nuctabler['Area'] > 100) & (nuctabler['Area'] < 1500) & (nuctabler['Circularity'] < 5)
    nuctable = nuctabler[nucfilter]
    nuctable = nuctable.dropna()
    gfptable = pd.concat(gfparealist)
    goodnuclei = np.array([x in nuctable['index'] for x in nuctabler.index])
    troponuclei = np.array([x[1]['index'] in nuctable.index and x[1]['TroponinPositive'] for x in nuctabler.iterrows()])
    nuctabler  = pd.concat([nuctabler, pd.Series(goodnuclei, name = 'goodnuclei')], axis = 1)
    nuctabler  = pd.concat([nuctabler, pd.Series(troponuclei, name = 'troponuclei')], axis = 1)
    #Calculates which nuclei are Edu+ and Edu- from their integrated intensities in the RFP channel.
    if analyzeedu:
        rfpintfull = nuctable['IntegratedEdu']
        krfpos = rfpintfull > filters.threshold_triangle(rfpintfull.astype('float32'))
        rfppos = pd.Series(krfpos, name = 'EduPositive')
        rfppos = rfppos[rfppos]
        nuctable  = pd.concat([nuctable, rfppos], axis = 1)
        rfpnuclei = np.array([x in nuctable['index'] and x in rfppos.index for x in nuctabler.index])
        nuctabler  = pd.concat([nuctabler, pd.Series(rfpnuclei, name = 'rfpnuclei')], axis = 1)
    nuctable = nuctable.fillna(False)
    freqtable = nuctable.groupby(['Row', 'Column', 'TroponinPositive', 'EduPositive']).count()
    #Saves collected data in .csv files
    os.chdir(os.path.expanduser('~/improject/'))
    if not os.path.exists('imresult/' + platenumname):
        os.makedirs('imresult/'+platenumname)
    nuctable.to_csv('imresult/' + platenumname + '/' + 'nucleustable.csv')
    gfptable.to_csv('imresult/' + platenumname + '/' + 'gfpareatable.csv')
    freqtable.to_csv('imresult/' + platenumname + '/' + 'freqtable.csv')
    #Saves DAPI and troponin positivity assessment diagnostic images
    allsub = nuctabler.query('Row == @row & Column == @col')['goodnuclei'].as_matrix()
    allborder = ndmorph.binary_dilation(seg.find_boundaries(allsub[dapilabel-1], mode = 'inner')).astype('uint8')
    allborder = (allborder*255).astype('uint8')
    troppossub = nuctabler.query('Row == @row & Column == @col')['troponuclei'].as_matrix()
    tropposborder = ndmorph.binary_dilation(seg.find_boundaries(troppossub[dapilabel - 1], mode = 'inner'))
    tropposborder = (tropposborder*255).astype('uint8')
    io.imsave('imresult/' + platenumname + '/' + row + str(col) + 'gfpimage.tiff', gfpsitecell)
    io.imsave('imresult/' + platenumname + '/' + row + str(col) + 'dapiimage.tiff', dapisitecell)
    io.imsave('imresult/' + platenumname + '/' + row + str(col) + 'rfpimage.tiff', rfpsitecell)
    io.imsave('imresult/' + platenumname + '/' + row + str(col) + 'nucleusborder.png', allborder)
    io.imsave('imresult/' + platenumname + '/' + row + str(col) + 'cmnucleusborder.png', tropposborder)
    #Saves EDU positivity assessment diagnostic image
    if analyzeedu:
        rfppossub = nuctabler.query('Row == @row & Column == @col')['rfpnuclei'].as_matrix()
        rfpposborder = ndmorph.binary_dilation(seg.find_boundaries(rfppossub[dapilabel - 1], mode = 'inner'))
        rfpposborder = (rfpposborder*255).astype('uint8')
        io.imsave('imresult/' + platenumname+ '/' + row + str(col) + 'rfpnucleusborder.png', rfpposborder)
