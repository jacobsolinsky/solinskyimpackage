#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:12:33 2018

@author: Lab
"""
from skimage import filters, io,  measure, draw, morphology as morph, segmentation as seg
import numpy as np
import solinskyim.solinskyim as solim
import random
import os
labelmatrixlist = []
nucimlist = []
dhbimlist = []
framenuclist = []
impath = '/Users/Lab/improject/implace/nd2-2/'
temppath = '/Users/Lab/improject/temp/nd2-2/'
if not os.path.exists(temppath):
    os.mkdir(temppath)
frameno = 65
diskrad = 20

class Nuc:
    def __init__(self, label, centroid, area, eccentricity, mean_intensity, frame, coords):
        self.label = label
        self.centroid = centroid
        self.area = area
        self.eccentricity = eccentricity
        self.mean_intensity = mean_intensity
        self.frame = frame
        self.coords = coords
        self.uniquelabel = '{}_{}'.format(frame, label)
class Track:
    TRACKNO = 0
    def __init__(self, nuc):
        self.nuclei = [nuc]
        self.uniquelabel = self.TRACKNO
        self.TRACKNO += 1
        self.color = (random.sample(range(1,65536), 1), random.sample(range(1,65536), 1), random.sample(range(1,65536), 1))
        self.daughters = []
    def pop(self):
        return self.nuclei.pop
    
    def append(self, nuc):
        self.nuclei.append(nuc)
    
    def extend(self, nuclei):
        self.nuclei.extend(nuclei)
        
    def firstframe(self):
        return self.nuclei[0].frame
    
    def lastframe(self):
        return self.nuclei[-1].frame
    
    def getframe(self, frame):
        return self.nuclei[frame-self.firstframe()]
    
    def mitosis(self, track):
        self.daughters.append[track]
#%%
'''
Get label images from each nucleus image in each frame
'''
nuclist = dict()
threshlist = []
for frame in range(1, frameno+1):
    dapiimpath= '{}NRVM-sensory vectorXY01T{:0>2}C1.tif'.format(impath,frame)
    dapiim = io.imread(dapiimpath)
    dapiim = filters.gaussian(dapiim)*65535
    '''
    This step is to remove shadows from the background that are created by the
    H2b negative cardiomyocyte nuclei. If these shadows are not properly removed,
    during the succeeding background subtraction step they become dramatically highlighted
    and greatly confuse the algorithm, as background subtraction carries with it 
    the assumption that there are no sharp transitions of brightness in 
    the background.
    '''
    dapiimseed = np.copy(dapiim)
    dapiimseed[1:-1, 1:-1] = dapiim.max()
    dapiimfilled = morph.reconstruction(dapiimseed, dapiim, method='erosion')
    dapiim = solim.diskfilter(dapiimfilled, 25)
    dapibin = solim.relperimeterthresh(dapiim)
    dapibin = morph.remove_small_objects(dapibin, 40)
    dapilab = solim.tunewatershed(dapiim, mask = dapibin, distance = 2, brightness = 600)
    dapilab = solim.segrandomizer(dapilab)
    '''
    Extract nucleus properties from each region in the label images
    '''
    nuclist[frame] = [Nuc(nuc.label, nuc.centroid, nuc.area, nuc.eccentricity, nuc.mean_intensity, frame, nuc.coords)\
           for nuc in measure.regionprops(dapilab, dapiim)]
    io.imsave(f'{temppath}dapilab{frame}.png',dapilab)
    print(frame)
'''
For each nucleus centroid in frame 1 birth a track
'''
#%%
tracklist = []
for nuc in nuclist[1]:
    tracklist.append(Track(nuc))
    
def eucliddis(a, b):
    return ((a[0] - b[0]) **2) + ((a[1] - b[1]) ** 2) ** 0.5
'''
For each nucleus in frame i link to a track in frame i + 1
'''

for frame in range(2, frameno+1):
    framenuclei = nuclist[frame]
    print(frame)
    for track in tracklist:
        if track.lastframe() == frame - 1:
            nuccandid = track.nuclei[-1]
        else:
            continue
        trackdistances = np.zeros(len(framenuclei))
        if len(framenuclei) < 1:
            continue
        for i, nuc in enumerate(framenuclei):
            trackdistances[i] = eucliddis(nuccandid.centroid, nuc.centroid)
        if np.min(trackdistances) < 20:
            track.append(framenuclei.pop(np.argmin(trackdistances)))
    for nuc in framenuclei:
        tracklist.append(Track(nuc))
toino = np.where(np.array([len(t.nuclei) for t in tracklist]) > 20)
toiist = [tracklist[i] for i in toino[0]]
#%%
'''
This code segment identifies mitoses
'''
for frame in range(2, frameno + 1):
    birthlist = []
    for track in tracklist:
        if frame == track.firstframe():
            birthlist.append(track)
            
    interlist = []
    for track in tracklist:
        if track.firstframe() < frame < track.lastframe():
            interlist.append(track)

    for intertrack in interlist:
        intercentroid = intertrack.getframe(frame).centroid
        birthdislist = np.zeros(len(birthlist))
        for i, birthtrack in enumerate(birthlist):
            birthdislist[i] = eucliddis(intercentroid, birthtrack.nuclei[0].centroid)
        try:
            if np.min(birthdislist) < 20:
                intertrack.mitosis(birthlist.pop(np.argmin(birthdislist)))
        except:
            pass
    print(frame)

#%%
'''
This code segment creates diagnostic images tracking a nucleus of your choosing
'''
j = 0
k = 0
for frame in range(1, frameno+1):
    dapiimpath= '{}NRVM-sensory vectorXY01T{:0>2}C1.tif'.format(impath,frame)
    dapiim = io.imread(dapiimpath)
    nuctrackim = np.zeros([dapiim.shape[0], dapiim.shape[1], 3])
    greydapiim = np.copy(nuctrackim)
    greydapiim[:, :, 2] = dapiim
    print(j)
    j+=1
    for toi in toiist:
        if toi.firstframe() <= frame <= toi.lastframe():
            nuc = toi.getframe(frame)
            xcoords = list(zip(*nuc.coords))[0]
            ycoords = list(zip(*nuc.coords))[1]
            nuctrackim[xcoords, ycoords] = toi.color
    nucbordim = seg.find_boundaries(nuctrackim[:,:,0])
    nucbordim = np.stack([nucbordim, nucbordim, nucbordim], axis = 2)
    nuctrackim = np.where(nucbordim, nuctrackim, greydapiim).astype('uint16')
    for toi in toiist:
        for daughter in toi.daughters:
            if frame == daughter.firstframe():
                circent = daughter.nuclei[0].centroid
                circoor = draw.circle(circent[0], circent[1], 4)
                nuctrackim[circoor] = daughter.color
                print('m' + str(k))
                k += 1
    io.imsave(f'{temppath}nuctrackim{frame}.png', nuctrackim)
    

            

    