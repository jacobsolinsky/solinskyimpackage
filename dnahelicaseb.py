#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:12:33 2018

@author: Lab
"""
from skimage import filters, io,  measure, morphology as morph, segmentation as seg
import numpy as np
import solinskyim.solinskyim as solim
import random
import os
import scipy.ndimage as nd
from scipy.optimize import linear_sum_assignment as linear_sum_assignment
from PIL import ImageDraw, Image
labelmatrixlist = []
nucimlist = []
dhbimlist = []
framenuclist = []
impath = '/Users/Lab/improject/implace/DhbProbeNoTreatment/'
temppath = '/Users/Lab/improject/temp/DhbProbeNoTreatment/'
if not os.path.exists(temppath):
    os.mkdir(temppath)
startframe = 20
endframe = 91
xno= 73
diskrad = 25
minnucsize = 200 #Minimum nucleus size in pixels, objects smaller than this will not be tracked
distthresh = 30

class Nuc:
    def __init__(self, label, centroid, area, perimeter, mean_intensity, frame, coords, bbox, origin = ''):
        self.label = label
        self.centroid = centroid
        self.area = area
        self.perimeter = perimeter / area
        self.mean_intensity = mean_intensity
        self.frame = frame
        self.coords = coords
        self.uniquelabel = '{}_{}'.format(frame, label)
        self.bbox = bbox
        self.track = None
        xcoords = list(zip(*coords))[0]
        ycoords = list(zip(*coords))[1]
        self.coords = (xcoords, ycoords)
        self.origin = origin
        self.validity = True
class Track:
    def __init__(self, nuc):
        self.nuclei = [nuc]
        nuc.track = self
        self.uniquelabel = str(nuc.frame) + ':' + str(nuc.label)
        self.color = np.array([random.sample(range(1,256), 1), random.sample(range(1,256), 1), random.sample(range(1,256), 1)]).T
        self.daughters = []
        self.site = site
        self.inmeanlist = []
        self.outmeanlist = []
        self.meanratiolist = []
    def pop(self):
        return self.nuclei.pop
    
    def append(self, nuc):
        if (nuc.track is None):
            nuc.track = self
            self.nuclei.append(nuc)
        else:
            print(f'{self.uniquelabel}doublematched')
            pass
    
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
    def dhbmvenus(self):
        self.inmeanlist = []
        self.outmeanlist = []
        self.meanratiolist = []
        self.totalintensitylist = []
        for nuc in self.nuclei:
            frame = nuc.frame
            site = self.site
            dapilab = io.imread(f'{temppath}dapilab_site{site}_frame{frame}.png')
            dhbim = io.imread('{}rat7_{:0>2}rat7_{}_20t{}.tif'.format(impath, site, site,frame))[:,:,2]
            dhbim = filters.gaussian(dhbim, 2)
            dhbim = solim.diskfilter(dhbim, 25)
            premask = np.zeros(dapilab.shape).astype('bool_')
            premask[nuc.coords] = True
            inmask = morph.binary_erosion(premask, morph.disk(2))
            outmask = morph.binary_dilation(premask, morph.disk(2))
            dapilab = (dapilab > 0) & (~premask)
            outmaskother = morph.binary_dilation(dapilab, morph.disk(2))
            outmask = outmask & (~premask) & (~outmaskother)
            inmean = measure.regionprops(inmask.astype('uint8'), dhbim)[0].mean_intensity
            self.inmeanlist.append(inmean)
            outmean = measure.regionprops(outmask.astype('uint8'), dhbim)[0].mean_intensity
            self.outmeanlist.append(outmean)
            meanratio = outmean/inmean
            self.meanratiolist.append(meanratio)
            totalmask = inmask | outmask
            totalintensity = np.sum(dhbim[solim.tuplist2coords(measure.regionprops(totalmask.astype('uint8'), dhbim)[0].coords)])
            self.totalintensitylist.append(totalintensity)
class Tracklist:
    def __init__(self):
        self.tracklist = {}
    def __iter__(self):
        return (track for track in self.tracklist.values())
    def append(self, track):
        self.tracklist[track.uniquelabel] = track
    def get(self, uniquelabel):
        return [t for t in self.tracklist if t.uniquelabel == uniquelabel][0]
trackdict = dict()
lapmatrixlist = []
solutionlist = []
#%%
'''
Get label images from each nucleus image in each frame
'''
nuclist = dict()
threshlist = []
for site in range(xno,xno+1):
    for frame in range(startframe, endframe+1):
        dapiimpath= '{}rat7_{:0>2}rat7_{}_20t{}.tif'.format(impath, site, site,frame)
        dapiim = io.imread(dapiimpath)[:,:,1]
        dapiim = filters.gaussian(dapiim)*65535
        '''d
        This step is to remove shadows from the background that are created by the
        H2b negative cardiomyocyte nuclei. If these shadows are not properly removed,
        during the succeeding background subtraction step they become dramatically highlighted
        and greatly confuse the algorithm, as background subtraction carries with it 
        the assumption that there are no sharp transitions of brightness in 
        the background.
        '''
        dapiim = solim.diskfilter(dapiim, diskrad)
        dapibin = dapiim > solim.relperimeterthresh(dapiim, 1)
        dapibin = morph.remove_small_objects(dapibin, minnucsize)
        dapilab = solim.tunewatershed(dapiim, binmask = dapibin, distance = 8, brightness = 90)
        dapilab = solim.segrandomizer(dapilab)
        '''
        Extract nucleus properties from each region in the label images
        '''
        nuclist[frame] = [Nuc(nuc.label, nuc.centroid, nuc.area, nuc.perimeter, nuc.mean_intensity, frame, nuc.coords, nuc.bbox)\
               for nuc in measure.regionprops(dapilab, dapiim)]
        io.imsave(f'{temppath}dapilab_site{site}_frame{frame}.png',dapilab)
        io.imsave(f'{temppath}dapiim_site{site}_frame{frame}.png',dapiim.astype('uint16'))
        print(frame)
#%%
#Initial births
tracklist = Tracklist()
for site in range(xno, xno+1):
    for nuc in nuclist[startframe]:
        tracklist.append(Track(nuc))
    for frame in range(startframe+1, endframe+1):
        pretracks = [track for track in tracklist if track.lastframe() == frame - 1]
        postnuclei = nuclist[frame]
        dapilab = io.imread(f'{temppath}dapilab_site{site}_frame{frame}.png')
        dapiim = io.imread(f'{temppath}dapiim_site{site}_frame{frame}.png')
        #Construct fusecompare
        fusecompare = np.zeros((len(postnuclei), len(pretracks)))
        for pretrack in pretracks:
            for postnuc in postnuclei:
                if solim.eucliddis(pretrack.nuclei[-1].centroid, postnuc.centroid) < distthresh:
                    fusecompare[postnuclei.index(postnuc), pretracks.index(pretrack)] = pretrack.nuclei[-1].area / postnuc.area
                else:
                    fusecompare[postnuclei.index(postnuc), pretracks.index(pretrack)] = 0
        for pretrack in pretracks:
            fusecomparerow = fusecompare[:, pretracks.index(pretrack)] > 1.8
            if np.sum(fusecomparerow) > 1:
                tofusenuclei = [postnuc for (postnuc, bo) in zip(postnuclei, fusecomparerow) if bo]
                if np.sum(solim.boundarycollector(dapilab, [tofusenuc.label for tofusenuc in tofusenuclei])) > 1:
                    fusemask = np.zeros(dapilab.shape).astype('uint8')
                    for tofusenuc in tofusenuclei:
                        fusemask[tofusenuc.coords] = tofusenuclei[0].label
                    for nuc in measure.regionprops(fusemask, dapiim):
                        fusednuc = Nuc(nuc.label, nuc.centroid, nuc.area, nuc.perimeter, nuc.mean_intensity, frame, nuc.coords, nuc.bbox, origin = 'fusion')
                    for tofusenuc in tofusenuclei:
                        fusecompare = np.delete(fusecompare, postnuclei.index(tofusenuc) , 0)
                        postnuclei.remove(tofusenuc)
                    pretrack.append(fusednuc)
                    fusecompare = np.delete(fusecompare, pretracks.index(pretrack), 1)
                    pretracks.remove(pretrack)
                else:
                    continue
        #Construct breakcompare
        breakcompare = np.zeros((len(postnuclei), len(pretracks)))
        for pretrack in pretracks:
            for postnuc in postnuclei:
                if solim.eucliddis(pretrack.nuclei[-1].centroid, postnuc.centroid) < distthresh:
                    breakcompare[postnuclei.index(postnuc), pretracks.index(pretrack)] = postnuc.area / pretrack.nuclei[-1].area
                else:
                    breakcompare[postnuclei.index(postnuc), pretracks.index(pretrack)] = 0
        for postnuc in postnuclei:
            breakcomparerow = breakcompare[postnuclei.index(postnuc), :] > 1.8
            if np.sum(breakcomparerow) > 1:
                toreassignnuclei = [pretrack for (pretrack, bo) in zip(pretracks, breakcomparerow) if bo]
                if np.sum(solim.boundarycollector(dapilab, [toreassignnuc.nuclei[-1].label for toreassignnuc in toreassignnuclei])):
                    continue
                #Distance Watershed
                distmask = np.zeros(dapilab.shape).astype('bool_')
                distmask[postnuc.coords] = True
                distim = nd.distance_transform_edt(distmask)
                distwat = solim.tunewatershed(distim, binmask=distmask, brightness = 10, distance = 0)
                #Centroid cut
                if np.max(distwat) < len(toreassignnuclei):
                    toreassigncoords = [] 
                    for i in range(len(toreassignnuclei)):
                        toreassigncoords.insert(0, [])
                    toreassigncentroids = [toreassignnuc.nuclei[-1].centroid for toreassignnuc in toreassignnuclei]
                    for postnuccoord in zip(postnuc.coords[0], postnuc.coords[1]):
                        postnucdistances = [solim.eucliddis(postnuccoord, toreassigncentroid) for toreassigncentroid in toreassigncentroids]
                        toreassigncoords[np.argmin(postnucdistances)].append(postnuccoord)
                    for i, nn in enumerate(toreassigncoords):
                        toreassigncoords[i] = solim.tuplist2coords(toreassigncoords[i])
                    cutmask = np.zeros(dapilab.shape).astype('uint8')
                    for i, nn in enumerate(toreassignnuclei):
                        cutmask[toreassigncoords[i]] = np.max(dapilab) + i
                    brokennuclei = []
                    for nuc in measure.regionprops(cutmask, dapiim):
                        brokennuclei.append(Nuc(nuc.label, nuc.centroid, nuc.area, nuc.perimeter, nuc.mean_intensity, frame, nuc.coords, nuc.bbox, origin = 'cbroken'))
                    for brokennuc in brokennuclei:
                        brokendistances = [solim.eucliddis(brokennuc.centroid, toreassignnuc.nuclei[-1].centroid) for toreassignnuc in toreassignnuclei]
                        toreassignnuclei[np.argmin(brokendistances)].append(brokennuc)
                        pretracks.remove(toreassignnuclei[np.argmin(brokendistances)])
                        toreassignnuclei.remove(toreassignnuclei[np.argmin(brokendistances)])
                #Distance Watershed Continue
                else:
                    brokennuclei = []
                    for nuc in measure.regionprops(distwat, dapiim):
                        brokennuclei.append(Nuc(nuc.label, nuc.centroid, nuc.area, nuc.perimeter, nuc.mean_intensity, frame, nuc.coords, nuc.bbox, origin = 'dbroken'))
                    if len(brokennuclei) > len(toreassignnuclei):
                        excess = len(brokennuclei) > len(toreassignnuclei)
                        fusemask = np.zeros(dapilab.shape)
                        arealist = [brokennuc.area for brokennuc in brokennuclei]
                        tofusenuclei = [brokennuclei[i] for i in np.argsort(arealist)[-excess:]]
                        for tofusenuc in tofusenuclei:
                            brokennuclei.remove(tofusenuc)
                        for tofusenuc in tofusenuclei:
                            tofuseddareas = [brokennuc.area for brokennuc in brokennuclei]
                            adherent = brokennuclei[np.argmin(tofuseddareas)]
                            tofusemask = np.zeros(dapilab.shape).astype('uint8')
                            tofusemask[tofusenuc.coords] = adherent.label
                            tofusemask[adherent.coords] = adherent.label
                            for nuc in measure.regionprops(tofusemask, dapiim):
                                brokennuclei.append(Nuc(nuc.label, nuc.centroid, nuc.area, nuc.perimeter, nuc.mean_intensity, frame, nuc.coords, nuc.bbox, origin = 'rebroken'))
                            brokennuclei.remove(adherent)
                    for brokennuc in brokennuclei:
                        brokendistances = [solim.eucliddis(brokennuc.centroid, toreassignnuc.nuclei[-1].centroid) for toreassignnuc in toreassignnuclei]
                        toreassignnuclei[np.argmin(brokendistances)].append(brokennuc)
                        pretracks.remove(toreassignnuclei[np.argmin(brokendistances)])
                        toreassignnuclei.remove(toreassignnuclei[np.argmin(brokendistances)])
                breakcompare = np.delete(breakcompare, postnuclei.index(postnuc), 0)
                for toreassignnuc in toreassignnuclei:
                    breakcompare = np.delete(breakcompare, pretracks.index(toreassignnuc) ,1)
                    pretracks.remove(toreassignnuc)
                postnuclei.remove(postnuc)
        #Lapmatrix construction
        lapmatrix = np.zeros((len(postnuclei), len(pretracks)))
        for pretrack in pretracks:
            for postnuc in postnuclei:
                distance = solim.eucliddis(postnuc.centroid, pretrack.nuclei[-1].centroid)
                if distance < distthresh:
                    brightnessdissimilarity = max(postnuc.mean_intensity/ pretrack.nuclei[-1].mean_intensity, pretrack.nuclei[-1].mean_intensity / postnuc.mean_intensity)
                    lapmatrix[postnuclei.index(postnuc), pretracks.index(pretrack)] = distance * brightnessdissimilarity
                else:
                    lapmatrix[postnuclei.index(postnuc), pretracks.index(pretrack)] = 200
        deleterows = []
        deletecols = []
        for i, lrow in enumerate(lapmatrix):
            if np.all(lrow == 200):
                deleterows.append(i)
        for i, lcol in enumerate(lapmatrix.T):
            if np.all(lcol == 200):
                deletecols.append(i)
        lapmatrix = np.delete(lapmatrix, deleterows, 0)
        lapmatrix = np.delete(lapmatrix, deletecols, 1)
        unmatchable = []
        for i in reversed(sorted(deleterows)):
            unmatchable.append(postnuclei.pop(i))
        for i in reversed(sorted(deletecols)):
            pretracks.pop(i)
        #Lapmatrix assignment
        solution = linear_sum_assignment(lapmatrix)
        for i, j in zip(solution[0], solution[1]):
            sol1 = pretracks[j]
            sol2 = postnuclei[i]
            if sol2.area/sol1.nuclei[-1].area > 1.3:
                distmask = np.zeros(dapilab.shape).astype('bool_')
                distmask[sol2.coords] = True
                distim = nd.distance_transform_edt(distmask)
                distwat = solim.tunewatershed(distim, binmask = distmask, brightness = 10, distance = 0) + (sol2.label - 1)
                cutnuclei = []
                for nuc in measure.regionprops(distwat, dapiim):
                    cutnuclei.append(Nuc(nuc.label, nuc.centroid, nuc.area, nuc.perimeter, nuc.mean_intensity, frame, nuc.coords, nuc.bbox))
                cutdistances = [solim.eucliddis(cutnuc.centroid, sol1.nuclei[-1].centroid) for cutnuc in cutnuclei]
                sol1.append(cutnuclei[np.argmin(cutdistances)])
                sol2.validity = False
            else:
                sol1.append(sol2)
        for postnuc in (postnuclei + unmatchable):
            if (postnuc.track is None) and postnuc.validity:
                postnuc.origin = 'birth'
                tracklist.append(Track(postnuc))
        print(frame)
        
                        
    trackdict[site] = tracklist
    toiist = [track for track in tracklist if len(track.nuclei) > 0]
    for frame in range(startframe, endframe+1):
        dapiimpath= '{}rat7_{:0>2}rat7_{}_20t{}.tif'.format(impath, site, site,frame)
        dapiim = io.imread(dapiimpath)[:, :, 1]
        nuctrackim = np.zeros([dapiim.shape[0], dapiim.shape[1], 3])
        greydapiim = np.copy(nuctrackim)
        greydapiim[:, :, 2] = dapiim
        for toi in toiist:
            if toi.firstframe() <= frame <= toi.lastframe():
                nuc = toi.getframe(frame)
                nuctrackim[nuc.coords] = toi.color
        nucbordim = seg.find_boundaries(nuctrackim[:,:,0])
        nucbordim = np.stack([nucbordim, nucbordim, nucbordim], axis = 2)
        nuctrackim = np.where(nucbordim, nuctrackim, greydapiim).astype('uint8')
        nuctrackim = Image.fromarray(nuctrackim)
        d = ImageDraw.Draw(nuctrackim)
        for toi in toiist:
            if toi.firstframe() <= frame <= toi.lastframe():
                nuc = toi.getframe(frame)
                d.text((nuc.centroid[1], nuc.centroid[0]), toi.uniquelabel + ' ' + nuc.origin, fill = tuple(toi.color[0]))
        nuctrackim.save(f'{temppath}nuctrackim_site{site}_frame{frame}.png')
    #filetext = fil 'tracklabel \t area \t coordinates \t centroid \t frame\t{site} \n'
    

            

    