import re
import os
import itertools as it
import cv2
import types
import pickle
import random
import imageio
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from skimage import io, measure, morphology as morph, segmentation as seg
from matplotlib import pyplot as plt
from . import solinskyim as solim
def focusmeasure(image):
    return cv2.Laplacian(cv2.GaussianBlur(image, (5,5), 0), cv2.CV_64F).var()
    
class Imageset(types.SimpleNamespace):
    def __init__(self, imagefolder, platename, rowletters, colnumbers, sitenumbers,
                 dapich, dapichno, gfpch, gfpchno,
                 rfpch, rfpchno, imagenameformat):
        #expects input path/images/imagefolder
        self.imagefolder = imagefolder
        self.platename = platename
        self.rowletters = rowletters
        #give colnumbers as a range
        self.colnumbers = list(colnumbers)
        #give sitenumbers as a range
        self.sitenumbers = list(sitenumbers)
        self.branchfolder = re.search('(.*)(?:/images/.*/[^/]+$)', self.imagefolder).group(1)
        self.tempfolder = f'{self.branchfolder}/temp/{platename}'
        if not os.path.exists(self.tempfolder):
            os.makedirs(self.tempfolder)
        self.resultfolder = f'{self.branchfolder}/result/{platename}'
        if not os.path.exists(self.resultfolder):
            os.makedirs(self.resultfolder)
        self.dapich = dapich
        self.dapichno = dapichno
        self.gfpch = gfpch
        self.gfpchno = gfpchno
        self.rfpch = rfpch
        self.rfpchno = rfpchno
        self.imagenameformat = imagenameformat
        self.dapifocuspath = f'{self.resultfolder}/dapifocusdata.npy'
        if not os.path.exists(self.dapifocuspath):
            Path(self.dapifocuspath).touch()
        self.gfpfocuspath = f'{self.resultfolder}/gfpfocusdata.npy'
        if not os.path.exists(self.gfpfocuspath):
            Path(self.gfpfocuspath).touch()
        self.nucleusdatapath = f'{self.resultfolder}/nucleusdata.pickle'
        if not os.path.exists(self.nucleusdatapath):
            Path(self.nucleusdatapath).touch()
        self.gfpareadatapath = f'{self.resultfolder}/gfpareadata.pickle'
        if not os.path.exists(self.gfpareadatapath):
            Path(self.gfpareadatapath).touch()
        self.size = len(self.rowletters) * len(self.colnumbers) * len(self.sitenumbers)
    @property
    def dapifocusdata(self):
        return np.load(f'{self.dapifocuspath}')
    @dapifocusdata.setter
    def dapifocusdata(self, value):
        np.save(f'{self.dapifocuspath}', value)
    @property
    def gfpfocusdata(self):
        return np.load(f'{self.gfpfocuspath}')
    @gfpfocusdata.setter
    def gfpfocusdata(self, value):
        np.save(f'{self.gfpfocuspath}', value)
    @property
    def nucleusdata(self):
        with open(f'{self.nucleusdatapath}', 'rb') as pik:
            return pickle.load(pik)
    @nucleusdata.setter
    def nucleusdata(self, value):
        with open(self.nucleusdatapath, 'wb') as pik:
            pickle.dump(value, pik)
    @property
    def gfpareadata(self):
        with open(f'{self.gfpareadatapath}', 'rb') as pik:
            return pickle.load(pik)
    @gfpareadata.setter
    def gfpareadata(self, value):
        with open(self.gfpareadatapath, 'wb') as pik:
            pickle.dump(value, pik)
        
    
    def getimage(self, row, col, site, channelname='', channelno=''):
        return io.imread(f'{self.imagefolder}/' + self.imagenameformat.format(row=row,
                                           col=col,
                                           site=site,
                                           channelname=channelname,
                                           channelno=channelno))
    def imagegenerator(self, channelname, channelno, filtering=False):
        index = 0
        if filtering:
            dapifocusdata = self.dapifocusdata
            gfpfocusdata = self.gfpfocusdata
            for row, col, site in it.product(self.rowletters, self.colnumbers, self.sitenumbers):
                if dapifocusdata[index] >self.dapifocusthresh and gfpfocusdata >self.gfpfocusthresh:
                    yield self.getimage(row, col, site, channelname=channelname, channelno = channelno)
                    index += 1
        else:
            for row, col, site in it.product(self.rowletters, self.colnumbers, self.sitenumbers):
                yield self.getimage(row, col, site, channelname=channelname, channelno = channelno)
    def dapigenerator(self, filtering=False):
        return self.imagegenerator(self.dapich, self.dapichno, filtering)
    def gfpgenerator(self, filtering=False):
        return self.imagegenerator(self.gfpch, self.gfpchno, filtering)
    def rfpgenerator(self, filtering=False):
        return self.imagegenerator(self.rfpch, self.rfpchno, filtering)
    def sitesetgenerator(self):
        for row, col, site in it.product(self.rowletters, self.colnumbers, self.sitenumbers):
            siteset = {}
            siteset['dapiim'] = self.getimage(row, col, site, self.dapich, self.dapichno)
            siteset['gfpim'] = self.getimage(row, col, site, self.gfpch, self.gfpchno)
            siteset['rfpim'] = self.getimage(row, col, site, self.rfpch, self.rfpchno)
            siteset['row'] = row
            siteset['col'] = col
            siteset['site'] = site
            yield siteset
    
    def getimageatindex(self, index, channelname = '', channelno = ''):
        site = self.sitenumbers[index % len(self.sitenumbers)]
        col = self.colnumbers[(index % (len(self.sitenumbers) * len(self.colnumbers))) // len(self.sitenumbers)]
        row = self.rowletters[index // (len(self.sitenumbers) * len(self.colnumbers))]
        return self.getimage(row, col, site, channelname, channelno)

    def collectfocusdata(self):
        def focusmeasure(image):
            return cv2.Laplacian(cv2.GaussianBlur(image, (5,5), 0), cv2.CV_64F).var()
        self.dapifocusdata = Parallel(n_jobs = 4, verbose=5)(delayed(focusmeasure)(image) for image in self.dapigenerator())
        self.gfpfocusdata = Parallel(n_jobs = 4, verbose=5)(delayed(focusmeasure)(image) for image in self.gfpgenerator())
        
    def plotfocusdata(self):
        dapifocusdata = self.dapifocusdata
        gfpfocusdata = self.gfpfocusdata
        dno = [np.sum(dapifocusdata < d) for d in range(0, 1000000, 100)]
        plt.figure(1)
        plt.title('dapi focus measure')
        plt.plot(list(range(0,1000000,100)), dno)
        gno = [np.sum(gfpfocusdata < d) for d in range(0, 1000000, 100)]
        plt.figure(2)
        plt.title('gfp focus measure')
        plt.plot(list(range(0,1000000,100)), gno)
        
    def setfocusthreshes(self):
        self.dapifocusthresh = float(input('enter dapi focus threshold: '))

    def collectbackgrounddata(self):
        def backgroundmedian(mask):
            inspectee = mask
            mask = mask < self.gfpthresh
            return np.median(inspectee[mask])
        self.gfpbacklist = Parallel(n_jobs = 4, verbose=5)(delayed(np.median)(gfpim[gfpim < self.gfpthresh])\
                                    for  gfpim in self.gfpgenerator())
        self.rfpbacklist = Parallel(n_jobs = 4, verbose=5)(delayed(np.median)(rfpim[gfpim < self.gfpthresh]) \
                                    for  gfpim, rfpim in zip(self.gfpgenerator(), self.rfpgenerator()))
    
    def setbackgroundthreshes(self):
        self.gfpback = float(input('enter gfp background threshold: '))
        self.rfpback = float(input('enter rfp background threshold: '))
    
    def collectratiodata(self):
        self.rogratiolist = Parallel(n_jobs = 4, verbose=5) \
        (delayed(np.median)((rfpim-self.rfpback)[gfpim<self.gfpthresh] / (gfpim-self.gfpback)[gfpim<self.gfpthresh]) \
         for gfpim, rfpim in zip(self.gfpgenerator(), self.rfpgenerator()))

    def setrogratio(self):
        self.rogratio = float(input('enter red over green ratio: '))
        
    def collectgfpthreshdata(self):
        self.gfpthreshlist = Parallel(n_jobs=4, verbose=5) \
        (delayed(solim.relperimeterthresh)(gfpim) \
        for gfpim in self.gfpgenerator())
        
    def collectdapithreshdata(self):
        self.dapithreshlist = Parallel(n_jobs=4, verbose=5) \
        (delayed(solim.relperimeterthresh)(dapiim, 2) \
        for dapiim in self.dapigenerator())

    def setgfpthreshdata(self):
        self.gfpthresh = float(input("enter gfp threshold: "))
    def setdapithreshdata(self):
        self.dapithresh = float(input("enter dapi threshold: "))
    
    def collectnucleusdata(self):
            def inspectsite(siteset):
                try:
                    dapiim = siteset['dapiim'].astype('int')
                    gfpim = siteset['gfpim'].astype('int')
                    rfpim = siteset['rfpim'].astype('int')
                    rfpim = (rfpim - np.percentile(rfpim, 5))
                    row = siteset['row']
                    col = siteset['col']
                    site = siteset['site']
                    dapibin = dapiim > self.dapithresh
                    gfpbin = gfpim > solim.relperimeterthresh(gfpim)
                    gfparea = np.sum(gfpbin)
                    gfpbin = morph.remove_small_holes(gfpbin, 200)
                    dapibin = seg.clear_border(dapibin)
                    dapilab = measure.label(dapibin)
                    io.imsave(f'{self.tempfolder}/{row}{col}site{site}dapilab.png', dapilab)
                    dapiim = dapiim - np.percentile(dapiim, 5)
                    dapiprops = measure.regionprops(dapilab, dapiim)
                    framearray = np.array([[
                      nuc.label,
                      np.sum(dapiim[[nuc.coords.T[0], nuc.coords.T[1]]]),
                      nuc.eccentricity,
                      ((nuc.perimeter ** 2 / 4 * 3.14) / nuc.area),
                      nuc.area,
                      np.percentile(rfpim[[nuc.coords.T[0], nuc.coords.T[1]]], 95),
                      np.min(gfpbin[[nuc.coords.T[0], nuc.coords.T[1]]])
                      ] for nuc in dapiprops])
                    framearray = framearray.T
                    integrated_dapi, eccentricity, form_factor, area, rfp_peak, gfp_pos = \
                    framearray [1:,:]
                    gfp_area = [gfparea] * len(area)
                    nuclabels = framearray[0,:]
                    indexrow = (row, col, site)
                    labelmultiindex = [indexrow + (nuc,) for nuc in nuclabels]
                    labelmultiindex = pd.MultiIndex.from_tuples(labelmultiindex, names=['row', 'col', 'site', 'label'])
                    sitenwf = pd.DataFrame({'integrated_dapi':integrated_dapi, 'eccentricity':eccentricity, 'form_factor':form_factor,
                                           'area':area, 'rfp_peak':rfp_peak, 'gfp_pos':gfp_pos, 'gfp_area':gfp_area},
                                            index = labelmultiindex)
                    return sitenwf
                except:
                    return None
            nwflist = Parallel(n_jobs = 4, verbose=5)(delayed(lambda x: x)(inspectsite(siteset)) for siteset in self.sitesetgenerator())
            nwflist = [nwf for nwf in nwflist if nwf is not None]
            finalnwf = pd.concat(nwflist)
            self.nucleusdata = finalnwf
    def dapilabfinder(self, row, col, site):
        return io.imread(f'{self.tempfolder}/{row}{col}site{site}dapilab.png')
    def printquery(self, querystrings, names, rows, cols, sites):
        nucleusdata = self.nucleusdata
        name = '-'.join(names)
        queryresults = [nucleusdata.query(querystring) for querystring in querystrings]
        for row,col,site in it.product(rows, cols, sites):
            subqueries = []
            for queryresult in queryresults:
                subqueries.append(queryresult.query('row == @row and col == @col and site == @site'))
            dapilab = self.dapilabfinder(row, col, site)
            labshape = dapilab.shape
            queryimages = [seg.mark_boundaries(np.zeros(labshape), np.in1d(dapilab.ravel(), subquery.index.get_level_values('label')).reshape(labshape))[:,:,0] \
                           for subquery in subqueries]
            filename = f'{self.resultfolder}/{row}{col}site{site}{name}.tiff'
            dapiim = self.getimage(row, col, site, self.dapich, self.dapichno) // 256
            gfpim = self.getimage(row, col, site, self.gfpch, self.gfpchno) // 256
            rfpim = self.getimage(row, col, site, self.rfpch, self.rfpchno) // 256
            imlist = [dapiim, gfpim, rfpim] + queryimages
            fullimage = np.stack(imlist).astype('uint8')
            fullimage[3:] = 255*fullimage[3:]
            imageio.mimsave(filename, fullimage)
            os.system(f'''open -a "fiji" "{filename}" ''')
    def printrfpcsv(self, querystrings, names, groupby):
        nwf = self.nucleusdata
        querystrings = ['validity == 1 and site in [9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,27,30,31,32,33,34,35,37,38,39,40,41,44,45,46,47,48] and ' + querystring for querystring in querystrings]
        for querystring, name in zip(querystrings, names):
            datafraction = nwf.query(querystring)
            printcsv = datafraction.groupby(groupby).size().unstack()
            filename = f'{self.resultfolder}/{name}.csv'
            printcsv.to_csv(filename)
class Channel:
    def __init__(self, name, number, threshmethod = solim.relperimeterthresh, backgroundsubtractmethod = None):
        self.name = name
        self.number = number
        self.threshmethod = threshmethod
        self.threshmethod = threshmethod
        self.backgroundsubtractmethod = backgroundsubtractmethod
    def backgroundsubtract(self, image):
        if self.backgroundsubtractmethod:
            return self.backgroundsubtractmethod(image)
        else:
            return image
    def thresh(self, image):
        if self.backgroundsubtractmethod:
            image = self.backgroundsubtract(image)
        return image > self.threshmethod(image)
        
            
        
        
        
        

    