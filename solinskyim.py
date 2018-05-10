import numpy as np
from skimage import feature, filters, measure,  morphology as morph
from math import sqrt
from scipy.optimize import linear_sum_assignment
import cv2
import matplotlib
import random
distancethresh = 10
simthresh = 1





def _middler(a):
    abeg = a[:-1]
    aend = a[1:]
    retval = np.zeros(len(a)-1)
    k = 0
    for i, j in zip(abeg, aend):
        retval[k] = i + abs(i-j)/2
        k += 1
    return retval

#Image display finctions

def randcmap():
    return matplotlib.colors.ListedColormap(np.random.rand(256,3))

def segrandomizer(array):
    choice = list(range(1, np.amax(array) + 1))
    choice = [0] + random.sample(choice, len(choice))
    for i, n in np.ndenumerate(array):
        array[i] = choice[n]
    return array






#Image analysis functions
    #FIX THESE TO HANDLE OVERFLOW AND UNDERFLOW ERRORS, ESPECIALLY FOR THE UINT16 DATATYPE!
def hmaxima(image, brightness):
    image = image.astype('int64')
    mask = image + brightness
    return morph.reconstruction(image, mask)

def regmaxima(image, brightness):
    image = image.astype('int64')
    mask = image + brightness
    compare = morph.reconstruction(image, mask)
    return mask - compare

def watershedseed(image, brightness, distance, binmask = None):
    maxima = regmaxima(image, brightness) > (brightness - 1)
    if binmask is not None:
        maxima = maxima &  binmask
    maxima = morph.binary_dilation(maxima, selem = morph.disk(distance))
    maxima = measure.label(maxima)
    return maxima

def tunewatershed(image, brightness, distance, binmask = None):
    maxima = regmaxima(image, brightness) > (brightness - 1)
    maxima = morph.binary_dilation(maxima, selem = morph.disk(distance))
    maxima = measure.label(maxima)
    return morph.watershed(-image, maxima, mask = binmask)

def seedwatershed(image, seeds, binmask = None):
    return morph.watershed(-image, seeds, mask = binmask)

def diskfilter(image, radius):
    disk = morph.disk(radius)
    return(cv2.morphologyEx(image, cv2.MORPH_TOPHAT, disk))
    
    
#Customized threshold functions
    
def relperimeterthresh(image):
    hithresh = int(filters.threshold_otsu(image))
    lothresh = int(filters.threshold_triangle(image))
    threshqual = []
    threshsearch = []
    for thresh in range(lothresh, hithresh, (hithresh-lothresh)//40):
        sample = image > thresh
        threshqual.append((measure.perimeter(sample) ** 2)/np.sum(sample))
        threshsearch.append(thresh)
    goodthresh = threshsearch[np.argmin(threshqual)]
    return goodthresh

def histtriangle(hist, nbins = 256):
    #Takes images or image histograms as inputs
    if hist.__class__ == tuple:
        bin_edges = hist[1]
        hist = hist[0]
    else:
        histthing = np.histogram(hist, bins = nbins)
        hist = histthing[0]
        bin_edges = histthing[1]
    bin_centers = _middler(bin_edges)
    # Find peak, lowest and highest gray levels.
    arg_peak_height = np.argmax(hist)
    peak_height = hist[arg_peak_height]
    arg_low_level, arg_high_level = np.where(hist>0)[0][[0, -1]]

    # Flip is True if left tail is shorter.
    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = hist[::-1]
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1

    # If flip == True, arg_high_level becomes incorrect
    # but we don't need it anymore.
    del(arg_high_level)

    # Set up the coordinate system.
    width = arg_peak_height - arg_low_level
    x1 = np.arange(width)
    y1 = hist[x1 + arg_low_level]

    # Normalize.
    norm = np.sqrt(peak_height**2 + width**2)
    peak_height /= norm
    width /= norm

    # Maximize the length.
    # The ImageJ implementation includes an additional constant when calculating
    # the length, but here we omit it as it does not affect the location of the
    # minimum.
    length = peak_height * x1 - width * y1
    arg_level = np.argmax(length) + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    return bin_centers[arg_level]

###################################################################################################################


class Pairassessment:
    def __init__(self, nucleus1, nucleus2):
        self.nucleus1 = nucleus1
        self.nucleus2 = nucleus2
        self.cost = sqrt((nucleus1.x - nucleus2.x) ** 2 + \
                         (nucleus1.y - y) ** 2)
def lapmatrix(frame1, frame2):
        '''
        frame1 is the list of nuclei in frame 1,
        frame 2 is the list of nuclei in frame 2
        '''
        costmatrix = np.zeros((2*len(frame1), 2*len(frame2))).astype('uint8')
        for ino, i in enumerate(frame1):
            for jno, j in enumerate(frame2):
                thisassesment = Pairassessment(i, j)
                if thisassesment.cost < 10:
                    costmatrix[ino, jno] = thisassesment.cost
                else:
                    costmatrix[ino, jno] = 13
        costmatrix[len(frame1):2*len(frame1), 0:len(frame2)] = 11
        costmatrix[0:len(frame1), len(frame2):2*len(frame2)] = 11
        costmatrix[len(frame1): 2*len(frame1), frame2:2*len(frame2)] = 12
        lsarow, lsacol = linear_sum_assignment(costmatrix) 
        lsarow[lsarow >= len(frame1)] = len(frame1)
        lsacol[lsacol >= len(frame2)] = len(frame2)
        frame1.append(None)
        frame2.append(None)
        return  [[frame1[i], frame2[j]] for i, j in zip(lsarow, lsacol)]
'''
There will be no gap closing, nuclei aren't seen to dissapear for a frame
For each apparent fusion, attempt to resegment the nuclei using the locations of the
last known centroids as seeds for the watershed algorithm. Load the shape of the fusion nucleus into memory.
'''
def recordbirths(tracks):
    #Birth Identification
    for i in tracks:
        birthdistances = np.zeros(len(tracks)).fill(distancethresh+1)
        birthsimilarities = np.array(len(tracks)).fill(simthresh+1)
        inuc = i[0]
        for jno, j in enumerate(tracks):
            try:
                jnuc = j.getframe(inuc.frame)

            except:
                continue
            else:
                birthdistances[jno] = ((j.x - i.x) ** 2 + (j.y - i.y) ** 2) ** 0.5
                birthsimilarities = inuc.integrated_intensity / jnuc.integrated_intensity
                candidate = np.argmin(birthdistances)
                if min(birthdistances) < distancethresh and birthsimilarities[candidate] < simthresh: #similarity threshold
                    j.mitosis(i)
def findoverlap(tracks):
    #Overlap identification
    for i in tracks:
        overlapdistances = np.zeros(len(tracks)).fill(distancethresh+1)
        overlapadds = np.array(len(tracks)).fill(simthresh+1)
        overlapnuc = i[-1]
        fusenuclist = []
        for jno, j in enumerate(tracks):
            try:
                jnuc = j.getframe(overlapnuc.frame)
                fusenuc = j.getframe(overlapnuc.frame + 1)
            except:
                continue
            else:
                overlapdistances[jno] = ((jnuc.x - i.x) ** 2 + (jnuc.y - i.y) ** 2) ** 0.5
                overlapsimilarities[jno] = fusenuc.integrated_intensity / (i.integrated_intensity + jnuc.integrated_intensity)
                candidate = np.argmin(overlapdistances)
                if min(overlapdistances) < distancethresh and overlapsimilarities[candidate] < simthresh:
                    fusenuclist.append(fusenuc)
        return(fusenuc)           
                
                
            
            
            
            