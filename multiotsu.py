#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:55:58 2018

@author: Lab
"""
import numpy as np
from skimage import dtype_limits

def threshold_multiotsu(image, nclass=3, nbins=255):
    """Generates multiple thresholds for an input image. Based on the
    Multi-Otsu approach by Liao, Chen and Chung.

    Parameters
    ----------
    image : (N, M) ndarray
        Grayscale input image.
    nclass : int, optional
        Number of classes to be thresholded, i.e. the number of resulting
        regions. Accepts an integer from 2 to 5. Default is 3.
    nbins : int, optional
        Number of bins used to calculate the histogram. Default is 255.

    Returns
    -------
    idx_thresh : (nclass) array
        Array containing the threshold values for the desired classes.
    max_sigma : float
        Maximum sigma value achieved on the classes.

    References
    ----------
    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
    multilevel thresholding", Journal of Information Science and
    Engineering 17 (5): 713-727, 2001. Available at:
    http://www.iis.sinica.edu.tw/page/jise/2001/200109_01.html
    .. [2] Tosa, Y., "Multi-Otsu Threshold", a java plugin for ImageJ.
    Available at:
    http://imagej.net/plugins/download/Multi_OtsuThreshold.java

    Examples
    --------
    >>> from skimage import data
    >>> image = data.camera()
    >>> thresh = threshold_multiotsu(image)
    >>> region1 = image <= thresh[0]
    >>> region2 = (image > thresh[0]) & (image <= thresh[1])
    >>> region3 = image > thresh[1]
    """
    if image.shape[-1] in (3, 4):
        raise TypeError("The input image seems to be RGB (shape: {0}. Please"
                        "use a grayscale image.".format(image.shape))

    if image.min() == image.max():
        raise TypeError("The input image seems to have only one color: {0}."
                        "Please use a grayscale image.".format(image.min()))

    # check if nclass is between 2 and 5.
    if nclass not in np.array((2, 3, 4, 5)):
        raise ValueError("Please choose a number of classes between "
                         "2 and 5.")

    # receiving minimum and maximum values for the image type.
    type_min, type_max = dtype_limits(image)
    # calculating the histogram and the probability of each gray level.
    hist, _ = np.histogram(image.ravel(), bins=nbins,
                           range=(type_min, type_max))
    prob = hist / image.size

    max_sigma = 0
    momP, momS, var_btwcls = [np.zeros((nbins, nbins)) for n in range(3)]

    # building the lookup tables.
    # step 1: calculating the diagonal.
    for u in range(1, nbins):
        momP[u, u] = prob[u]
        momS[u, u] = u * prob[u]

    # step 2: calculating the first row.
    for u in range(1, nbins-1):
        momP[1, u+1] = momP[1, u] + prob[u+1]
        momS[1, u+1] = momS[1, u] + (u+1)*prob[u+1]

    # step 3: calculating the other rows recursively.
    for u in range(2, nbins):
        for v in range(u+1, nbins):
            momP[u, v] = momP[1, v] - momP[1, u-1]
            momS[u, v] = momS[1, v] - momS[1, u-1]

    # step 4: calculating the between class variance.
    for u in range(1, nbins):
        for v in range(u+1, nbins):
            if (momP[u, v] != 0):
                var_btwcls[u, v] = momS[u, v]**2 / momP[u, v]
            else:
                var_btwcls[u, v] = 0

    # finding max threshold candidates, depending on nclass.
    # number of thresholds is equal to number of classes - 1.
    if nclass == 2:
        for idx in range(1, nbins - nclass):
            part_sigma = var_btwcls[1, idx] + var_btwcls[idx+1, nbins-1]
            if max_sigma < part_sigma:
                aux_thresh = idx
                max_sigma = part_sigma

    elif nclass == 3:
        for idx1 in range(1, nbins - nclass):
            for idx2 in range(idx1+1, nbins - nclass+1):
                part_sigma = var_btwcls[1, idx1] + \
                            var_btwcls[idx1+1, idx2] + \
                            var_btwcls[idx2+1, nbins-1]

                if max_sigma < part_sigma:
                    aux_thresh = idx1, idx2
                    max_sigma = part_sigma

    elif nclass == 4:
        for idx1 in range(1, nbins - nclass):
            for idx2 in range(idx1+1, nbins - nclass+1):
                for idx3 in range(idx2+1, nbins - nclass+2):
                    part_sigma = var_btwcls[1, idx1] + \
                                var_btwcls[idx1+1, idx2] + \
                                var_btwcls[idx2+1, idx3] + \
                                var_btwcls[idx3+1, nbins-1]

                    if max_sigma < part_sigma:
                        aux_thresh = idx1, idx2, idx3
                        max_sigma = part_sigma

    elif nclass == 5:
        for idx1 in range(1, nbins - nclass):
            for idx2 in range(idx1+1, nbins - nclass+1):
                for idx3 in range(idx2+1, nbins - nclass+2):
                    for idx4 in range(idx3+1, nbins - nclass+3):
                        part_sigma = var_btwcls[1, idx1] + \
                            var_btwcls[idx1+1, idx2] + \
                            var_btwcls[idx2+1, idx3] + \
                            var_btwcls[idx3+1, idx4] + \
                            var_btwcls[idx4+1, nbins-1]

                        if max_sigma < part_sigma:
                            aux_thresh = idx1, idx2, idx3, idx4
                            max_sigma = part_sigma

    # correcting values according to minimum and maximum values.
    idx_thresh = np.asarray(aux_thresh) * (type_max-type_min) / nbins

    return idx_thresh, max_sigma