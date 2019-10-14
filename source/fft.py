import cv2 as cv
import numpy as np
from utilities import *
from matplotlib import pyplot as plt

def spatial2freq(img):
    spectrum = np.fft.fft2(img)
    spectrum = np.fft.fftshift(spectrum)
    return spectrum


def freq2spatial(spectrum):
    spectrum  = np.fft.ifftshift(spectrum )
    img  = np.fft.ifft2(spectrum )
    img = get_magnitude(img)
    img = normalize(img)
    return img

def get_magnitude(spectrum):
    magnitude = np.abs(spectrum)
    return magnitude

def get_phase(spectrum):
    return np.angle(spectrum)

def to_logscale(img):
    return np.log(1+img)

