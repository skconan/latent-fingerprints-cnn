"""
    File name: utilities.py
    Author: skconan
    Date created: 2019/10/12
    Python Version: 3.7
"""


import cv2 as cv
import os
import numpy as np

def get_file_path(dir_name):
    """
      Get all files in directory "dir_name"
    """
    file_list = os.listdir(dir_name)
    files = []
    for f in file_list:
        abs_path = os.path.join(dir_name, f)
        if os.path.isdir(abs_path):
            files = files + get_file_path(abs_path)
        else:
            files.append(abs_path)
    return files
  
  
def get_file_name(img_path):
    img_path = img_path.replace('\\','/')
    
    name = img_path.split('/')[-1]

    name = name.replace('.gif', '')
    name = name.replace('.png', '')
    name = name.replace('.jpg', '')
    return name


def normalize(image, max=255, input_max=None, input_min=None):
    if input_max is not None:
        result = 255.*(image - input_min)/(input_max-input_min)
    else:
        result = 255.*(image - image.min())/(image.max()-image.min())
    result = np.uint8(result)
    return result

def implot(figname, image):
    f= plt.figure(figname)
    plt.imshow(image)
    plt.show()
    cv.waitKey(-1)
    plt.close()    

def imshow(name, mat, mapping=False):
    if len(mat.shape) < 3:
        r,c = mat.shape
    else:
        r,c,_ = mat.shape

    if mapping:
        mat = cv.applyColorMap(mat, cv.COLORMAP_HOT)

    if r < 100:
        mat = cv.resize(mat,None,fx=4,fy=4)
    cv.imshow(name,mat)