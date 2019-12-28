"""
    File name: utilities.py
    Author: skconan
    Date created: 2019/10/12
    Python Version: 3.7
"""


import cv2 as cv
import os
import numpy as np
# import tensorflow as tf
import colorama
from PIL import Image
from keras.preprocessing import image

colorama.init()
DEBUG = True


def print_debug(*args, **kwargs):
    global DEBUG
    text = ""
    if not "mode" in kwargs:
        mode = "DETAIL"
    else:
        mode = kwargs['mode']
    color_mode = {
        "METHOD": colorama.Fore.BLUE,
        "RETURN": colorama.Fore.GREEN,
        "DETAIL": colorama.Fore.YELLOW,
        "DEBUG": colorama.Fore.RED,
        "END": colorama.Style.RESET_ALL,
    }
    if DEBUG:
        for t in args:
            text += " "+str(t)
        print(color_mode[mode] + text + color_mode["END"])


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
        # mat = cv.applyColorMap(mat, cv.COLORMAP_HSV)
        mat = cv.applyColorMap(mat, cv.COLORMAP_HOT)

    if r < 100:
        mat = cv.resize(mat,None,fx=4,fy=4)
    cv.imshow(name,mat)

def normalize_zeromean(img):
    img = np.float32(img)
    img = (img - 127.5)/127.5
    return img 

def load_image(path, img_rows=64, img_cols=64):
    image_list = np.zeros((len(path),  img_rows, img_cols, 1))
    for i, fig in enumerate(path):
        #try:
        img = image.load_img(fig, target_size=(img_rows, img_cols))
        x = image.img_to_array(img).astype('float32')
        # print(x.shape)
        x = cv.cvtColor(x,cv.COLOR_BGR2GRAY)
        x = np.reshape(x,(64,64,1))
        # print(x.shape)
        # x = normalize_zeromean(x)
        x /= 255.
        image_list[i] = x
    return image_list

def load_image_flat(path, img_rows=64, img_cols=64):
    image_list = np.zeros((len(path),  img_rows*img_cols))
    for i, fig in enumerate(path):
        #try:
        img = image.load_img(fig, target_size=(img_rows, img_cols))
        x = image.img_to_array(img).astype('float32')
        # print(x.shape)
        x = cv.cvtColor(x,cv.COLOR_BGR2GRAY)
        x = np.reshape(x,(img_rows*img_cols))
        # x = x.ravel()
        # print(x.shape)
        x = normalize_zeromean(x)
        image_list[i] = x
    return image_list
