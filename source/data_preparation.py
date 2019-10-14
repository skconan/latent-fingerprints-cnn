"""
    File name: data_preparation.py
    Author: skconan
    Date created: 2019/10/12
    Python Version: 3.7
"""

from utilities import *
from constansts import *
from fft import *

def gaussian_blur(img,patch):
    """
        img is gray scale
        patch is size of each blocks
    """
    patch = 16
    rows,cols = img.shape
    img_gaun = img.copy()
    for row in range(0,rows, patch):
        for col in range(0,cols, patch):
            roi = img[row:row+patch, col:col+patch]
            img_gaun[row:row+patch, col:col+patch] = cv.GaussianBlur(roi,(3,3),sigmaX=8,sigmaY=8)
    return img_gaun

def segmentation():
    file_lsit = get_file_path(PATH_NIST14_BINARY)

    for fpath in file_lsit:
        fname = get_file_name(fpath)
        img = cv.imread(PATH_NIST14 + "/" + fname + ".png",0)
        binary = cv.imread(fpath,0)
        count = 0 
        rows,cols = binary.shape
        step = 64
        for r in range(0,rows-step, step):
            for c in range(0,cols-step, step):
                roi = binary[r:r+step,c:c+step]
                # spectrum = spatial2freq(roi)
                # magnitude = get_magnitude(spectrum)
                # magnitude = normalize(to_logscale(magnitude))
                # imshow("magnitude_mapping1",magnitude.copy(), mapping=True)
                varianece = np.var(roi)
                print(varianece)
                
                if varianece >= 10000:
                    # cv.imshow("roi1",roi.copy())
                    cv.imwrite(PATH_TARGET_SEG + "\\" + fname + "_" + "%03d" % count + ".jpg",roi)
                    roi = img[r:r+step,c:c+step]
                    cv.imwrite(PATH_INPUT_SEG + "\\" + fname + "_" + "%03d" % count + ".jpg",roi)
                    count += 1
                    # spectrum = spatial2freq(roi)
                    # magnitude = get_magnitude(spectrum)
                    # magnitude = normalize(to_logscale(magnitude))
                    # magnitude_hist = cv.equalizeHist(magnitude)
                    # result = freq2spatial(magnitude_hist*spectrum)
                    # imshow("magnitude_mapping2",magnitude, mapping=True)
                    # imshow("magnitude_hist_mapping2",magnitude_hist, mapping=True)
                    # print(np.var(roi))
                    # cv.imshow("roi2",roi.copy())
                    # cv.imshow("result",result.copy())
                    # cv.waitKey(-1)

def main():
    segmentation()
        

if __name__ == "__main__":
    main()