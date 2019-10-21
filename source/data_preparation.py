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
    file_list = get_file_path(PATH_NIST14_BINARY)

    for fpath in file_list:
        fname = get_file_name(fpath)
        img = cv.imread(PATH_NIST14 + "/" + fname + ".png",0)
        # imshow("img",img.copy())
        # img = freq2spatial(spatial2freq(img,True))
        # img = spatial2freq(img)
        spectrum = spatial2freq(img)
        magnitude = get_magnitude(spectrum)
        magnitude = normalize(to_logscale(magnitude))
        imshow("magnitude_img",magnitude.copy(), mapping=True)
        # img = 255 - img
        # imshow("img_rm_dc",img.copy())
        binary = cv.imread(fpath,0)
        # binary = freq2spatial(spatial2freq(binary,True))
        count = 0 
        rows,cols = binary.shape
        step = 64
        for r in range(0,rows-step, step):
            for c in range(0,cols-step, step):
                roi = binary[r:r+step,c:c+step]
                spectrum = spatial2freq(roi)
                # img_rm_dc = freq2spatial(spectrum)
                magnitude = get_magnitude(spectrum)
                # mag = magnitude.copy()
                magnitude = normalize(to_logscale(magnitude))
                imshow("magnitude_mapping1",magnitude.copy(), mapping=True)
                imshow("img1",roi.copy())
                # imshow("img_rm_dc1",img_rm_dc.copy())
                varianece = np.var(roi)
                print(varianece)
                
                if varianece >= 10000:
                    # cv.imshow("roi1",roi.copy())
                    # cv.imshow("magnitude",magnitude)
                
                    # magnitude = normalize(to_logscale(magnitude))

                    # cv.imshow("magnitude_rm_dc",magnitude)
                    # cv.imwrite(PATH_TARGET_SEG + "_stft\\" + fname + "_" + "%03d" % count + ".jpg",roi)
                    roi = img[r:r+step,c:c+step]
                    # cv.imwrite(PATH_INPUT_SEG + "_stft\\" + fname + "_" + "%03d" % count + ".jpg",roi)
                    count += 1
                    spectrum = spatial2freq(roi)
                    img_rm_dc = freq2spatial(spectrum)

                    magnitude = get_magnitude(spectrum)
                    magnitude = normalize(to_logscale(magnitude))
                    # magnitude_hist = cv.equalizeHist(magnitude)
                    # result = freq2spatial(magnitude_hist*spectrum)
                    imshow("magnitude_mapping2",magnitude, mapping=True)
                    imshow("img2",roi)
                    # imshow("img_rm_dc2",img_rm_dc)
                    # imshow("magnitude_hist_mapping2",magnitude_hist, mapping=True)
                    # print(np.var(roi))
                    # cv.imshow("roi2",roi.copy())
                    # cv.imshow("result",result.copy())
                    cv.waitKey(-1)

def main():
    segmentation()
        

if __name__ == "__main__":
    main()