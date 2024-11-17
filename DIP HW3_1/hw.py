import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_result(img, name, size):
    image = img.reshape(size)
    cv2.imwrite(name + '_img.png', image)
    img_flat = img.flatten()
    plt.hist(img_flat, bins=256, range=(0, 255), density=True)
    plt.title('Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.savefig(name + '_histogram.png')
    plt.clf()
    
if __name__ == '__main__':
    # 3.1
    img = cv2.imread('aerial_view.tif', cv2.IMREAD_GRAYSCALE)
    show_result(img, 'oringinal', img.shape)

    #3.2
    img_eq = cv2.equalizeHist(img)
    show_result(img_eq, 'equalized', img.shape)

    #3.3
    img_eq_flatted = img_eq.flatten()
    c = 1 / sum(pow(i, 0.4)for i in range(256))
    hist_mat = c * np.power(np.arange(256), 0.4)
    hist_mat = np.round(255 * np.cumsum(hist_mat))

    img_match = np.searchsorted(hist_mat, img_eq_flatted)
    show_result(img_match, 'matched', img.shape)
