import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def FFT(img):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    return f_shift

def inv_FFT(f_shift):
    f_ishift = np.fft.ifftshift(f_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def sinsoide_mask(img, d, w):
    mask = np.ones((img.shape[0], img.shape[1]), np.uint8)
    x_center, y_center= img.shape[0] // 2, img.shape[1] // 2
    mask[x_center - d - w: x_center - d + w, y_center - d - w: y_center - d + w] = 0
    mask[x_center + d - w: x_center + d + w, y_center + d - w: y_center + d + w] = 0
    return mask

def moire_mask(img, d, w):
    # """Create a mask to filter out moire pattern frequencies."""
    mask = np.ones((img.shape[0], img.shape[1]), np.uint8)
    x_center, y_center = img.shape[0] // 2, img.shape[1] // 2
    # Set the specified frequency regions to 0 to remove moire patterns
    mask[x_center - d - w: x_center - d + w, y_center - d - w: y_center - d + w] = 0  
    mask[x_center + d - w: x_center + d + w, y_center - d - w: y_center + d + w] = 0 
    mask[x_center - d - w: x_center - d + w, y_center + d - w: y_center + d + w] = 0 
    mask[x_center + d - w: x_center + d + w, y_center + d - w: y_center + d + w] = 0 

def filter(img , mask):
    img_filtered = img * mask
    return img_filtered

if __name__ == '__main__':
    img1 = cv2.imread('astronaut-interference.tif', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('car-moire-pattern.tif', cv2.IMREAD_GRAYSCALE)

    # FFT
    f1_shift = FFT(img1)
    f2_shift = FFT(img2)

    # mask
    mask_1 = sinsoide_mask(img1, 30, 10)
    mask_2 = moire_mask(img2, 30, 15)

    # filter
    f1_shift_filtered = filter(f1_shift, mask_1)
    f2_shift_filtered = filter(f2_shift, mask_2)

    # Inverse FFT
    img_1_processed = inv_FFT(f1_shift_filtered)
    img_2_processed = inv_FFT(f2_shift_filtered)

    # create mkdir if not exist
    path = 'results/'
    if not os.path.exists(path):
        os.makedirs(path)
    # Save the processed images in processed_images folder
    cv2.imwrite(path +'astronaut.png', img_1_processed)
    cv2.imwrite(path +'car.png', img_2_processed)
