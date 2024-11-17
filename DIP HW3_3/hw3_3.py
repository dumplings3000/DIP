import cv2
import numpy as np
import os

kernel_size_1 = 250
kernel_size_2 = 120
std_1 = 64
std_2 = 512

def guassian_kernel(size, std):
    kernel = np.zeros([size, size])
    center = size // 2
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-((i - center)**2 + (j - center)**2) / (2 * std**2))
    return kernel / np.sum(kernel)

def low_pass_filter(img, kernel, img_pad):
    noise = cv2.filter2D(img_pad, -1, kernel)
    noise = noise[kernel.shape[0] // 2 : kernel.shape[0] // 2 + img.shape[0] ,
                 kernel.shape[1] // 2 : kernel.shape[1] // 2 + img.shape[1]]
    noise[noise == 0] = 1
    img_filtered = (img / noise) *255/2
    return img_filtered, noise


if __name__ == '__main__':
    img_1 = cv2.imread('checkerboard1024-shaded.tif')
    img_2 = cv2.imread('N1.bmp')

    img_pad1 = cv2.copyMakeBorder(img_1, kernel_size_1 // 2, kernel_size_1 // 2, 
                kernel_size_1 // 2, kernel_size_1 // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img_pad2 = cv2.copyMakeBorder(img_2, kernel_size_2 // 2, kernel_size_2 // 2,
                kernel_size_2 // 2, kernel_size_2 // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    kernel_1 = guassian_kernel(kernel_size_1, std_1)
    kernel_2 = guassian_kernel(kernel_size_2, std_2)


    img_filtered_1, noise_1 = low_pass_filter(img_1, kernel_1, img_pad1)
    img_filtered_2, noise_2 = low_pass_filter(img_2, kernel_2, img_pad2)

    output_dir = 'img1_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, 'original_image_1.png'), img_1)
    cv2.imwrite(os.path.join(output_dir, 'filtered_image_1.png'), img_filtered_1)
    cv2.imwrite(os.path.join(output_dir, 'noise_image_1.png'), noise_1)
    output_dir = 'img2_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, 'original_image_2.png'), img_2)
    cv2.imwrite(os.path.join(output_dir, 'filtered_image_2.png'), img_filtered_2)
    cv2.imwrite(os.path.join(output_dir, 'noise_image_2.png'), noise_2)
