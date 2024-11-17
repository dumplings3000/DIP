import cv2
import numpy as np
import os

def sobel_kernel():
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return kernel_x, kernel_y

def laplacian_kernel():
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return kernel

def kernel_filter(img, kernel, img_pad):
    img_filter = cv2.filter2D(img_pad, -1, kernel)
    img_filter = img_filter[kernel.shape[0] // 2 : kernel.shape[0] // 2 + img.shape[0] , 
                    kernel.shape[1] // 2 : kernel.shape[1] // 2 + img.shape[1]]
    return img_filter

def filter(img, kernel):
    kernel_size = kernel.shape[0]
    img_pad = cv2.copyMakeBorder(img, kernel_size // 2, kernel_size // 2, kernel_size // 2,
                kernel_size // 2, cv2.BORDER_REPLICATE)
    filtered_img = kernel_filter(img, kernel, img_pad)
    return filtered_img

def sobel (img):
    kernel_x, kernel_y = sobel_kernel()
    sobel_x = filter(img, kernel_x)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = filter(img, kernel_y)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    
    sobel_xy =  cv2.add(sobel_x, sobel_y)
    sobel_xy = cv2.normalize(sobel_xy, None, 0, 255, cv2.NORM_MINMAX)
    return sobel_xy

def laplacian (img):
    kernel = laplacian_kernel()
    laplacian_img = filter(img, kernel)
    laplacian_img = cv2.normalize(laplacian_img, None, 0, 255, cv2.NORM_MINMAX)
    return laplacian_img
          
if __name__ == '__main__':
    img_1 = cv2.imread('Bodybone.bmp')
    img_2 = cv2.imread('fish.jpg')
    # sobel
    sobel_xy_1 = sobel(img_1)
    sobel_xy_2 = sobel(img_2)
    # laplacian
    laplacian_img_1 = laplacian(img_1)
    laplacian_img_2 = laplacian(img_2)
    # combine two filters
    # bodybone
    result_1 = cv2.addWeighted(sobel_xy_1, 1, laplacian_img_1, 1, 0)
    result_1 = cv2.normalize(result_1, None, 0, 255, cv2.NORM_MINMAX)
    # fish
    result_2 = cv2.addWeighted(sobel_xy_2, 1, laplacian_img_2, 1, 0)
    result_2 = cv2.normalize(result_2, None, 0, 255, cv2.NORM_MINMAX)
    # save bodybone images
    output_dir = 'img1_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, 'original_image_1.png'), img_1)
    cv2.imwrite(os.path.join(output_dir, 'sobel_image_1.png'), sobel_xy_1)
    cv2.imwrite(os.path.join(output_dir, 'laplacian_image_1.png'), laplacian_img_1)
    cv2.imwrite(os.path.join(output_dir, 'final_image_1.png'), result_1)
    # save fish images
    output_dir = 'img2_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, 'original_image_2.png'), img_2)
    cv2.imwrite(os.path.join(output_dir, 'sobel_image_2.png'), sobel_xy_2)
    cv2.imwrite(os.path.join(output_dir, 'laplacian_image_2.png'), laplacian_img_2)
    cv2.imwrite(os.path.join(output_dir, 'final_image_2.png'), result_2)