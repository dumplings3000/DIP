# !/usr/bin/env python3
import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

class EdgeDetection:
    def __init__(self, file_path, show=True, save=True):
        self.file_path = file_path
        self.frames = []
        self.image_type = None
        self.show = show
        self.save = save
        self.main()

    def detect_file_type(self):
        _, ext = os.path.splitext(self.file_path)
        self.image_type = ext.lower()
        if self.image_type not in ['.gif', '.tif', '.png', '.jpg', '.jpeg']:
            raise ValueError("Unsupported file type! Supported types: .gif, .tif, .png, .jpg, .jpeg")

    def load_image(self):
        if self.image_type == '.gif':
            gif = Image.open(self.file_path)
            while True:
                frame = np.array(gif.convert("RGB"))
                self.frames.append(frame)
                try:
                    gif.seek(gif.tell() + 1)
                except EOFError:
                    break
        else:
            image = cv2.imread(self.file_path)
            if image is None:
                raise ValueError("Failed to load the image!")
            self.frames = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)] 

    def compute_gradient(self, image):
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)        
        return cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def detect_edges(self, threshold=50):
        edge_results = []
        for frame in self.frames:
            gradient_magnitude = self.compute_gradient(frame)
            _, edge_image = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
            edge_results.append(edge_image)
        return edge_results

    def visualize_results(self, edge_results):
        for i, (original, edge) in enumerate(zip(self.frames, edge_results)):
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title(f'Original Image')
            plt.imshow(original)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title(f'Edge Detected Image')
            plt.imshow(edge, cmap='gray')
            plt.axis('off')

            plt.show()

    def save_results(self, edge_results, output_prefix="edge_result"):
        output_prefix = f"{output_prefix}_for_{os.path.basename(self.file_path)}"
        for i, edge_image in enumerate(edge_results):
            output_path = f"{output_prefix}_frame_{i + 1}.png"
            cv2.imwrite(output_path, edge_image)
            print(f"Saved: {output_path}")

    def main(self, threshold=50, output_prefix="edge_result"):
        self.detect_file_type()
        self.load_image()
        edge_results = self.detect_edges(threshold=threshold)
        if self.show:
            self.visualize_results(edge_results)
        if self.save:
            self.save_results(edge_results, output_prefix=output_prefix)

if __name__ == '__main__':
    file_path1 = "Visual resolution.gif"
    detector = EdgeDetection(file_path1)

    file_path2 = "lenna-RGB.tif"
    detector2 = EdgeDetection(file_path2)