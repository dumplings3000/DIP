#/usr/bin/env python3
import cv2
import pytesseract
import os

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarize the image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    return binary

def recognize_text(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Recognize text using Tesseract
    custom_config = r'--oem 3 --psm 1'  # 改变 OCR 的引擎模式和页面分割模式
    text = pytesseract.image_to_string(processed_image, lang='eng', config=custom_config)
    # text = pytesseract.image_to_string(processed_image)
    
    return text

def main(image_files):
    results = {}
    if not os.path.exists('result'):
        os.makedirs('result')
    
    for image_file in image_files:
        text = recognize_text(image_file)
        results[image_file] = text

        output_file_name = image_file.replace('.tif', '.txt')
        output_file_path = os.path.join('result', output_file_name)

        # Write recognized text to the corresponding text file
        with open(output_file_path, 'w') as f:
            f.write(f"--- {image_file} ---\n{text}\n\n")
            
    print("Text recognition completed. Results saved to 'recognized_text.txt'.")

# Example usage
if __name__ == "__main__":
    image_files = ['text-broken.tif', 'text-spotshade.tif', 'text.tif', 'denoised_image.tif']
    main(image_files)