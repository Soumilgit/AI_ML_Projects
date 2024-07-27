import cv2
import numpy as np

def image_to_sketch(image_path):
  
    img = cv2.imread(r'C:\Users\Soumil\Desktop\Name\imagemountain.jpg')
  
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    inverted_gray_image = cv2.bitwise_not(gray_image)

    blurred_image = cv2.GaussianBlur(inverted_gray_image, (21, 21), sigmaX=0, sigmaY=0)

    def dodge(front, back):
        result = cv2.divide(front, 255-back, scale=256)
        return result
    
    final_image = dodge(blurred_image, gray_image)

    cv2.imwrite("sketch_output.jpg", final_image)

image_to_sketch('imagemountain.jpg')
