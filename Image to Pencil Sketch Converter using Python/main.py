import cv2
import numpy as np
import os
import time

def detect_shapes_from_video():
    cap = cv2.VideoCapture(0)
    frame_counter = 0
    last_image_time = time.time()
    
    images_folder = r'C:\Users\Soumil\Desktop\Name\image_folder'  # Adjust this path
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process video frame as usual for live feed
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                if len(approx) == 3:
                    shape = "Triangle"
                elif len(approx) == 4:
                    (x, y, w, h) = cv2.boundingRect(approx)
                    ar = w / float(h)
                    shape = "Rectangle" if 0.9 <= ar <= 1.1 else "Quadrilateral"
                else:
                    shape = "Circle"
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int((M["m10"] / M["m00"]) + 0.5)
                    cY = int((M["m01"] / M["m00"]) + 0.5)
                    
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                    cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Periodically check for new images in the folder
            if frame_counter % 30 == 0 and time.time() - last_image_time > 1:
                new_images = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]
                
                for image_name in new_images:
                    image_path = os.path.join(images_folder, image_name)
                    img = cv2.imread(image_path)
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
                    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        peri = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                        
                        if len(approx) == 3:
                            shape = "Triangle"
                        elif len(approx) == 4:
                            (x, y, w, h) = cv2.boundingRect(approx)
                            ar = w / float(h)
                            shape = "Rectangle" if 0.9 <= ar <= 1.1 else "Quadrilateral"
                        else:
                            shape = "Circle"
                        
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int((M["m10"] / M["m00"]) + 0.5)
                            cY = int((M["m01"] / M["m00"]) + 0.5)
                            
                            # Draw the shape name on the current frame of the video feed
                            cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Optionally, remove the image after processing
                    os.remove(image_path)
                    last_image_time = time.time()
            
            # Display the resulting frame with potentially overlaid shape names
            cv2.imshow('Video Frame', frame)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
    finally:
        cap.release()
        cv2.destroyAllWindows()

detect_shapes_from_video()
