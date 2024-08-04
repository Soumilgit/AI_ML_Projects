import cv2
import numpy as np
import time

def detect_shapes_from_video():
    cap = cv2.VideoCapture(0)
    frame_counter = 0
    last_image_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Ignore small contours
                if cv2.contourArea(contour) < 100:
                    continue
                
                # Approximate the contour
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Identify the shape
                if len(approx) == 3:
                    shape = "Triangle"
                elif len(approx) == 4:
                    shape = "Quadrilateral"
                elif len(approx) == 5:
                    shape = "Pentagon"
                elif len(approx) == 6:
                    shape = "Hexagon"
                else:
                    shape = "Circle"
                
                # Draw the shape name on the current frame of the video feed
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)
            
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
