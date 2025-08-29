
import cv2
import numpy as np

# Step 1: Capture video from webcam
cap = cv2.VideoCapture(0)   # 0 = default camera

while True:
    # Step 2: Read a frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Step 3: Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Step 4: Apply different edge detection methods
    
    # 4.1 Sobel (detect horizontal and vertical edges)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # X direction
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # Y direction
    sobel = cv2.magnitude(sobelx, sobely)  # Combine both
    
    # 4.2 Canny Edge Detection
    canny = cv2.Canny(gray, 100, 200)
    
    # 4.3 Laplacian of Gaussian (LoG)
    blur = cv2.GaussianBlur(gray, (3,3), 0)  # Smooth first
    log = cv2.Laplacian(blur, cv2.CV_64F)
    
    # Step 5: Contour Detection (based on Canny result)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_frame = frame.copy()
    cv2.drawContours(contour_frame, contours, -1, (0,255,0), 2)  # Draw green contours
    
    # Step 6: Show results in separate windows
    cv2.imshow("Original", frame)
    cv2.imshow("Sobel", np.uint8(sobel))
    cv2.imshow("Canny", canny)
    cv2.imshow("LoG", np.uint8(log))
    cv2.imshow("Contours", contour_frame)
    
    # Step 7: Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 8: Release resources
cap.release()
cv2.destroyAllWindows()
=======
import cv2
import numpy as np

# Step 1: Capture video from webcam
cap = cv2.VideoCapture(0)   # 0 = default camera

while True:
    # Step 2: Read a frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Step 3: Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Step 4: Apply different edge detection methods
    
    # 4.1 Sobel (detect horizontal and vertical edges)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # X direction
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # Y direction
    sobel = cv2.magnitude(sobelx, sobely)  # Combine both
    
    # 4.2 Canny Edge Detection
    canny = cv2.Canny(gray, 100, 200) # (upper and lower limits)
    
    # 4.3 Laplacian of Gaussian (LoG)
    blur = cv2.GaussianBlur(gray, (3,3), 0) # smooth 
    log = cv2.Laplacian(blur, cv2.CV_64F)
    
    # Step 5: Contour Detection (based on Canny result)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_frame = frame.copy()
    cv2.drawContours(contour_frame, contours, -1, (0,255,0), 2)  # Draw green contours
    
    # Step 6: Show results in separate windows
    cv2.imshow("Original", frame)
    cv2.imshow("Sobel", np.uint8(sobel))
    cv2.imshow("Canny", canny)
    cv2.imshow("LoG", np.uint8(log))
    cv2.imshow("Contours", contour_frame)
    
    # Step 7: Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 8: Release resources
cap.release()
cv2.destroyAllWindows()
>>>>>>> 741c0f3 (template detection code)
