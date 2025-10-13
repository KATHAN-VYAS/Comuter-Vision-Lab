import cv2
import numpy as np

# --- Step 1: Read video ---
cap = cv2.VideoCapture(r'D:\kathan\python_codes\7th sem\CV lab\Comuter-Vision-Lab\moving light.mp4')


# --- Step 2: Parameters for Shi-Tomasi Corner Detection ---
feature_params = dict(maxCorners=200,     # Maximum number of corners to detect
                      qualityLevel=0.3,   # Minimum accepted quality (0 to 1)
                      minDistance=7,      # Minimum distance between corners
                      blockSize=7)        # Size of neighborhood

# --- Step 3: Parameters for Lucas–Kanade Optical Flow ---
lk_params = dict(winSize=(15, 15),        # Search window size
                 maxLevel=2,              # Number of pyramid layers
                 criteria=(cv2.TERM_CRITERIA_EPS | 
                           cv2.TERM_CRITERIA_COUNT, 10, 0.03))  # Stopping criteria

# --- Step 4: Create random colors for visualizing tracks ---
color = np.random.randint(0, 255, (200, 3))

# --- Step 5: Take the first frame and detect corners to track ---
ret, old_frame = cap.read()
if not ret:
    print("Error: Cannot read video file.")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# --- Step 6: Create a mask image for drawing motion trails ---
mask = np.zeros_like(old_frame)

# --- Step 7: Processing loop for each frame ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Step 8: Calculate Optical Flow using Lucas–Kanade ---
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # --- Step 9: Select good points where flow is found ---
    if p1 is None:
        break
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # --- Step 10: Draw motion tracks ---
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    # --- Step 11: Combine mask and current frame ---
    img = cv2.add(frame, mask)

    # --- Step 12: Display the result ---
    cv2.imshow('Lucas-Kanade Optical Flow', img)

    # Exit if ESC is pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

    # --- Step 13: Update for next iteration ---
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# --- Step 14: Cleanup ---
cv2.destroyAllWindows()
cap.release()
