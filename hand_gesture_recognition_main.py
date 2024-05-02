import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

background = None

accumulated_weight = 0.5

roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600

def calc_accum_avg(frame,accumulated_weight):
    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None
    
    cv2.accumulateWeighted(frame,background,accumulated_weight)

def segment(frame,threshold_min=25):
    
    diff = cv2.absdiff(background.astype('uint8'),frame)
    
    ret, thresholded = cv2.threshold(diff,threshold_min,255,cv2.THRESH_BINARY)
    
    image,contours,heirarchy = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == None:
        return None
    
    else:
        #assuming the largest external contoru in ROI, is the hand
        hand_segment = max(contours,key=cv2.contourArea)
        
        return (thresholded,hand_segment)
    

def count_fingers(thresholded, hand_segment):
    # Compute the convex hull of the hand segment
    conv_hull = cv2.convexHull(hand_segment)

    # Calculate extreme points of the convex hull
    leftmost = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    rightmost = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])
    topmost = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottommost = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])

    # Calculate the center of the hand segment
    cX = (leftmost[0] + rightmost[0]) // 2
    cY = (topmost[1] + bottommost[1]) // 2

    # Calculate distances from the center to extreme points
    distances = euclidean_distances([[cX, cY]], [leftmost, rightmost, topmost, bottommost])[0]

    # Maximum distance among the extreme points
    max_distance = distances.max()

    # Radius of circular region of interest
    radius = int(0.9 * max_distance)

    # Circumference of the circular region of interest
    circumference = 2 * np.pi * radius

    # Create a circular region of interest mask
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)

    # Apply the circular region of interest mask
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # Find contours within the circular ROI
    contours, _ = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Initialize finger count
    count = 0

    # Iterate through contours and count fingers
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Check if the contour is above the wrist and occupies less than 25% of the circumference
        out_of_wrist = (cY + int(cY * 0.25)) > (y + h)
        limit_points = (circumference * 0.25) > cnt.shape[0]

        if out_of_wrist and limit_points:
            count += 1

    return count

cam = cv2.VideoCapture(0)

num_frames = 0

while True:
    
    ret, frame = cam.read()
    
    frame_copy = frame.copy()
    
    roi = frame[roi_top:roi_bottom,roi_right:roi_left]
    
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray,(7,7),0)
    
    if num_frames < 60:
        calc_accum_avg(gray,accumulated_weight)
        
        if num_frames <= 59:
            cv2.putText(frame_copy,'WAIT. GETTING BACKGROUND',(200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('Finger Count',frame_copy)
    else:
        
        hand = segment(gray)
        
        if hand is not None:
            
            thresholded , hand_segment = hand
            
            # DRAWS CONTOURS AROUND REAL HAND IN LIVE STREAM
            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(255,0,0),5)
            
            fingers = count_fingers(thresholded,hand_segment)
            
            cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
            cv2.imshow('Thresholded',thresholded)
            
    cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(0,0,255),5)
    
    num_frames += 1
    
    cv2.imshow('Finger Count',frame_copy)
    
    k = cv2.waitKey(1) & 0xFF
    
    if k == 27:
        break
        
cam.release()
cv2.destroyAllWindows()
    