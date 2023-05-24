from process import *
from weak_classifier import *
from strong_classifier import *
from time import time, sleep
import cv2
import cloudpickle as pickle
from numba import jit
import numpy as np
import statistics

WEAK_CLASSIFIERS_FILE = "weak_classifiers_caltech/6*"

#Set size of video frame
WID = 500
HEI = 400

#Times to test smaller frames
TIMES_ARRAY = [4, 6, 8]

#Begin video capture with selected frame size
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WID)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEI)

#All strong clasiffier calculations in one function
# without any dependencies on other programs
@jit
def face_position(integral, win):

    rows, cols = integral.shape[0:2]
    face_positions = []
    for row in range(HALF_WINDOW + 1, rows - HALF_WINDOW):
        for col in range(HALF_WINDOW + 1, cols - HALF_WINDOW):
            window = integral[row-HALF_WINDOW-1:row+HALF_WINDOW+1,
                col-HALF_WINDOW-1:col+HALF_WINDOW+1]
            
            sum_hypotheses = 0.
            sum_alphas = 0.
            for c in win:
                white = 0
                black = 0
                x1, y1, w1, h1 = c.classifier[1], c.classifier[2], c.classifier[3], c.classifier[4]
                # For calculations explanation check 3 pav. in Course project
                # Index 0 -> (2, 1) -> two horizontal rectangles
                if c.classifier[0] == 0:
                    x, y, w, h = int(x1), int(y1), int(w1/2), int(h1)
                    white = window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                    x, y, w, h = int(x1) + int(w1/2), int(y1), int(w1/2), int(h1)
                    black = window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                # Index 1 -> (1, 2) -> two vertical rectangles

                elif c.classifier[0] == 1:
                    x, y, w, h = int(x1), int(y1), int(w1), int(h1/2)
                    white = window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                    x, y, w, h = int(x1), int(y1 + h1/2), int(w1), int(h1/2)
                    black = window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                # Index 2 -> (3, 1) -> three horizontal rectangles

                elif c.classifier[0] == 2:
                    x, y, w, h = int(x1), int(y1), int(w1/3), int(h1)
                    white = window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                    x, y, w, h = int(x1 + w1*2/3), int(y1), int(w1/3), int(h1)
                    white += window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                    x, y, w, h = int(x1 + w1/3), int(y1), int(w1/3), int(h1)
                    black = window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                # Index 3 -> (1, 3) -> three vertical rectangles
                elif c.classifier[0] == 3:
                    x, y, w, h = int(x1), int(y1), int(w1), int(h1/3)
                    white = window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                    x, y, w, h = int(x1), int(y1 + h1*2/3), int(w1), int(h1/3)
                    white += window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                    x, y, w, h = int(x1), int(y1 + h1/3), int(w1), int(h1/3)
                    black = window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                # Index 4 -> (2, 2) -> four diagonal rectangles
                elif c.classifier[0] == 4:
                    x, y, w, h = int(x1), int(y1), int(w1/2), int(h1/2)
                    white = window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                    x, y, w, h = int(x1 + w1/2), int(y1 + h1/2), int(w1/2), int(h1/2)
                    white += window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                    x, y, w, h = int(x1 + w1/2), int(y1), int(w1/2), int(h1/2)
                    black = window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                    x, y, w, h = int(x1), int(y1 + h1/2), int(w1/2), int(h1/2)
                    black += window[y+h, x+w] - window[y+h, x] - window[y, x+w] + window[y, x]
                integral_feature_value = white - black
                sum_hypotheses += c.alpha * ((np.sign((c.polarity * c.threshold) - (c.polarity * integral_feature_value)) + 1) // 2) 
                sum_alphas += c.alpha
            if (sum_hypotheses >= .5*sum_alphas):
                probably_face = 1
            else:
                probably_face = 0            
            
            if probably_face < .5:
                continue
            face_positions.append((row, col))
    return face_positions

#Read already calculated weak clasiffiers
weak_classifiers = []
for pc in glob.glob(WEAK_CLASSIFIERS_FILE):
    with open(pc, "rb") as openfile:
        while True:
            try:
                weak_classifiers.append(pickle.load(openfile))
            except EOFError:
                break

mean_duration = []
index = 0
while(True):
    start = time()
    #Saving info from video frame
    ret,frame = cap.read()
    #Convering image into gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Loop of different image sizes
    for TIMES in TIMES_ARRAY:
        #Resize image
        gray_resized = cv2.resize(gray, (int(WID/TIMES),int(HEI/TIMES)))
        #Calculate integral image
        integral = integral_image(normalize(gray_resized))
        #Found face positions array according to strong clasiffier
        calculated_face_positions = face_position(integral, weak_classifiers)
        #Adding all found face positions as rectangles into frame
        for row, col in calculated_face_positions:
            color = (0, 0, 0)
            stroke = 1
            end_cord_x = (row*TIMES) - (SUB_WINDOW*TIMES)
            end_cord_y = (col*TIMES) + (SUB_WINDOW*TIMES)
            cv2.rectangle(gray, (col*TIMES, row*TIMES), (end_cord_y, end_cord_x), color, stroke)
    #Present current view
    cv2.imshow('Video detect', gray)
    #Save currect frame as image
    #cv2.imwrite(str(index) + '_image.jpg' , gray)
    index = index + 1
    duration = round(time() - start, 2)
    mean_duration.append(duration)
    print(f"finished after {duration} seconds")
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#print mean duration of one image calculation
print(statistics.mean(mean_duration))
cap.release()
cv2.destroyAllWindows() 