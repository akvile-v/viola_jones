import cv2
import glob
import os
import numpy as np

CAL = 'caltech'
SAVE = "C:/Users/akvil/Desktop/viola-jones-bakalauro-praktika/extracted"
CAL_FILES = glob.glob(
    os.path.join(CAL, '**', '*.jpg'), recursive=True)

#Function to convert photos to gray scale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

iterate = 0

#Saving only faces that were found by cv2 library
for imagePath in CAL_FILES:
    print(imagePath)
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("[INFO] Found {0} Faces.".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        image_gray = rgb2gray(roi_color)
        cv2.imwrite(os.path.join(SAVE, str(iterate)+"_face.jpg"), image_gray)
        iterate = iterate + 1
