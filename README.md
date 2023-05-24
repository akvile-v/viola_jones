# Viola-Jones algoritm 

## Goal

The goal of the program is to use the [Viola-Jones algorithm](https://doi.org/10.1023/B:VISI.0000013087.49260.fb) to train a program to perform
face detection from a window of features of a selected size, computing weak and strong classifiers.

## Programs and their usage

- ### process.py:
    - image processing and reading functions. 
    - In this program SUB_WINDOW can be changed for different feature count calculations.
    - Also TRAIN_FACES_PATH and TRAIN_NONFACES_PATH should be changed in order to have learning database folder or testing database folder.

- ### weak_classifier.py:
    -  functions to build weak classifiers.
    - STATS_AFTER can be set to numbet of features after which current status will be printed out. 
    - PART_OF_FEATURES should be change to number smaller than 1, that indicates what part of features should be in the calculations.

- ### strong_classifier.py:
    - functions to build strong classifiers. No data need to be changed.

- ### simple_features.py:
    - functions to build simple features in wea classifiers. Mostly used by weak_classifiers.
    -  No data needs to be changed.
    - There is availability to change simple_features structure (if needed) in extract_features funtion.

- ### build_weak_classifier.py:
    - uses weak_classifier functions to build different size classifiers.
    - FEATURE_FILE should be changed to already
 calculated features (it can be done with extract_features function).
    - Also weak classifiers numbers can be changed in calculations as needed.

- ### extract_faces.py: 
    - was used for specific database to extract faces ant to have photos only with croped face frame.
    - Data should be changed only if program is needed.

- ### test_classificator.py: 
    - program is used for classifing test images and printing statisics as bar chart. 
    - Program outcome should be changed according to raised goals.
    - WEAK_CLASSIFIERS_FILE should be changed to wanted classifier file.

- ### group_test.py:
     - program examines group photo with different sizes strong classifiers. 
     - WEAK_CLASSIFIERS_FILE should be changed to wanted classifier file. 
    - IMG_WID and IMG_HEI should be modified according to wanted frame size. 
    - IMAGE_PATH should be path to wanted to test group photo.

- ### video_detect.py 
    - program examines video separate into photos and detects possible face variants.
    - WEAK_CLASSIFIERS_FILE should be changed to wanted classifier file.
    - WID and HEI should be modified according to wanted frame size.-
    - TIMES_ARRAY shoul be modified regarding times, how much user wants frame to be smaller ar bigger.

## Files

- ### features_17.pickle
    - file with all 17x17px sub-window feature information
- ### weak_classifiers
    - in earlier works calculated weak classifiers folder
- ### weak_classifiers_caltech
    - bachelor thesis calculated weak classifiers folder

## Details about programs

Programs calculates and stores the number of features, their coordinates, height, and width of all available features in a subwindow of the specified size. In the article, the authors of the Viola-Jones algorithm present a subwindow of size 24px x 24px as one of the most suitable for searching facial features, because in this way the number of features is maximally reduced without losing resolution.

 Due to limited resources (the sub-window mentioned in the article contains over 160,000 features), a sub-window of size 17px x 17px was selected in the program of this work, in which 40986 features are found.

Only 25\% of all features are included in the calculations.

## Data bases used for learning and testing

The input of the program is the training and test images. The program uses the freely available [CalTech 10k Web Faces](https://www.vision.caltech.edu/datasets/caltech_10k_webfaces/) photo database 
for non-profit research on face recognition tasks for face training. The database contains various resolution photos of faces found in 
online searches for research that requires unlimited photo backgrounds and a variety of individual facial features (coordinates of 
eyes, nose, mouth are provided).

 For the training and testing of the algorithm, the database was adapted with the help of the Python 
programming language [OpenCV](https://opencv.org/) library, by cutting out the faces separately and converting them to gray color (a three-dimensional array 
turns into a two-dimensional array).

Also, the freely available [Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02) photo database was used for faceless photo training for 
testing recognition of different categories (eg: airplanes, watches). 5000 face photos and 2500 photos of various categories where the 
face does not exist are selected for training. For testing data, 2000 face photos and 2000 photos of various categories were selected 
from the used database. Also, the classifiers were tested with a part of the [The Images of Groups Dataset](http://chenlab.ece.cornell.edu/people/Andy/ImagesOfGroups.html) database, which contains 
group photos of individuals for the study of social phenomena in groups.

From earlier works, there was used another databases. A portion of the freely available photo database [A Century of Portraits]( 	
https://doi.org/10.48550/arXiv.1511.02575), consisting of 37921 student faces from 1905 to 2013, was used to train the program, and the photo size is 186px x 171px  and [Stanford background dataset](http://dags.stanford.edu/projects/scenedataset.html), which consists of 715 different nature and urban photos of approximately 320px x 240px. 2000 unique randomly selected face pictures and 1000 (with repetition) pictures where the face does not exist are selected for training. For testing data, 1000 unique randomly selected face photos and 500 nature photos were selected from the used database, thus calculating the accuracy of strong classifiers with different number of features and determining the frequency of correctly identified positive and negative cases.

## Usage

All programs should be used depending on scope and goal of research.

Video detection can be tested with video_detect.py program.

Also it can be executed:
```
python video_detect.py
```

## Scope of usage

Program was used in bachelor thesis practical part.