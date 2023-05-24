from process import *
from weak_classifier import *
from strong_classifier import *

#Set image dimensions depending on sub-window size
# and faces size in image
IMG_WID = 1024/5
IMG_HEI = 768/5

#Group photo path
IMAGE_PATH = "group_images_test/12_test.jpg"

#Group photo path
WEAK_CLASSIFIER_FILE = "weak_classifiers/*"

#Load weak classifiers from file
weak_classifiers_array = []
for pc in glob.glob(WEAK_CLASSIFIER_FILE):
    with (open(pc, "rb")) as openfile:
        while True:
            try:
                weak_classifiers_array.append(pickle.load(openfile))
            except EOFError:
                break

positive_features_2 = []
positive_features_10 = []
positive_features_25 = []
positive_features_40 = []
positive_features_50 = []
positive_features_60 = []

#Resize given group photo
image_array = to_float_array(read_resize(IMAGE_PATH, IMG_WID, IMG_HEI))
#Calculating integral image of given group photo
integral = integral_image(normalize(image_array))

rows, cols = integral.shape[0:2]

#Calculate strong classifier for every different set of weak classifiers
for row in range(HALF_WINDOW + 1, rows - HALF_WINDOW):
    for col in range(HALF_WINDOW + 1, cols - HALF_WINDOW):
        selected_sub_window = integral[row-HALF_WINDOW-1:row+HALF_WINDOW+1,
             col-HALF_WINDOW-1:col+HALF_WINDOW+1]
                

        is_face = strong_classifier(selected_sub_window, weak_classifiers_array[0:2])
        if not is_face:
            continue
        positive_features_2.append((row, col))

        is_face = strong_classifier(selected_sub_window, weak_classifiers_array[2:12])
        if not is_face:
            continue
        positive_features_10.append((row, col))

        is_face = strong_classifier(selected_sub_window, weak_classifiers_array[12:37])
        if not is_face:
            continue
        positive_features_25.append((row, col))

        is_face = strong_classifier(selected_sub_window, weak_classifiers_array[37:77])
        if not is_face:
            continue
        positive_features_40.append((row, col))

        is_face = strong_classifier(selected_sub_window, weak_classifiers_array[77:127])
        if not is_face:
            continue
        positive_features_50.append((row, col))

        is_face = strong_classifier(selected_sub_window, weak_classifiers_array[127:187])
        if not is_face:
            continue
        positive_features_60.append((row, col))
    
print(f'Found {len(positive_features_2)} face candidates with 2 classifiers')
print(f'Found {len(positive_features_10)} face candidates with 10 classifiers')
print(f'Found {len(positive_features_25)} face candidates with 25 classifiers')
print(f'Found {len(positive_features_40)} face candidates with 40 classifiers')
print(f'Found {len(positive_features_50)} face candidates with 50 classifiers')
print(f'Found {len(positive_features_60)} face candidates with 60 classifiers')

# Open image
image_test = Image.open(IMAGE_PATH)
image_test.thumbnail((IMG_WID, IMG_HEI), Image.Resampling.LANCZOS)
render_candidates(image_test, positive_features_60).save("test_12_old.png")