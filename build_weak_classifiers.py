from process import *
from weak_classifier import *
from time import time

FEATURE_FILE = "features_17.pickle"

start = time()
np.random.seed(20230124)

#Example photo saving
#show_image=read_resize('train/faces_aligned_small_mirrored_co_aligned_cropped_cleaned/F/2013_Virginia_Arlington_Washington-Lee_25-5.png', 186,171)
#show_image.save("original.png")
#to_image(normalize(to_float_array(show_image))).save("normalized.png")
#to_image(integral_image(normalize(to_float_array(show_image)))).save("integral_image.png")

#Learning set is made of 5000 positive examples and 2500 negative examples
xs, ys = sample_data(5000, 2500, True)
xis = np.array([integral_image(normalize(to_float_array(x))) for x in xs])

# Read features from calculated features file
with (open(FEATURE_FILE, "rb")) as openfile:
        while True:
            try:
                features = pickle.load(openfile)
            except EOFError:
                break

# Build different classifiers

# Building 2 weak classifiers
weak_classifiers = build_weak_classifiers('1', 2, xis, ys, features)

# Building 10 weak classifiers
weak_classifiers_10 = build_weak_classifiers('2', 10, xis, ys, features)

# Building 25 weak classifiers
weak_classifiers_25 = build_weak_classifiers('3', 25, xis, ys, features)

# Building 40 weak classifiers
weak_classifiers_40 = build_weak_classifiers('4', 40, xis, ys, features)

# Building 50 weak classifiers
weak_classifiers_50 = build_weak_classifiers('5', 50, xis, ys, features)

# Building 60 weak classifiers
weak_classifiers_60 = build_weak_classifiers('6', 60, xis, ys, features)

print(f"finished after {round(time() - start, 2)} seconds")

