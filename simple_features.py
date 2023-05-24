import numpy as np

def integral_image(image):
    # Save image size
    height, width = image.shape

    # Make integral image array where 
    # first row and column would be 0 (for easier calculations)
    ii = np.zeros((height+1, width+1))

    for y in range(1, height+1):
        for x in range(1, width+1):
            # Integral image is calculated from all pixels
            # to the left and above. For faser calculation 
            # every iteration takes already calculated values
            ii[y, x] = ii[y, x-1] + ii[y-1, x] - ii[y-1, x-1] + image[y-1, x-1]

    return ii

# Function to extract all possible features in image
def extract_features(image):
    # For this implementation 5 different features are selected
    # Index 0 -> (2, 1) -> two horizontal rectangles
    # Index 1 -> (1, 2) -> two vertical rectangles
    # Index 2 -> (3, 1) -> three horizontal rectangles
    # Index 3 -> (1, 3) -> three vertical rectangles
    # Index 4 -> (2, 2) -> four diagonal rectangles
    simple_features = [(2, 1), (1, 2), (3, 1), (1, 3), (2, 2)]
    
    # Converting image into array 
    # (special function is not used as values does not matter)
    image_array = np.array(image)
    # Saving shape of array
    height, width = image_array.shape

    # Feature count variable
    feature_count = 0

    # Calculating all possible combinations count for features
    for feature in simple_features:
        x, y = feature
        for x_i in range(0, width-x+1):
            for y_i in range(0, height-y+1):
                    feature_count += int((width-x_i)/x)*int((height-y_i)/y)

    # Creating array to store feature index, coordinates and shape
    features_info = np.zeros((feature_count, 5))
    index = 0

    # Saving information about every possible combination
    for feature in range(len(simple_features)):
        x, y = simple_features[feature]
        # width can be from given simple feature shape (1, 2, 3)
        # to given image width
        for w in range(x, width+1, x):
            # height can be from given simple feature shape (1, 2, 3)
            # to given image height
            for h in range(y, height+1, y):
                # x coordinate can be from 0 to (image width - feature width)
                for x_i in range(0, width-w+1):
                    # y coordinate - from 0 to (image height - feature height)
                    for y_i in range(0, height-h+1):
                        # All possible coombination array is saved
                        features_info[index] = [feature, x_i, y_i, w, h]
                        index += 1
    return features_info

# Function to calculate integral sum in integral image given feature
def integral_sum(ii, x, y, w, h):
    x, y, w, h = int(x), int(y), int(w), int(h)
    # For calculations explanation check 3 pav. in Course project
    return  ii[y+h, x+w] - ii[y+h, x] - ii[y, x+w] + ii[y, x]

#Function to calculate feature value in integral image depending on feature
def integral_feature_value(integral_image, feature, x, y, w, h):
    white = 0
    black = 0
    # Index 0 -> (2, 1) -> two horizontal rectangles
    if feature == 0:
        white = integral_sum(integral_image, x, y, w/2, h)
        black = integral_sum(integral_image, x + w/2, y, w/2, h)
    # Index 1 -> (1, 2) -> two vertical rectangles

    elif feature == 1:
        white = integral_sum(integral_image, x, y, w, h/2)
        black = integral_sum(integral_image, x, y + h/2, w, h/2)
    # Index 2 -> (3, 1) -> three horizontal rectangles

    elif feature == 2:
        white = integral_sum(integral_image, x, y, w/3, h) 
        white += integral_sum(integral_image, x + w*2/3, y, w/3, h)
        black = integral_sum(integral_image, x + w/3, y, w/3, h)
    # Index 3 -> (1, 3) -> three vertical rectangles
    elif feature == 3:
        white = integral_sum(integral_image, x, y, w, h/3) 
        white += integral_sum(integral_image, x, y + h*2/3, w, h/3)
        black = integral_sum(integral_image, x, y + h/3, w, h/3)
    # Index 4 -> (2, 2) -> four diagonal rectangles
    elif feature == 4:
        white = integral_sum(integral_image, x, y, w/2, h/2) 
        white += integral_sum(integral_image, x + w/2, y + h/2, w/2, h/2)
        black = integral_sum(integral_image, x + w/2, y, w/2, h/2) 
        black += integral_sum(integral_image, x, y + h/2, w/2, h/2)
    return white - black