from weak_classifier import *
from PIL import Image
from process import *
from numba import njit

HALF_WINDOW = SUB_WINDOW // 2

# Function that calculates strong classifier from  weak classifiers
def strong_classifier(x, weak_classifiers):
    sum_hypotheses = 0.
    sum_alphas = 0.
    for c in weak_classifiers:
        sum_hypotheses += c.alpha * weak_classifier(x, c.classifier, c.polarity, c.threshold)
        sum_alphas += c.alpha
    return 1 if (sum_hypotheses >= .5*sum_alphas) else 0

#Function to render classifier square around founded face
def render_candidates(image, candidates):
    canvas = to_float_array(image.copy())
    for row, col in candidates:
        canvas[row-HALF_WINDOW-1:row+HALF_WINDOW, col-HALF_WINDOW-1, :] = [1., 1., 1.]
        canvas[row-HALF_WINDOW-1:row+HALF_WINDOW, col+HALF_WINDOW-1, :] = [1., 1., 1.]
        canvas[row-HALF_WINDOW-1, col-HALF_WINDOW-1:col+HALF_WINDOW, :] = [1., 1., 1.]
        canvas[row+HALF_WINDOW-1, col-HALF_WINDOW-1:col+HALF_WINDOW, :] = [1., 1., 1.]
    return to_image(canvas)