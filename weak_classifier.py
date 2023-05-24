import numpy as np
import pickle
from sklearn.metrics import *
from typing import *
from datetime import datetime
from simple_features import *

#########################GLOBAL##################################

STATS_AFTER     = 1000
PART_OF_FEATURES = 1./4.
ThresholdPolarity = NamedTuple('ThresholdPolarity', [('threshold', float), ('polarity', float)])

ClassifierResult = NamedTuple('ClassifierResult', [('threshold', float), ('polarity', int), 
                                                   ('classification_error', float),
                                                   ('classifier', Callable[[np.ndarray], float])])

WeakClassifier = NamedTuple('WeakClassifier', [('threshold', float), ('polarity', int), 
                                               ('alpha', float), 
                                               ('classifier', Callable[[np.ndarray], float])])

################WEAK CLASSIFIERS#################

# Function that returns weak classifier value for feature in image

def weak_classifier(x, f, polarity, theta):
    # Returns 1 or 0 (Viola-Jones article)
    return (np.sign((polarity * theta) - (polarity * 
        integral_feature_value(x, f[0], f[1], f[2], f[3], f[4]))) + 1) // 2

# Function that calculated sums that will be used in error calculations
# t_plus -> all pozitive examples weights
# t_minus -> all negative examples weights
# s_plus ->  positive weights below the current example 
# s_minus -> negative weights below the current example
def build_running_sums(ys, ws):
    s_minus, s_plus, t_minus, t_plus, s_minuses, s_pluses = 0., 0., 0., 0., [], []
    
    for y, w in zip(ys, ws):
        if y < .5:
            s_minus += w
            t_minus += w
        else:
            s_plus += w
            t_plus += w
        s_minuses.append(s_minus)
        s_pluses.append(s_plus)
    return t_minus, t_plus, s_minuses, s_pluses

# Function that finds best minimizing threshold and polarity
def find_best_threshold(zs, t_minus, t_plus, s_minuses, s_pluses):
    min_e, min_z, polarity = float('inf'), 0, 0
    for z, s_m, s_p in zip(zs, s_minuses, s_pluses):
        error_1 = s_p + (t_minus - s_m)
        error_2 = s_m + (t_plus - s_p)
        # Minimum value between calculated error is selected
        # And from that polarity and threshold are selected
        if error_1 < min_e:
            min_e, min_z, polarity = error_1, z, -1
        elif error_2 < min_e:
            min_e, min_z, polarity = error_2, z, 1
    return min_z, polarity

# Function that connects thershold calculations
def determine_threshold_polarity(ys, ws, zs): 
    # Sort according to score
    p = np.argsort(zs)
    zs, ys, ws = zs[p], ys[p], ws[p]
    
    # Determine the best threshold: build running sums
    t_minus, t_plus, s_minuses, s_pluses = build_running_sums(ys, ws)
    
    # Determine the best threshold: select optimal threshold.
    return find_best_threshold(zs, t_minus, t_plus, s_minuses, s_pluses)

# Function that applies features to all images
def apply_feature(f, xis, ys, ws):   
    # Selected feature values for all given images
    zs = np.array([integral_feature_value(x, f[0], f[1], f[2], f[3], f[4]) for x in xis])
    # Determine the best threshold and polarity
    min_z, polarity = determine_threshold_polarity(ys, ws, zs)
    result = ThresholdPolarity(threshold=min_z, polarity=polarity)        
    # Determine the classification error
    classification_error = 0.
    for x, y, w in zip(xis, ys, ws):
        h = weak_classifier(x, f, result.polarity, result.threshold)
        classification_error += w * np.abs(h - y)
            
    return ClassifierResult(threshold=result.threshold, polarity=result.polarity, 
                            classification_error=classification_error, classifier=f)

# Function to calculate normalized weights
# Benefitial after weight are changing
def normalize_weights(w):
    return w / w.sum()

# Function that selects defined number of best classifiers
def build_weak_classifiers(prefix, num_features, xis, ys, features):
    # number of negative example (those which are 0)
    m = len(ys[ys < .5])
    # number of positive examples (those which are 1)
    l = len(ys[ys > .5])  
    
    # Creating weight array with 0 values
    ws = np.zeros_like(ys)
    # Calculating weights for positive and negative cases
    ws[ys < .5] = 1./(2.*m)
    ws[ys > .5] = 1./(2.*l)

    total_start_time = datetime.now()
    weak_classifiers = []
    # Build selected number of features
    for t in range(num_features):
        print(f'{t+1}/{num_features} weak classifier:')
        start_time = datetime.now()
        
        # Normalize the weights
        ws = normalize_weights(ws)
        status_counter = STATS_AFTER

        # Selecting best classifier
        best = ClassifierResult(polarity=0, threshold=0,
        classification_error=float('inf'), classifier=None)
        for i, f in enumerate(features):
            # Variables for output
            status_counter -= 1
            improved = False

            # For faster selection features selected randomly with PROBABILITY
            if PART_OF_FEATURES < 1.:
                skip_probability = np.random.random()
                if skip_probability > PART_OF_FEATURES:
                    continue

            # Calculating feature values and comparing with the best founded            
            result = apply_feature(f, xis, ys, ws)
            if result.classification_error < best.classification_error:
                improved = True
                best = result

            # Status updates based on algorithm
            if improved or status_counter == 0:
                current_time = datetime.now()
                duration = current_time - start_time
                total_duration = current_time - total_start_time
                status_counter = STATS_AFTER
                if improved:
                    print(f'Weak classifier {t+1}/{num_features} {total_duration.total_seconds():.2f}s ({duration.total_seconds():.2f}s calculating this feature) {i+1}/{len(features)} features evaluated. Classification error after improving {best.classification_error:.3f} using feature: {str(best.classifier)}')
                else:
                    print(f'Weak classifier {t+1}/{num_features} {total_duration.total_seconds():.2f}s ({duration.total_seconds():.2f}s calculating this feature) {i+1}/{len(features)} features evaluated.')

        # Claculate alpha for strong clasiffier calculations
        beta = best.classification_error / (1 - best.classification_error)
        alpha = np.log(1. / beta)
        
        # Best Weak classifier is saved
        classifier = WeakClassifier(threshold=best.threshold, polarity=best.polarity, classifier=best.classifier, alpha=alpha)
        
        # Update the weights for misclassified examples
        for i, (x, y) in enumerate(zip(xis, ys)):
            h = weak_classifier(x, classifier.classifier, classifier.polarity, classifier.threshold)
            e = np.abs(h - y)
            ws[i] = ws[i] * np.power(beta, 1-e)
            
        # Register this weak classifier           
        weak_classifiers.append(classifier)

        #Saving clasifier in file
        pickle.dump(classifier, open(f'{prefix}-weak-learner-{t+1}-of-{num_features}.pickle', 'wb'))

    print(f'{num_features} weak classifiers built successfully.')
    return weak_classifiers

