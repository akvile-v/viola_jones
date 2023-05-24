import random
import numpy as np
from process import *
from weak_classifier import *
from strong_classifier import *
import matplotlib.pyplot as plt

WEAK_CLASSIFIERS_FILE = "weak_classifiers/*"

w = []
#Load weak classifiers from file
for pc in glob.glob(WEAK_CLASSIFIERS_FILE):
    with (open(pc, "rb")) as openfile:
        while True:
            try:
                w.append(pickle.load(openfile))
            except EOFError:
                break

random.seed(20230127)
np.random.seed(20230127)
test_xs, test_ys = sample_data(2000, 2000, True)
test_xis = np.array([integral_image(normalize(to_float_array(x))) for x in test_xs])
true_pos = []
true_neg = []

#Calculating statistics for different stong classifiers
test_strong_2 = np.array([strong_classifier(x, w[0:2]) for x in test_xis])
c = confusion_matrix(test_ys, test_strong_2)
tn, fp, fn, tp = c.ravel()
true_pos.append(tp/2000.*100)
true_neg.append(tn/2000.*100)


test_strong_10 = np.array([strong_classifier(x, w[2:12]) for x in test_xis])
c = confusion_matrix(test_ys, test_strong_10)
tn, fp, fn, tp = c.ravel()
true_pos.append(tp/2000.*100)
true_neg.append(tn/2000.*100)

test_strong_25 = np.array([strong_classifier(x, w[12:37]) for x in test_xis])
c = confusion_matrix(test_ys, test_strong_25)
tn, fp, fn, tp = c.ravel()
true_pos.append(tp/2000.*100)
true_neg.append(tn/2000.*100)

test_strong_40 = np.array([strong_classifier(x, w[37:77]) for x in test_xis])
c = confusion_matrix(test_ys, test_strong_40)
tn, fp, fn, tp = c.ravel()
true_pos.append(tp/2000.*100)
true_neg.append(tn/2000.*100)

test_strong_50 = np.array([strong_classifier(x, w[77:127]) for x in test_xis])
c = confusion_matrix(test_ys, test_strong_50)
tn, fp, fn, tp = c.ravel()
true_pos.append(tp/2000.*100)
true_neg.append(tn/2000.*100)

test_strong_60 = np.array([strong_classifier(x, w[127:187]) for x in test_xis])
c = confusion_matrix(test_ys, test_strong_60)
tn, fp, fn, tp = c.ravel()
true_pos.append(tp/2000.*100)
true_neg.append(tn/2000.*100)

true_pos = true_pos 
true_neg = true_neg
X = ['2','10','25','40', '50', '60']
X_axis = np.arange(len(X))
bottom = min(min(true_neg), min(true_pos))- 10
top = max(max(true_neg), max(true_pos))
plt.ylim(bottom, top)
plt.bar(X_axis - 0.2, true_pos, 0.4, label = 'Teisingai klasifikuoti teigiami pavyzdžiai', color='grey') 
plt.bar(X_axis + 0.2, true_neg, 0.4, label = 'Teisingai klasifikuoti neigiami pavyzdžiai', color='black')
  
plt.xticks(X_axis, X)
plt.xlabel("Ypatybių skaičius stipriame klasifikatoriuje")
plt.ylabel("Procentai")
plt.title("Skirtingų stiprių klasifikatorių klasifikavimo tikslumas")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.show()
