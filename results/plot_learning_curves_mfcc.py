print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import pandas as pd

#np.array([0.1, 0.2, 0.4, 0.6, 0.8])
def plot_learning_curve(estimators_dic, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1])):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
   
    output_file = open("learn-scores-mfcc.csv", "w")

    #For each classifier...
    scores_data = {}
    for classif_name in estimators_dic:
	    print ">> Values for " + classif_name  
	    classifier =  estimators_dic[classif_name]
	
	    train_points, train_scores, test_scores = learning_curve(
		classifier, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

	    #train_scores_mean = np.mean(train_scores, axis=1)
	    #train_scores_std = np.std(train_scores, axis=1)
	    test_scores_mean = np.mean(test_scores, axis=1)
	    test_scores_std = np.std(test_scores, axis=1)

	    scores_data[classif_name] = [test_scores_mean, test_scores_std]

    #Creating plot
    plt.grid()
    formats = [["r", "o-"], ["g", "x-"], ["b", "p-"], ["c", "s-"], ["y", "v-"]]
    index = 0
    for classif_name in estimators_dic:
	    format_to_use = formats[index]
	    scores = scores_data[classif_name]

	    output_file.write(classif_name+","+str(np.max(scores[0]))+","+str(np.min(scores[0]))+","+",".join(str(x) for x in scores[0])+"\n")

	    plt.fill_between(train_points, scores[0] - scores[1],
		             scores[0] + scores[1], alpha=0.1,
		             color=format_to_use[0])
	    #plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
		#             test_scores_mean + test_scores_std, alpha=0.1, color="g")
	    plt.plot(train_points, scores[0], format_to_use[1], color=format_to_use[0], label=classif_name, markersize=12)
	    #plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
	#	     label="Cross-validation score")
	    index = index + 1

    output_file.close()
    plt.legend(loc="best")
    return plt



#Loading data
raw_features = pd.read_csv('../data/balanced_threewise-distances_deepchromaOTI_sl4.csv')
raw_features['diff_mfcc'] = (raw_features['base_similar_mfcc'] - raw_features['base_dissimilar_mfcc'])
raw_features['diff_chroma'] = (raw_features['base_similar_chroma'] - raw_features['base_dissimilar_chroma'])

raw_features.reset_index(drop=True)
features = raw_features[['diff_mfcc', 'ab_gt_ac']]

#digits = load_digits()
#X, y = digits.data, digits.target

title = "Learning Curves"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

#Bayes
bayes = GaussianNB()
#plot_learning_curve(estimator, title, features[['diff_mfcc', 'diff_chroma', 'similar_dissimilar_chroma', 'similar_dissimilar_mfcc']], features[['ab_gt_ac']], ylim=(0.0, 1.01), cv=cv, n_jobs=4)

#Logistic
#title = "Learning Curves (Logistic)"
logistic = linear_model.LogisticRegressionCV(max_iter=10000, solver="saga")#(max_iter=10, solver="liblinear")
#plot_learning_curve(estimator, title, features[['diff_mfcc', 'diff_chroma', 'similar_dissimilar_chroma', 'similar_dissimilar_mfcc']], features[['ab_gt_ac']], ylim=(0.0, 1.01), cv=cv, n_jobs=4)

#Extra Trees
#title = "Learning Curves (Extra Trees)"
extra_trees = ExtraTreesClassifier(n_estimators=60, bootstrap=False, criterion="entropy", max_features="auto", max_depth=None, min_samples_split=4, min_samples_leaf=2, n_jobs=-1)
#plot_learning_curve(estimator, title, features[['diff_mfcc', 'diff_chroma', 'similar_dissimilar_chroma', 'similar_dissimilar_mfcc']], features[['ab_gt_ac']], ylim=(0.0, 1.01), cv=cv, n_jobs=4)

#Random Forest
#title = "Learning Curves (Random Forest)"
random_forest = RandomForestClassifier(n_estimators=1000, bootstrap=False, criterion="gini", max_features="auto", max_depth=None, min_samples_split=32, min_samples_leaf=2, n_jobs=-1)
#RandomForestClassifier(n_estimators=1000, criterion="entropy")
#plot_learning_curve(estimator, title, features[['diff_mfcc', 'diff_chroma', 'similar_dissimilar_chroma', 'similar_dissimilar_mfcc']], features[['ab_gt_ac']], ylim=(0.0, 1.01), cv=cv, n_jobs=4)

#Decision Tree
#title = "Learning Curves (Decision Tree)"
decision_tree = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None, min_samples_split=16, min_samples_leaf=8)
#(criterion="gini", splitter="best", max_depth=None, min_samples_split=4, min_samples_leaf=16)

plot_learning_curve({"Naive Bayes": bayes, "Logistic": logistic, "Extra Trees": extra_trees, "Random Forest": random_forest, "Decision Tree": decision_tree}, title, features[['diff_mfcc']], features[['ab_gt_ac']], ylim=(0.55, 1.01), cv=cv, n_jobs=4)

#title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#estimator = SVC(gamma=0.001)
#plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

f = plt.figure()
plt.show()
f.savefig("all-mfcc.pdf", bbox_inches='tight')


