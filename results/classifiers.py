import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import linear_model

features = pd.read_csv('../data/balanced_threewise-distances_deepchromaOTI_sl4.csv')

# Switching ~50% of occurrences to false
# features_update = features.sample(200)
# features_update["y = sim(A,B) > sim(A,C) ^ sim(B,A) > sim(B,C)"] = 0
# features.update(features_update)

# Adding difference columns
# features['simple_chroma_difference'] = features.apply(
#     lambda row: row['simple_chroma_similar'] - row['simple_chroma_dissimilar'], axis=1)
# features['simple_mfcc_difference'] = features.apply(
#     lambda row: row['simple_mfcc_similar'] - row['simple_mfcc_dissimilar'], axis=1)

# Labels are the values we want to predict
labels = np.array(features['ab_gt_ac'])

# Remove labels from predictors
features = features.drop(
    'ab_gt_ac', axis=1)

# Remove the labels non-predictors from the feature set
# features = features.drop('clip_A', axis=1)
# features = features.drop('clip_B', axis=1)
# features = features.drop('clip_C', axis=1)
# features = features.drop('simple_chroma_similar', axis=1)
# features = features.drop('simple_chroma_dissimilar', axis=1)
# features = features.drop('simple_mfcc_similar', axis=1)
# features = features.drop('simple_mfcc_dissimilar', axis=1)
# features= features.drop('simple_chroma_AB', axis = 1)
# features= features.drop('simple_chroma_AC', axis = 1)
# features= features.drop('simple_chroma_BC', axis = 1)
# features= features.drop('simple_mfcc_AB', axis = 1)
# features= features.drop('simple_mfcc_AC', axis = 1)
# features= features.drop('simple_mfcc_BC', axis = 1)
# features= features.drop('simple_chroma(A,B) - simple_chroma(A,C)', axis = 1)
# features= features.drop('simple_mfcc(A,B) - simple_mfcc(A,C)', axis = 1)
# features= features.drop('(simple_chroma(A,B) - simple_chroma(A,C)) / simple_chroma(B,C)', axis = 1)
# features= features.drop('(simple_mfcc(A,B) - simple_mfcc(A,C)) / simple_mfcc(B,C)', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)
print feature_list
# Convert to numpy array
features = np.array(features)

#====== David
parameters_dic = { "Extra Trees" : {
	    'n_estimators': [20, 40, 60, 1000],
	    'criterion': ['entropy'],
	    'min_samples_split': [2, 4, 8, 16, 32],
	    'min_samples_leaf': [2, 4, 8, 16, 32]
	},
	"Random Forest" : {
	    'n_estimators': [20, 40, 60, 1000],
	    'criterion': ['entropy'],
	    'min_samples_split': [2, 4, 8, 16, 32],
	    'min_samples_leaf': [2, 4, 8, 16, 32]
	}
	}
classifiers_names = ["Extra Trees", "Random Forest"]#, "Nearest Neighbors", "RBF SVM", "Naive Bayes", "Linear SVM"]
classifiers = [ExtraTreesClassifier(n_jobs=-1, criterion='entropy'), RandomForestClassifier(n_jobs=-1)]

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20)
for classifier_index in range(0, len(classifiers)):
	parameters_to_optimize = parameters_dic[classifiers_names[classifier_index]]
	best_clf = None
	best_f1 = []

	for train, test in StratifiedKFold(train_labels, n_folds=5):
		X_train_scaled = train_features[train]
		answer_train = train_labels[train]
		X_test_scaled = train_features[test]
		answer_test = train_labels[test]

		classifier = GridSearchCV(classifiers[classifier_index], param_grid=parameters_to_optimize, cv=3)
		clf = classifier.fit(X_train_scaled, answer_train)
		
		y_pred = clf.predict(X_test_scaled)
		f1_micro = f1_score(answer_test, y_pred, average='micro')
		f1_macro = f1_score(answer_test, y_pred, average='macro')

		#Storing the best configuration
		if len(best_f1) == 0 or f1_micro > best_f1[0]:
			best_f1 = [f1_micro, f1_macro]
			best_clf = clf.best_estimator_

	best_clf.fit(train_features, train_labels)#Fitting for test
	accuracy = best_clf.score(test_features, test_labels)#Accuracy
	y_pred = best_clf.predict(test_features)#Estimated values
	metrics_macro = precision_recall_fscore_support(test_labels, y_pred, average='macro')#Calculates for each label and compute the mean!
	metrics_micro = precision_recall_fscore_support(test_labels, y_pred, average='micro')

	print "CONF " + str(classifiers_names[classifier_index]) + str(best_clf.n_estimators) + "\t" + str(best_clf.max_features) + "\t" + str(best_clf.max_depth)+ "\t" + str(best_clf.min_samples_split)+ "\t" + str(best_clf.min_samples_leaf)
	#CONF Extra Trees1000	auto	None	4	2
		
#==========

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.20)


# print('Training Features Shape:', train_features.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing Features Shape:', test_features.shape)
# print('Testing Labels Shape:', test_labels.shape)

# Instantiate model with 1000 decision trees
et = ExtraTreesClassifier(n_estimators=1000)
rf = RandomForestClassifier(n_estimators=1000)
logreg = linear_model.LogisticRegression()

# Train the models on training data
et.fit(train_features, train_labels)
rf.fit(train_features, train_labels)
logreg.fit(train_features, train_labels)

print "############### Random Forest Results ###############"
test_scores = cross_val_score(rf, features, labels, cv=5, scoring='f1_macro')
print "Test Accuracy :: ", test_scores.mean()
train_scores = cross_val_score(
    rf, train_features, train_labels, cv=5, scoring='f1_macro')
print "Train Accuracy :: ", accuracy_score(train_labels, rf.predict(train_features))

print "############### Extra Trees Results ###############"
# Use the forest's predict method on the test data
test_scores = cross_val_score(et, features, labels, cv=5, scoring='f1_macro')
print "Test Accuracy :: ", test_scores.mean()
train_scores = cross_val_score(
    et, train_features, train_labels, cv=5, scoring='f1_macro')
print "Train Accuracy :: ", accuracy_score(train_labels, et.predict(train_features))

print "############### Logistic Regression Results ###############"
test_scores = cross_val_score(
    logreg, features, labels, cv=5, scoring='f1_macro')
print "Test Accuracy :: ", test_scores.mean()
train_scores = cross_val_score(
    logreg, train_features, train_labels, cv=5, scoring='f1_macro')
print "Train Accuracy :: ", accuracy_score(train_labels, logreg.predict(train_features))
