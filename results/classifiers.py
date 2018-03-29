# coding=utf-8

import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB


if __name__ == "__main__": 

	if len(sys.argv) < 2:
		print "Uso: <mode: tune or classify>"
		sys.exit(1)

	mode = sys.argv[1]
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
	features = features.drop('ab_gt_ac', axis=1)

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
	#x_features, test_features, y_labels, test_labels = train_test_split(features, labels, test_size=0.20, train_size=0.8)
	train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, train_size=0.8)

	if mode == "tune":
		parameters_dic = { "Extra Trees" : {
	    		'n_estimators': [20, 40, 60, 1000],
	    		'criterion': ['entropy', 'gini'],
	    		'min_samples_split': [2, 4, 8, 16, 32],
	    		'min_samples_leaf': [2, 4, 8, 16, 32],
			'bootstrap' : [True, False]
			},
			"Random Forest" : {
			    'n_estimators': [20, 40, 60, 1000],
			    'criterion': ['entropy', 'gini'],
			    'min_samples_split': [2, 4, 8, 16, 32],
			    'min_samples_leaf': [2, 4, 8, 16, 32],
			    'bootstrap': [True, False]
			},
			"Decision Tree": {
			    'splitter': ["best", "random"],
			    'criterion': ['entropy', 'gini'],
			    'min_samples_split': [2, 4, 8, 16, 32],
			    'min_samples_leaf': [1, 2, 4, 8, 10, 12, 14, 16, 32]
			},
			"Logistic": {
				#'Cs': list(np.power(10.0, np.arange(-10, 10))),
				'max_iter': [10, 100, 1000, 10000], 
				'solver': ["liblinear", 'newton-cg', 'sag', 'saga']
			}
		}
		classifiers_names = ["Extra Trees", "Random Forest", "Decision Tree", "Logistic"]#, "Nearest Neighbors", "RBF SVM", "Naive Bayes", "Linear SVM"]
		classifiers = [ExtraTreesClassifier(n_jobs=-1, criterion='entropy'), RandomForestClassifier(n_jobs=-1), DecisionTreeClassifier(), linear_model.LogisticRegressionCV(cv=5)]

		for classifier_index in range(2, len(classifiers)):
			parameters_to_optimize = parameters_dic[classifiers_names[classifier_index]]
			best_clf = None
			best_f1 = []

			#for index in range(0,5):
			for train, test in StratifiedKFold(train_labels, n_folds=5):
				X_train_scaled = train_features[train]
				answer_train = train_labels[train]
				X_test_scaled = train_features[test]
				answer_test = train_labels[test]
				#train_features, valid_features, train_labels, valid_labels = train_test_split(x_features, y_labels, test_size=0.25, train_size=0.75)
				#X_train_scaled = train_features
				#answer_train = train_labels
				#X_test_scaled = valid_features
				#answer_test = valid_labels

				classifier = GridSearchCV(classifiers[classifier_index], param_grid=parameters_to_optimize, cv=3, n_jobs=-1)
				clf = classifier.fit(X_train_scaled, answer_train)
		
				y_pred = clf.predict(X_test_scaled)
				f1_micro = f1_score(answer_test, y_pred, average='micro')
				f1_macro = f1_score(answer_test, y_pred, average='macro')

				#Storing the best configuration
				if len(best_f1) == 0 or f1_micro > best_f1[0]:
					best_f1 = [f1_micro, f1_macro]
					best_clf = clf.best_estimator_

		#train_features = x_features
		#train_labels = y_labels

		best_clf.fit(train_features, train_labels)#Fitting for test
		accuracy = best_clf.score(test_features, test_labels)#Accuracy
		y_pred = best_clf.predict(test_features)#Estimated values
		metrics_macro = precision_recall_fscore_support(test_labels, y_pred, average='macro')#Calculates for each label and compute the mean!
		metrics_micro = precision_recall_fscore_support(test_labels, y_pred, average='micro')

		if classifiers_names[classifier_index] == classifiers_names[0] or classifiers_names[classifier_index] == classifiers_names[1]:
			print "CONF " + str(classifiers_names[classifier_index]) + "\t" + str(best_clf.criterion) + "\t" + str(best_clf.bootstrap) + "\t" + str(best_clf.n_estimators) + "\t" + str(best_clf.max_features) + "\t" + str(best_clf.max_depth)+ "\t" + str(best_clf.min_samples_split)+ "\t" + str(best_clf.min_samples_leaf)
			classifiers[classifier_index].criterion = best_clf.criterion
			classifiers[classifier_index].bootstrap = best_clf.bootstrap
			classifiers[classifier_index].n_estimators = best_clf.n_estimators
			classifiers[classifier_index].max_features = best_clf.max_features
			classifiers[classifier_index].max_depth = best_clf.max_depth
			classifiers[classifier_index].min_samples_split = best_clf.min_samples_split
			classifiers[classifier_index].min_samples_leaf = best_clf.min_samples_leaf
		elif classifiers_names[classifier_index] == "Decision Tree":
			print "CONF " + str(classifiers_names[classifier_index]) + "\t" + str(best_clf.criterion) + "\t" + str(best_clf.splitter) + "\t" + str(best_clf.max_depth)+ "\t" + str(best_clf.min_samples_split)+ "\t" + str(best_clf.min_samples_leaf)
			classifiers[classifier_index].criterion = best_clf.criterion
			classifiers[classifier_index].splitter = best_clf.splitter
			classifiers[classifier_index].max_depth = best_clf.max_depth
			classifiers[classifier_index].min_samples_split = best_clf.min_samples_split
			classifiers[classifier_index].min_samples_leaf = best_clf.min_samples_leaf
		else:
			print "CONF " + str(classifiers_names[classifier_index]) + "\t" + str(best_clf.max_iter) + "\t" + str(best_clf.solver)
			classifiers[classifier_index].max_iter = best_clf.max_iter
			classifiers[classifier_index].solver = best_clf.solver
		#CONF Extra Trees	entropy	False	20	auto	None	4	2
		#CONF Random Forest	entropy	True	60	auto	None	8	4
		#CONF Decision Tree	gini	random	None	2	8
		#CONF Decision Tree	gini	random	None	2	16
		#CONF Decision Tree	gini	random	None	4	16
		#CONF Logistic	10	liblinear
		
		
	# Split the data into training and testing sets
	#train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20)

	# print('Training Features Shape:', train_features.shape)
	# print('Training Labels Shape:', train_labels.shape)
	# print('Testing Features Shape:', test_features.shape)
	# print('Testing Labels Shape:', test_labels.shape)

	# Instantiate model with 1000 decision trees
	et = classifiers[0]#ExtraTreesClassifier(n_estimators=20, bootstrap=False, criterion="entropy", max_features="auto", max_depth=None, min_samples_split=4, min_samples_leaf=2, n_jobs=-1)
	rf = classifiers[1]#RandomForestClassifier(n_estimators=60, bootstrap=True, criterion="entropy", max_features="auto", max_depth=None, min_samples_split=8, min_samples_leaf=4, n_jobs=-1)#RandomForestClassifier(n_estimators=1000, criterion="entropy")
	dt = classifiers[2]#DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None, min_samples_split=4, min_samples_leaf=16)
	logreg_cv = classifiers[3]#linear_model.LogisticRegressionCV(cv=5, solver="liblinear")

	logreg = linear_model.LogisticRegression()
	bayes = GaussianNB()

	# Train the models on training data
	dt.fit(train_features, train_labels)
	et.fit(train_features, train_labels)
	rf.fit(train_features, train_labels)
	logreg.fit(train_features, train_labels)
	logreg_cv.fit(train_features, train_labels)
	bayes.fit(train_features, train_labels)

	acc_dt = dt.score(test_features, test_labels)
	acc_et = et.score(test_features, test_labels)
	acc_rf = rf.score(test_features, test_labels)
	acc_logreg = logreg.score(test_features, test_labels)
	acc_logreg_cv = logreg_cv.score(test_features, test_labels)
	acc_bayes = bayes.score(test_features, test_labels)

	output_file = open("accuracies.csv", "w")
	output_file.write("classifier,accuracy\n")
	output_file.write("DecisionTree,"+str(acc_dt)+"\n")
	output_file.write("ExtraTree,"+str(acc_et)+"\n")
	output_file.write("RandomForest,"+str(acc_rf)+"\n")
	output_file.write("Logistic,"+str(acc_logreg)+"\n")
	output_file.write("LogisticCV,"+str(acc_logreg_cv)+"\n")
	output_file.write("Bayes,"+str(acc_bayes)+"\n")
	output_file.close()

	
	#print "############### Decision Tree Results ###############"
	#test_scores = cross_val_score(dt, features, labels, cv=5, scoring='f1_macro')
	#print "Test Accuracy :: ", test_scores.mean()
	#train_scores = cross_val_score(
	 #   dt, train_features, train_labels, cv=5, scoring='f1_macro')
	#print "Train Accuracy :: ", accuracy_score(train_labels, dt.predict(train_features))

	#print "############### Random Forest Results ###############"
	#test_scores = cross_val_score(rf, features, labels, cv=5, scoring='f1_macro')
	#print "Test Accuracy :: ", test_scores.mean()
	#train_scores = cross_val_score(
	#    rf, train_features, train_labels, cv=5, scoring='f1_macro')
	#print "Train Accuracy :: ", accuracy_score(train_labels, rf.predict(train_features))

	#print "############### Extra Trees Results ###############"
	# Use the forest's predict method on the test data
	#test_scores = cross_val_score(et, features, labels, cv=5, scoring='f1_macro')
	#print "Test Accuracy :: ", test_scores.mean()
	#train_scores = cross_val_score(
	#    et, train_features, train_labels, cv=5, scoring='f1_macro')
	#print "Train Accuracy :: ", accuracy_score(train_labels, et.predict(train_features))

	#print "############### Logistic Regression CV Results ###############"
	#test_scores = cross_val_score(
	#    logreg_cv, features, labels, cv=5, scoring='f1_macro')
	#print "Test Accuracy :: ", test_scores.mean()
	#train_scores = cross_val_score(
	#    logreg_cv, train_features, train_labels, cv=5, scoring='f1_macro')
	#print "Train Accuracy :: ", accuracy_score(train_labels, logreg_cv.predict(train_features))

	#print "############### Logistic Regression Results ###############"
	#test_scores = cross_val_score(
	#    logreg, features, labels, cv=5, scoring='f1_macro')
	#print "Test Accuracy :: ", test_scores.mean()
	#train_scores = cross_val_score(
	#    logreg, train_features, train_labels, cv=5, scoring='f1_macro')
	#print "Train Accuracy :: ", accuracy_score(train_labels, logreg.predict(train_features))

	#print "############### Naive Bayes Results ###############"
	#test_scores = cross_val_score(
	#    bayes, features, labels, cv=5, scoring='f1_macro')
	#print "Test Accuracy :: ", test_scores.mean()
	#train_scores = cross_val_score(
	#    bayes, train_features, train_labels, cv=5, scoring='f1_macro')
	#print "Train Accuracy :: ", accuracy_score(train_labels, bayes.predict(train_features))
