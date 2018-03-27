import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cross_validation import cross_val_score
from sklearn import linear_model

features = pd.read_csv('../data/training_set_deepchromaOTI_mfcc_sl=10.csv')

# Switching ~50% of occurrences to false
features_update = features.sample(200)
features_update["y = sim(A,B) > sim(A,C) ^ sim(B,A) > sim(B,C)"] = 0
features.update(features_update)

# Adding difference columns
features['simple_chroma_difference'] = features.apply(
    lambda row: row['simple_chroma_similar'] - row['simple_chroma_dissimilar'], axis=1)
features['simple_mfcc_difference'] = features.apply(
    lambda row: row['simple_mfcc_similar'] - row['simple_mfcc_dissimilar'], axis=1)

# Labels are the values we want to predict
labels = np.array(features['y = sim(A,B) > sim(A,C) ^ sim(B,A) > sim(B,C)'])

# Remove labels from predictors
features = features.drop(
    'y = sim(A,B) > sim(A,C) ^ sim(B,A) > sim(B,C)', axis=1)

# Remove the labels non-predictors from the feature set
features = features.drop('clip_A', axis=1)
features = features.drop('clip_B', axis=1)
features = features.drop('clip_C', axis=1)
features = features.drop('simple_chroma_similar', axis=1)
features = features.drop('simple_chroma_dissimilar', axis=1)
features = features.drop('simple_mfcc_similar', axis=1)
features = features.drop('simple_mfcc_dissimilar', axis=1)
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
# Convert to numpy array
features = np.array(features)

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
