import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
import numpy as np

url = 'titanic.csv'
dataset = pandas.read_csv(url)

# check if there are missing values in the dataset
dataset.isnull().any()

dataset['survived']=dataset['survived'].fillna(dataset['survived'].mode().iloc[0])

# check if there are missing values in the dataset
dataset.isnull().any()

#Transform dataset
le = preprocessing.LabelEncoder()
dataset = dataset.apply(le.fit_transform)


# check if there are missing values in the dataset
dataset.isnull().any()

# replace missing value with 0
dataset = dataset.fillna(0)

# check if all the missing values have been replaced with 0
dataset.isnull().any()

# Dimensions of Dataset - To see instances and attributes, 1310 x 4 :
print(dataset.shape)

# To see the first 20 rows of the data, since the first row is the header
print(dataset.head(21))

# Statistical Summary - This includes the count, mean, the min and max values as well as some percentiles.
print(dataset.describe())

# Survived Distribution
print(dataset.groupby('survived').size())

# Univariate Plots
# Box plot
dataset.plot(kind='box', subplots=True, sharex=False, sharey=False)
plt.show()

# Histogram
dataset.hist()
plt.show()

# scatter plot matrix
#scatter_matrix(dataset)
#plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,1:14]
Y = array[:,0:1]
#X = transform(X, threshold=None)
validation_size = 0
seed = 0
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
Y_train = np.array(Y_train.ravel()).astype(int)

#Scale features equally - dataset
#Y_train = preprocessing.scale(Y_train)
X_train = preprocessing.scale(X_train)
# Test options and evaluation metric
scoring = 'recall'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(class_weight='balanced')))
models.append(('CART', DecisionTreeClassifier(criterion='entropy')))
models.append(('SVM', svm.SVC(kernel='linear', C=2)))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, shuffle=False, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: mean - %f, std - %f, max - %f, min - %f" % (name, cv_results.mean(), cv_results.std(), cv_results.max(), cv_results.min())
	print(msg)

print(results)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

validation_size = 0.2
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=0)


class Result(object):
    def __init__(self, parameter, val):
        self.parameter = parameter
        self.val = val

		
from sklearn import metrics


#LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, Y_train)


# use the model to predict the labels of the test data
y_pred = clf.predict(X_validation)
y_true = Y_validation
print(y_pred)
print(y_true.tolist())

#metrics
accuracy_score = metrics.accuracy_score(y_true, y_pred)	#Accuracy classification score.
#metrics.auc(x, y)	#Compute Area Under the Curve (AUC) using the trapezoidal rule
#metrics.average_precision_score(y_true, y_score)	#Compute average precision (AP) from prediction scores
metrics.confusion_matrix(y_true, y_pred)	#Compute confusion matrix to evaluate the accuracy of a classification
f1_score = metrics.f1_score(y_true, y_pred)	#Compute the F1 score, also known as balanced F-score or F-measure
precision_score = metrics.precision_score(y_true, y_pred)	#Compute the precision
recall_score = metrics.recall_score(y_true, y_pred)	#Compute the recall
#metrics.roc_auc_score(y_true, y_score)	#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
#metrics.roc_curve(y_true, y_score)	#Compute Receiver operating characteristic (ROC)
zero_one_loss = metrics.zero_one_loss(y_true, y_pred)	#Zero-one classification loss.

print("Model - SVM")
Results = [Result("accuracy_score", accuracy_score), Result("f1_score",f1_score), Result("precision_score",precision_score), Result("recall_score",recall_score), Result("zero_one_loss",zero_one_loss)]
allResults = dict([(p.parameter, p.val) for p in Results ])
print allResults

#DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, Y_train)

# use the model to predict the labels of the test data
y_pred = clf.predict(X_validation)
y_true = Y_validation
print(y_pred)
print(y_true.tolist())
# print(results)

#metrics
accuracy_score = metrics.accuracy_score(y_true, y_pred)	#Accuracy classification score.
#metrics.auc(x, y)	#Compute Area Under the Curve (AUC) using the trapezoidal rule
#metrics.average_precision_score(y_true, y_score)	#Compute average precision (AP) from prediction scores
metrics.confusion_matrix(y_true, y_pred)	#Compute confusion matrix to evaluate the accuracy of a classification
f1_score = metrics.f1_score(y_true, y_pred)	#Compute the F1 score, also known as balanced F-score or F-measure
precision_score = metrics.precision_score(y_true, y_pred)	#Compute the precision
recall_score = metrics.recall_score(y_true, y_pred)	#Compute the recall
#metrics.roc_auc_score(y_true, y_score)	#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
#metrics.roc_curve(y_true, y_score)	#Compute Receiver operating characteristic (ROC)
zero_one_loss = metrics.zero_one_loss(y_true, y_pred)	#Zero-one classification loss.

print("Model - Logistic Regression")
Results = [Result("accuracy_score", accuracy_score), Result("f1_score",f1_score), Result("precision_score",precision_score), Result("recall_score",recall_score), Result("zero_one_loss",zero_one_loss)]
allResults = dict([ (p.parameter, p.val) for p in Results ])
print allResults


#DecisionTreeClassifier
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train, Y_train)

# use the model to predict the labels of the test data
y_pred = clf.predict(X_validation)
y_true = Y_validation
print(y_pred)
print(y_true.tolist())

from sklearn import metrics
accuracy_score = metrics.accuracy_score(y_true, y_pred)	#Accuracy classification score.
#metrics.auc(x, y)	#Compute Area Under the Curve (AUC) using the trapezoidal rule
#metrics.average_precision_score(y_true, y_score)	#Compute average precision (AP) from prediction scores
metrics.confusion_matrix(y_true, y_pred)	#Compute confusion matrix to evaluate the accuracy of a classification
f1_score = metrics.f1_score(y_true, y_pred)	#Compute the F1 score, also known as balanced F-score or F-measure
precision_score = metrics.precision_score(y_true, y_pred)	#Compute the precision
recall_score = metrics.recall_score(y_true, y_pred)	#Compute the recall
#metrics.roc_auc_score(y_true, y_score)	#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
#metrics.roc_curve(y_true, y_score)	#Compute Receiver operating characteristic (ROC)
zero_one_loss = metrics.zero_one_loss(y_true, y_pred)	#Zero-one classification loss.

print("Model - Decision Tree")
Results = [Result("accuracy_score", accuracy_score), Result("f1_score",f1_score), Result("precision_score",precision_score), Result("recall_score",recall_score), Result("zero_one_loss",zero_one_loss)]
allResults = dict([ (p.parameter, p.val) for p in Results ])
print allResults


