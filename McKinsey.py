# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 15:29:18 2018

@author: TEST
"""

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
#Acquire data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#combine = [train_df, test_df]
#Analyze by describing data
print(train_df.columns.values)

# preview the data
train_df.head()
train_df.tail()

#What are the data types for various features?
train_df.info()
#What is the distribution of numerical feature values across the samples?
train_df.describe()
#What is the distribution of categorical features?
train_df.describe(include=['O'])
print(train_df['age_in_days'].value_counts(dropna=False))

#Analyze by visualizing data
g = sns.FacetGrid(train_df, col='renewal')
g.map(plt.hist, 'age_in_days', bins=20)

#Correlating numerical and ordinal features
grid = sns.FacetGrid(train_df, col='renewal', row='sourcing_channel',size=2.2,  aspect=1.6)
grid.map(plt.hist, 'age_in_days', alpha=.5, bins=20)

#Getting more info for each attribute
train_df.premium.describe()
train_df.iloc[:,8].describe()

#convert AGe columns
train_df.loc[:,'age_in_days']=np.ceil(train_df.loc[:,'age_in_days']/360)
test_df.loc[:,'age_in_days']=np.ceil(test_df.loc[:,'age_in_days']/360)

#Convert to categorical variables
for col in ['sourcing_channel','residence_area_type','renewal','age_in_days']:
    train_df[col] = train_df[col].astype('category')

for col in ['sourcing_channel','residence_area_type','age_in_days']:
    test_df[col] = test_df[col].astype('category')
#To find the data distribution
from collections import Counter
print(sorted(Counter(['perc_premium_paid_by_cash_credit']).items()))

#Correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(train_df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
  
#Checking for missing value
train_df.isnull().any().any()

# Plot the histogram /, logx=True, logy=True

csd_attribute=train_df[['perc_premium_paid_by_cash_credit']]
# Compute number of data points: n_data
n_data = len(csd_attribute[~np.isnan(csd_attribute)])
# Number of bins is the square root of number of data points: n_bins
n_bins = np.sqrt(n_data)
# Convert number of bins to integer: n_bins
n_bins = int(n_bins)
train_df['perc_premium_paid_by_cash_credit'].plot(kind='hist', rot=70,bins=n_bins)
# Specify axis labels
plt.xlabel('perc_premium_paid_by_cash_credit')

#Categorical Data display
sns.set(style="darkgrid")
ax = sns.countplot(x="age_in_days", data=train_df)
#Delete Missing Value
train_df=train_df.dropna()
#Create new age categories
age=[]
for val in train_df['age_in_days']:
    if 22<=val<=30:
        age.append(1)
    elif 30<val<=40:
         age.append(2)
    elif 40<val<=50:
         age.append(3)
    elif 50<val<64:
         age.append(4)
    elif 64<val<=70:
         age.append(5)
    else:
         age.append(6)
train_df['age_in_days']=age

testage=[]
for val in test_df['age_in_days']:
    if 22<=val<=30:
        testage.append(1)
    elif 30<val<=40:
         testage.append(2)
    elif 40<val<=50:
         testage.append(3)
    elif 50<val<64:
         testage.append(4)
    elif 64<val<=70:
         testage.append(5)
    else:
         testage.append(6)
test_df['age_in_days']=testage

#Missing data imputation
listfloat_data=['Count_3-6_months_late','Count_6-12_months_late',
                'Count_more_than_12_months_late',
                'application_underwriting_score']
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='median', axis=0)
imputer=imputer.fit(test_df.loc[:,listfloat_data])
test_df.loc[:,listfloat_data]=imputer.transform(test_df.loc[:,listfloat_data])


#Creating dummy variable 
train_df = pd.get_dummies(train_df, columns=['sourcing_channel'])
train_df = pd.get_dummies(train_df, columns=['residence_area_type'])
train_df = pd.get_dummies(train_df, columns=['age_in_days'])

test_df = pd.get_dummies(test_df, columns=['sourcing_channel'])
test_df = pd.get_dummies(test_df, columns=['residence_area_type'])
test_df = pd.get_dummies(test_df, columns=['age_in_days'])

#Remove one of the dummy variable from each category
train_df = train_df.drop('age_in_days_6', 1)
train_df = train_df.drop('residence_area_type_Urban', 1)
train_df = train_df.drop('sourcing_channel_E', 1)

test_df = test_df.drop('age_in_days_6', 1)
test_df = test_df.drop('residence_area_type_Urban', 1)
test_df = test_df.drop('sourcing_channel_E', 1)
#Seperate predictor and target data
Y= train_df['renewal']
X = pd.DataFrame( train_df[train_df.columns.difference(['id', 'renewal'])])

#Feature Scaling
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
X=scaler.fit_transform(X)






testdf=scaler.fit_transform(test_df[test_df.columns.difference(['id'])])


#Smooth Oversampling
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

#SVM Smooth Oversampling
X_resampledsvm, y_resampledsvm = SMOTE(kind='svm').fit_sample(X, Y)
print(sorted(Counter(y_resampledsvm).items()))

#Smooth 'borderline1'
X_resampledb1, y_resampledb1 = SMOTE(kind='borderline1').fit_sample(X, Y)
print(sorted(Counter(y_resampledb1).items()))

#Smooth 'borderline2'
X_resampledb2, y_resampledb2 = SMOTE(kind='borderline2').fit_sample(X, Y)
print(sorted(Counter(y_resampledb2).items()))

#ADASYN Oversampling
X_resampled_ada, y_resampled_ada = ADASYN().fit_sample(X, Y)
print(sorted(Counter(y_resampled_ada).items()))

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampledb1,y_resampledb1 , test_size = 0.30, random_state = 0)

#CLASSIFICATION
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
#classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Calculate Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X, y = Y , cv = 10)
accuracies.mean()
accuracies.std()

y_pred_proba = classifier.predict_proba(X_test)[:,1]

#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state = 0,solver='sag')
classifier_LR.fit(X_train, y_train.ravel())
# Predicting the Test set results
y_pred = classifier_LR.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Calculate Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_LR, X = X, y =Y , cv = 10)
accuracies.mean()

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier_KSVM = SVC(kernel = 'rbf', random_state = 0)
classifier_KSVM.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier_KSVM.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Calculate Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
#ROC 
y_pred_proba = classifier_KSVM.predict_proba(X_test)[:,1]

#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)

#RandomForest Classification
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RF.fit(X_train, y_train)
y_pred = classifier_RF.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Calculate Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_RF, X = X_resampled_under, y = y_resampled_under , cv = 10)
accuracies.mean()
accuracies.std()

y_pred_proba = classifier_RF.predict_proba(X_test)[:,1]

#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)


y_test_proba=classifier_RF.predict_proba(testdf)[:,1]
test_df['prob']=y_test_proba
total=[]
for index, row in test_df.iterrows():
    sumt=((row['prob']+(row['prob']*0.2))*row['premium'])-366
    if sumt<0:
        sumt=0   
    total.append(sumt)
test_df['total_sum']=total
result=test_df[['id','prob','total_sum']]
result.columns=['id','renewal','incentives']
result.to_csv('result.csv',index=False)



import math
test_df.premium.describe()
from sympy.solvers import solve
from sympy import Symbol
s = Symbol('s')
solve((test_df.iloc[0,19]+(test_df.iloc[0,19]*20*(1-math.exp((10*(1-math.exp(-x/400)))/5))))*test_df.iloc[0,8]-x, x)
(test_df.iloc[0,19]+(test_df.iloc[0,19]*20*(1-math.exp((10*(1-math.exp(-x/400)))/5))))*test_df.iloc[0,8]-x

from scipy.optimize import minimize_scalar
def func(x):
    return (test_df.iloc[0,19]+(test_df.iloc[0,19]*20*(1-math.exp((10*(1-math.exp(x/400)))/5)))*test_df.iloc[0,8]

res = minimize_scalar(func)
print(res.x)



solve(((test_df.iloc[0,19]+(test_df.iloc[0,19]*20*(1-math.exp((2*(1-math.exp(s/400)))))))*test_df.iloc[0,8]),x)




x1=((test_df.iloc[0,19]+(test_df.iloc[0,19]*20*(1-math.exp((2*(1-math.exp(s/400)))))))*test_df.iloc[0,8])





















