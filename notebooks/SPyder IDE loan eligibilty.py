# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 12:03:38 2025

@author: ND
"""
#!pip install fancyimpute


import warnings
warnings.filterwarnings("ignore")

from fancyimpute import KNN, SoftImpute

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.preprocessing import LabelBinarizer,StandardScaler,OrdinalEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import boxcox
from sklearn.linear_model import LogisticRegression,RidgeClassifier, PassiveAggressiveClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib
import operator

import six
import sys
sys.modules['sklearn.externals.six'] = six

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from sklearn.utils import _safe_indexing
sys.modules['sklearn.utils.safe_indexing'] = sklearn.utils._safe_indexing

from imblearn.over_sampling import SMOTE



#Function to evaluate performance of a model
def classify(est, x, y,X_test,y_test):
    #Passing the model and train test dataset to fit the model
    est.fit(x, y)
    #Predicting the probabilities of the Test data
    y2 = est.predict_proba(X_test)
    y1 = est.predict(X_test)

    print("Accuracy: ", metrics.accuracy_score(y_test, y1))
    print("Area under the ROC curve: ", metrics.roc_auc_score(y_test, y2[:, 1]))
    #Calculate different metrics
    print("F-metric: ", metrics.f1_score(y_test, y1))
    print(" ")
    print("Classification report:")
    print(metrics.classification_report(y_test, y1))
    print(" ")
    print("Evaluation by cross-validation:")
    print(cross_val_score(est, x, y))
    
    return est, y1, y2[:, 1]


#Function to find which features are more important than others through model
def feat_importance(estimator):
    feature_importance = {}
    for index, name in enumerate(df_LC.columns):
        feature_importance[name] = estimator.feature_importances_[index]

    feature_importance = {k: v for k, v in feature_importance.items()}
    sorted_x = sorted(feature_importance.items(), key=operator.itemgetter(1), reverse = True)
    
    return sorted_x

#Model to  predict the ROC curve for various models and finding the best one
def run_models(X_train, y_train, X_test, y_test, model_type='Non-balanced'):
    clfs = {
        'GradientBoosting': GradientBoostingClassifier(max_depth=6, n_estimators=100, max_features=0.3),
        'LogisticRegression': LogisticRegression(),
        # 'GaussianNB': GaussianNB(),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=10),
        'XGBClassifier': XGBClassifier()
    }

    cols = ['model_type', 'model', 'matthews_corrcoef', 'roc_auc_score', 'precision_score', 'recall_score', 'f1_score']
    models_report = pd.DataFrame(columns=cols)
    conf_matrix = dict()

    for clf_name, clf in clfs.items():
        clf.fit(X_train, y_train)
        #y_pred = clf.predict(X_test)
        #y_score = clf.predict_proba(X_test)[:, 1]
        y_score = clf.predict_proba(X_test)[:, 1]
        custom_threshold = 0.6
        y_pred = (y_score > custom_threshold).astype(int)


        print(f'computing {clf_name} - {model_type}')

        tmp = pd.DataFrame([{
    'model_type': model_type,
    'model': clf_name,
    'roc_auc_score': metrics.roc_auc_score(y_test, y_score),
    'matthews_corrcoef': metrics.matthews_corrcoef(y_test, y_pred),
    'precision_score': metrics.precision_score(y_test, y_pred),
    'recall_score': metrics.recall_score(y_test, y_pred),
    'f1_score': metrics.f1_score(y_test, y_pred)
}])


        models_report = pd.concat([models_report, tmp], ignore_index=True)


        conf_matrix[clf_name] = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=False)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, drop_intermediate=False, pos_label=1)
        plt.figure(1, figsize=(6, 6))
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC curve - {model_type}')
        plt.plot(fpr, tpr, label=clf_name)
        plt.legend(loc=2, prop={'size': 11})

    plt.plot([0, 1], [0, 1], color='black')
    #plt.show()

    return models_report, conf_matrix



#Importing the datasets
data =pd.read_csv("LoansTrainingSetV2.csv")

#view data frame and length
data.head()
print(len(data))

##Drop the duplicates with respect to LOAN ID
data.drop_duplicates(subset="Loan ID",keep='first',inplace=True)

##########PLotting the loan status 
status=data["Loan Status"].value_counts() 

plt.figure(figsize=(10,5))
sns.barplot(x=status.index, y=status.values, alpha=0.8)
plt.title('Loan Status distribution')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Loan Status', fontsize=12)
plt.show()


###Current Loan Amount ############
data["Current Loan Amount"].describe()


data["Current Loan Amount"].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Commute Times for 1,000 Commuters')
plt.xlabel('Counts')
plt.ylabel('Commute Time')
plt.grid(axis='y', alpha=0.75)


#Finding IQR's for outlier removal
Q1 = data["Current Loan Amount"].quantile(0.25)
Q3 = data["Current Loan Amount"].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

#identify outliers and current loan amount
data["Current Loan Amount"][((data["Current Loan Amount"] < (Q1 - 1.5 * IQR)) |(data["Current Loan Amount"] > (Q3 + 1.5 * IQR)))]

#clean up extreme or placeholder values in the "Current Loan Amount" column of a DataFrame called data.
temp=np.array(data["Current Loan Amount"].values.tolist())
data["Current Loan Amount_temp"] = np.where(temp > 9999998, 'NaN', temp).tolist()


#cleaning and visualizing the "Current Loan Amount_temp" data
temp=data["Current Loan Amount_temp"][data["Current Loan Amount_temp"]!='NaN'].astype(str).astype(int)
temp.plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.show()
                   
temp.describe()

#Replacing the data with 50% percentile or mean
temp=np.array(data["Current Loan Amount"].values.tolist())
data["Current Loan Amount"] = np.where(temp > 9999998,12038,temp).tolist()

data=data.drop(['Current Loan Amount_temp'],axis=1)

############Term ##############

status=data["Term"].value_counts() 

plt.figure(figsize=(10,5))
sns.barplot(x=status.index, y=status.values, alpha=0.8)
plt.title('Loan Term distribution')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Loan term', fontsize=12)
plt.show()


#####Credit Score#############

##Max is 7510. It should be between 0-800

data["Credit Score"].describe()

sns.boxplot(x=data["Credit Score"])
plt.show()

data["Credit Score"].isnull().sum()

#treat the data at hand. Let us firstly divide the values greater than 800 by 10

data["Credit Score"]=np.where(data["Credit Score"]>800, data["Credit Score"]/10, data["Credit Score"])

#Now lets replace the missing values with median 
median_score=statistics.median(data["Credit Score"])

data["Credit Score_1"]=data["Credit Score"]
data["Credit Score_1"].fillna(median_score, inplace = True) 

sns.distplot(data["Credit Score_1"])
plt.show()


#As we can see this data is skewed so when we replace it with median it is giving us problems. 
#Replacing with 75th percentile and taking log we get a better distribution

data["Credit Score"].fillna(741, inplace = True) 

sns.distplot(data["Credit Score"])
sns.distplot(np.log(data["Credit Score"]))
plt.show()

#visualize home ownership column
sns.countplot(x="Home Ownership", data=data, palette="pastel")
plt.title("Frequency of Home Ownership Types")
plt.xlabel("Home Ownership")
plt.ylabel("Count")
plt.xticks(rotation=45)  # Optional: rotates labels for readability
plt.show()
data['Home Ownership'].unique()


#As we can see it has Home Mortgage and haveMortgage as 2 different classes. Lets fix that

data['Home Ownership']=data['Home Ownership'].str.replace('HaveMortgage', 'Home Mortgage', regex=True)
data['Home Ownership'].unique()

#visualize again
sns.countplot(x="Home Ownership", data=data, palette="pastel")
plt.title("Frequency of Home Ownership Types")
plt.xlabel("Home Ownership")
plt.ylabel("Count")
plt.xticks(rotation=45)  # Optional: rotates labels for readability
plt.show()


##################Annual Income######################
data['Annual Income'].describe()

##Lets look at the quantiles of this columns
data['Annual Income'].quantile([.2,0.75,0.90,.95,0.99,.999])

#As we can see they lie in the 99th percentile of the data.Lets replace them
# Capping any values greater than 99% to 99th value
data.loc[data['Annual Income'] > 239287, 'Annual Income'] = 239287

data['Annual Income'].isna().sum()
#So we have about 21000 null values 

##We will impute the mising data with other columns towards the end


###############Loan Purpose ###############

data['Purpose'].value_counts()
#So other and Other mean the same thing. Lets make it the same

data['Purpose']=data['Purpose'].str.replace('Other', 'other', regex=True)


#######Monthly debt ###############
#data['Monthly Debt'].describe()
##So this is not numeric column. Lets explore

data['Monthly Debt'] 
# But this should be a numeric column. So lets convert it to float

#pd.to_numeric(data['Monthly Debt'] )
data['Monthly Debt'].unique()

data['Monthly Debt']=data['Monthly Debt'].str.replace('$', '')

data['Monthly Debt']=pd.to_numeric(data['Monthly Debt'] )

sns.distplot(data["Monthly Debt"])
plt.show()

#We can see that there are outliers in this data because of the plot
#Lets explore

data['Monthly Debt'].describe()
#The max value is too high here

data['Monthly Debt'].quantile([.2,0.75,0.90,.95,0.99,.999])
#Problem is with 99th percentile. lets dig deeper

data['Monthly Debt'].quantile([0.9995,.9999])
#So problem again is wit 99th percentile

data['Monthly Debt'].quantile([0.9997,.99999])
#0.99970     5978.574911
#0.99999    13262.762330

data['Monthly Debt'].quantile([0.999,1])

#Need to replace this
data.loc[data['Monthly Debt'] > 4926, 'Monthly Debt'] = 4926

sns.distplot(data["Monthly Debt"])
plt.show()
#Now we get the right distribution


####################Years of credit history #################

data['Years of Credit History'].value_counts()
sns.distplot(data["Years of Credit History"])
plt.show()
#Over all looks pretty clean! no need of doing anything

data['Months since last delinquent'].describe()

#Lets check if there are any NA's
data['Months since last delinquent'].isna().sum()
#We have nearly 48506 NA;s. We will try to handle them at last 

##############Number of open accounts ##############
data['Number of Open Accounts'].describe()
#The max number seems odd. Lets investigate

sns.distplot(data['Number of Open Accounts'])
plt.show()
#Yes there are outliers in this columns. Let dig deeper

data['Number of Open Accounts'].quantile([0.75,0.999,1])
#Ok so replacing anything greater than 99th percentile with 99th percentile values

data.loc[data['Number of Open Accounts'] > 36, 'Number of Open Accounts'] = 36
sns.distplot(data['Number of Open Accounts'])
plt.show()
#Looks good now


#######################Number of Credit problems##############
data['Number of Credit Problems'].describe() 
#Max looks a bit higher. Lets see
sns.distplot(data['Number of Credit Problems'])
plt.show()

#Okay lets look at value _counts
data['Number of Credit Problems'].value_counts()
#Okay looks good

##################Current Credit Balance###########
data['Current Credit Balance'].describe()
sns.distplot(data['Current Credit Balance'])
plt.show()
#It seems there are outliers in this data. Lets investigate

data['Current Credit Balance'].quantile([0.75,0.95,0.999,1])

#lets dig deeper
data['Current Credit Balance'].quantile([0.95,0.96,0.97,0.98,0.99,1])

#So lets replace it with 99th percentile
data.loc[data['Current Credit Balance'] > 81007, 'Current Credit Balance'] = 81007

sns.distplot(data['Current Credit Balance']**(1/2))
plt.show()
#The plot doesnt look good. We need to transform it

#lets use squareroot transformation
data['Current Credit Balance']=data['Current Credit Balance']**(1/2)
sns.distplot(data['Current Credit Balance'])
plt.show()
#######################Max open credit################
data['Maximum Open Credit'].describe()

data['Maximum Open Credit'].value_counts()

#sns.distplot(data['Maximum Open Credit'])
#plt.show()
#could not convert string to float: '#VALUE!'

#pd.to_numeric(data['Maximum Open Credit'])
#Unable to parse string "#VALUE!" at position 4930

#Lets replace #value with Nan
data['Maximum Open Credit']=data['Maximum Open Credit'].replace('#VALUE!', np.nan, regex=True)

data['Maximum Open Credit']=pd.to_numeric(data['Maximum Open Credit'])

data['Maximum Open Credit'].isnull().sum()

#Now we have only 2 Nan;s in the data. Lets replace them with mean 
mean_value = data['Maximum Open Credit'].mean()
mean_value
data['Maximum Open Credit']=data['Maximum Open Credit'].fillna(35965)

data['Maximum Open Credit'].quantile([0.55,0.76,0.87,0.98,0.99,1])
#0.55       24036.85
#0.76       37260.00
#0.87       50737.00
#0.98      114682.88
#0.99      171423.93
#1.00    69988086.00

#Lets replace the outliers with 0.99 quantile value
data.loc[data['Maximum Open Credit'] > 171423, 'Maximum Open Credit'] = 171423

###############Bankruptcies##########
data['Bankruptcies'].describe()
data['Bankruptcies'].value_counts()

data['Bankruptcies'].unique()
#So we have Nan's. Lets fill them with median

data['Bankruptcies']=data['Bankruptcies'].fillna(3)
data['Bankruptcies'].unique()

###Looks good
####Tax Liens######
data['Tax Liens'].describe()


#Categorical variables
cat_cols = ['Term','Years in current job','Home Ownership','Purpose']

for c in cat_cols:
    data[c] = pd.factorize(data[c])[0]

#Imputing missing data with soft impute
updated_data=pd.DataFrame(data=SoftImpute().fit_transform(data[data.columns[3:19]],), columns=data[data.columns[3:19]].columns, index=data.index)
#Getting the dataset ready pd.get dummies function for dropping the dummy variables
df_LC = pd.get_dummies(updated_data, drop_first=True)


#Binarizing the Target variable
lb_style = LabelBinarizer()
lb_results = lb_style.fit_transform(data['Loan Status'])
y=lb_results
y=y.ravel()


#Scaling the independent variables
X_scaled = preprocessing.scale(df_LC)
print(X_scaled)
print('   ')
print(X_scaled.shape)

#######Looking at other models using different classifiers
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=22)

#Finding accuracy and feature importance using XGB classifier
xgb0, y_pred_b, y_pred2_b = classify(XGBClassifier(), X_train, y_train,X_test,y_test)
print(xgb0.feature_importances_)
plot_importance(xgb0)
pyplot.show()
feat1 = feat_importance(xgb0)

# Unpack into two lists
features, importances = zip(*feat1)

# Plot importance on a bar plot
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Gradient Boosting Feature Importances')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()

#######XGB classifier ################
xgb0, y_pred_b, y_pred2_b = classify(XGBClassifier(n_estimators=47, learning_rate=0.015), X_train, y_train,X_test,y_test)



#######K nearest Neighbour classifier ################
knc, y_p, y_p2 = classify(KNeighborsClassifier(), X_train, y_train,X_test,y_test)

########Logistic Regression ##############
logit, y_p, y_p2 = classify(LogisticRegression(), X_train, y_train,X_test,y_test)

########Decision Tree Classifier ##########
dtc, y_p, y_p2 = classify(DecisionTreeClassifier(), X_train, y_train,X_test,y_test)

#model evaluation metric for xgboos, LR, gradient boosting, random forest
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)
models_report, conf_matrix = run_models(X_train, y_train, X_test, y_test, model_type = 'Non-balanced')
print(models_report)
print(models_report.to_string())

plt.show()


###########Synthetically balancing the dataset##################
index_split = int(len(X_scaled)/2)
X_train, y_train = SMOTE().fit_resample(X_scaled[0:index_split, :], y[0:index_split])
X_test, y_test = X_scaled[index_split:], y[index_split:]
models_report_bal, conf_matrix_bal = run_models(X_train, y_train, X_test, y_test, model_type = 'Balanced')
models_report_bal
plt.show()


################Now we  know that GBM model performed the best so 
# save model
gbm=GradientBoostingClassifier(max_depth= 6, n_estimators=100, max_features = 0.3)
gbm.fit(X_scaled, y)
joblib.dump(gbm, 'GBM_Model_version.pkl')

# load model
gbm_pickle = joblib.load('GBM_Model_version.pkl')