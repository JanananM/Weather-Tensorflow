import numpy as np 
#import matplotlib.pyplot as plt
import pandas as pd 

#just ignores the set of warnings about versions you get before output
import warnings
warnings.filterwarnings('ignore')

#Gaussian NB model imports
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

'''
#GoingOutside - label
#Temperature, Weather(Sunny, Rainy, Windy), Day
'''

#import temperature dataset
dataset = pd.read_csv("data.csv")


#analyze data
x = dataset.head()
print(x)

y = dataset.describe()
print(y)

print(dataset.shape)


#feature engineering
def get_combined_data():
    # reading train data
    training = pd.read_csv('data.csv')
    
    # reading test data
    testing = pd.read_csv('data.csv')

    # extracting and then removing the targets from the training data 
    targets = training.GoingOutside
    training.drop(['GoingOutside'], 1, inplace=True)
    

    # merging train data and test data for future feature engineering
    # we'll also remove the Day since this is not an informative feature
    combined = training.append(testing)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'Day'], inplace=True, axis=1)

    #WRITE CODE TO DROP THE GOINGOUTSIDE DATA
    combined.drop(['GoingOutside'], 1, inplace=True)

    return combined 

combined = get_combined_data()

print (combined.head())

def process_weather():
    global combined
    #mapping each weather with a weather value
    combined['Sunny'] = combined['Weather'].map(lambda s: 1 if s == "Sunny" else 0) 
    combined['Rainy'] = combined['Weather'].map(lambda s: 1 if s == "Rainy" else 0) 
    combined['Windy'] = combined['Weather'].map(lambda s: 1 if s == "Windy" else 0) 

    #cleaning out the previous weather data column
    combined.drop('Weather', axis = 1, inplace = True)

    return combined

combined = process_weather()

print (combined.head())

#spliting training and testing data

def compute_score(clf, X, y, scoring = 'accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring = scoring)
    return np.mean(xval)

def recover_train_test_target():
    global combined
    
    targets = pd.read_csv('data.csv', usecols=['GoingOutside'])['GoingOutside'].values
    training = combined.iloc[:30]
    testing = combined.iloc[30:]
    
    return training, testing, targets

training, testing, targets = recover_train_test_target()

#***************************************************************** to remove
#feature selection
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(training, targets)

model = SelectFromModel(clf, prefit=True)
training_reduced = model.transform(training)
print (training_reduced.shape)

testing_reduced = model.transform(testing)
print (testing_reduced.shape)
#****************************************************************** conclusion; just prints the new matrices size after feature selection

#Gaussian NB model

print ('******************************Gaussian NB*************************************')
model = GaussianNB()
model.fit(training,targets) #training data does not have target/label feature

expected = targets
predicted = model.predict(testing)

print("expected")
print(expected)
print("predicted")
print(predicted)

print("classification report\n")
print(metrics.classification_report(expected,predicted))
print("confusion matrix\n")
print(metrics.confusion_matrix(expected,predicted))
print("accuracy score\n")
print(accuracy_score(expected, predicted, normalize=True, sample_weight=None))

#*************************************************************************************

#Random Forest

print ('***********************************Random Forest******************************')
#print ('Cross-validation of : {0}'.format(RandomForestClassifier().__class__)
score = compute_score(clf=RandomForestClassifier(), X=training_reduced, y=targets, scoring='accuracy')
print ('Cross-validation score = {0}'.format(score))

print ('*******************************Predictions based on GNB********************************')

#1 data wanting to predict for future
new_data = dict([('Temperature',37),('Sunny',1),('Rainy',0),('Windy',0)])#dictionary based on filtered data


new_data = pd.Series(new_data).values.reshape(1,-1)
future = model.predict(new_data)
future_int = int(future)

if future_int == 1:
    print("Can Go Outside")
else:
    print("Can't Go Outside")

print ('**************************************************************************************')