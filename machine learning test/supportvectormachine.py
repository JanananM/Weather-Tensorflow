from sklearn import svm
from sklearn import datasets
#splits data set into random train and test subsets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#import matplotlib.pyplot as plt

iris = datasets.load_iris()

#type(iris)

print(iris.data)
#print(iris.feature_names)
#print(iris.target)

#stores the features of the iris (three flowers)
x = iris.data[:,2]
#need to predict what species it belongs to
y = iris.target

#split data
#training data and testing data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state = 4)

x_train_mod = x_train.reshape(-1,1)
x_test_mod = x_test.reshape(-1,1)
y_train_mod = y_train.reshape(-1,1)
y_test_mod = y_test.reshape(-1,1)

#plt.plot(x_train_mod, y_train_mod, label = "1D data")

#simplfies equations with too much curves into a simple dataset so you can apply a best fit to the data
model = svm.SVC(kernel ='linear') #asks to define hyperplane(since the dataset 
#is in 4 dimensional due to four attributes of the flowers) to separate data
model.fit(x_train_mod ,y_train_mod.ravel())#conversion to linear model

y_pred_mod = model.predict(x_test_mod)

#how accurate is the unknown flower to knowing which flower it is

print(y_test_mod)#the actual data
print(y_pred_mod)#prediction it makes from unknown test data

print(accuracy_score(y_test_mod,y_pred_mod))