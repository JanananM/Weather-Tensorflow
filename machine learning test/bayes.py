from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
dataset = datasets.load_iris()

print(dataset.data)

print(dataset.target)

model = GaussianNB()
model.fit(dataset.data,dataset.target)

expected = dataset.target
predicted = model.predict(dataset.data)

print(expected)
print(predicted)

print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))



