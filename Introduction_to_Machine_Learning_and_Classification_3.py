import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


SEED = 5
np.random.seed(SEED)


uri= 'https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv'

datas = pd.read_csv(uri)

change = {
    0 : 1,
    1 : 0
}
datas ['finished'] = datas.unfinished.map(change)
head = datas.head()
print(head)
tail = datas.tail()
print(tail)

#sns.scatterplot(data = datas, x = "expected_hours", y = "price", hue="finished")

#The graphics can be separated
#sns.relplot(data = datas, x = "expected_hours", y = "price", hue="finished", col="finished")

x = datas[['expected_hours', 'price']]
y = datas['finished']

rawTrainX, rawTestX, trainY, testY = train_test_split(x, y, test_size = 0.25, stratify = y)

scaler = StandardScaler()
scaler.fit(rawTrainX)
trainX = scaler.transform(rawTrainX)
testX = scaler.transform(rawTestX)

model = SVC(gamma='auto')
model.fit(trainX, trainY)
predictions = model.predict(testX)
accuracyScore = accuracy_score(testY, predictions)
print("The accuracy was: %.2f " % (accuracyScore * 100))

#sns.relplot(data = testX, x = "expected_hours", y = "price", hue = testY)
#plt.show()

dataX = testX[:, 0]
dataY = testX[:, 1]

xMin = dataX.min()
xMax = dataX.max()
yMin = dataY.min()
yMax = dataY.max()
print(xMin, xMax, yMin, yMax)

pixels = 100
horizontalAxis = np.arange(xMin, xMax, (xMax - xMin) / pixels)
print(horizontalAxis)

verticalAxis = np.arange(yMin, yMax, (yMax - yMin) / pixels)
print(verticalAxis)

xx, yy = np.meshgrid(horizontalAxis, verticalAxis)
points = np.c_[xx.ravel(), yy.ravel()]
print(points)
z = model.predict(points)
z = z.reshape(xx.shape)
print(z)

plt.contourf(xx,yy, z, alpha = 0.3)
plt.scatter(dataX, dataY, c = testY, s=1)
plt.show()
