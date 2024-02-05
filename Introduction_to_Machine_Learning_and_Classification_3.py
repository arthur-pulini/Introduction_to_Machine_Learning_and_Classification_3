import pandas as pd 
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


SEED = 20

model = LinearSVC()

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

import seaborn as sns

sns.scatterplot(data = datas, x = "expected_hours", y = "price", hue="finished")

#The graphics can be separated
sns.relplot(data = datas, x = "expected_hours", y = "price", hue="finished", col="finished")

x = datas[['expected_hours', 'price']]
y = datas['finished']

trainX, testX, trainY, testY = train_test_split(x, y, random_state = SEED, test_size = 0.25, stratify = y)

model.fit(trainX, trainY)

predictions = model.predict(testX)

accuracyScore = accuracy_score(testY, predictions)
print("The accuracy was: %.2f " % (accuracyScore * 100))