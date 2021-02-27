import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn import tree

#x_train, x_test , y_train,y_test =train_test_split(x,y,test_size=0.1)
# model.fit(x,y)
#prediction = model.predict(x_test)
#score = accuracy_score(y_test,prediction)

df = pd.read_csv('music.csv')
x = df.drop(columns=['genre'])
y = df['genre']

model = DecisionTreeClassifier()
model = joblib.load("music- reccommender.joblib")
predictions = model.predict([[21, 1]])
print(predictions)