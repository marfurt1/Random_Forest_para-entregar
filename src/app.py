import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import pickle


url='https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
df = pd.read_csv(url, index_col=[0,3])

df=df.drop(columns='Cabin')

df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df[['Sex','Embarked','Survived']]=df[['Sex','Embarked','Survived']].astype('category')

y=df['Survived']
X=df.drop(columns=['Ticket','Survived']).copy()

X['Sex']=X['Sex'].cat.codes
X['Embarked']=X['Embarked'].cat.codes

df[['Sex','Embarked','Survived']]=df[['Sex','Embarked','Survived']].astype('category')

y=df['Survived']
X=df.drop(columns=['Ticket','Survived']).copy()

X['Sex']=X['Sex'].cat.codes

X['Embarked']=X['Embarked'].cat.codes


filename = '../models/finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#Predict using the model
Pclass = 3
Sex = 1
Age = 70.0
SibSp = 2
Parch = 1
Ticket = 2666
Fare = 19.2583
Embarked = 2

#predigo el target para los valores seteados con modelo
print('Predicted Survived 1 : \n', loaded_model.predict([[Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,]]))

Pclass = 1
Sex = 0
Age = 28.0
SibSp = 0
Parch = 0
Ticket = 349207
Fare = 7.8958
Embarked = 0


#predigo el target para los valores seteados con modelo
print('Predicted Survived 2 : \n', loaded_model.predict([[Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,]]))

