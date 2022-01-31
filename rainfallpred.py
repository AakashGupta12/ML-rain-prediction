import pandas as pd
from matplotlib import pyplot
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df=pd.read_csv('C:\\Users\\HP-PC\\Downloads\\weatherAUS.csv\\weatherAUS.csv')
print(df.shape)
print(df.columns)
print(df.info)

le=LabelEncoder()
df['Rain']=le.fit_transform(df.RainToday)

df_clean=df[['MinTemp', 'MaxTemp','WindGustSpeed','Humidity9am','Rain']]
df_clean2=df_clean[df_clean['Rain']<2]
df_clean2.describe()

df_clean2.info()

df_clean2.head()

df_clean2.head(10)

df_clean2.head(10)

correlations=df_clean2.corr()
fig=pyplot.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=np.arange(0,5,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df_clean2.columns)
ax.set_yticklabels(df_clean2.columns)
pyplot.show()

df_clean2.dropna(inplace=True)
df_clean2.info()

from sklearn.model_selection import train_test_split
X=df_clean2.iloc[:,:4]
Y=df_clean2['Rain']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,Y_train)
Y_pred1=lr.predict(X_test)
print(accuracy_score(Y_test,Y_pred1))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
Y_pred2=knn.predict(X_test)
print(accuracy_score(Y_test,Y_pred2))

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,Y_train)
Y_pred3=rf.predict(X_test)
print(accuracy_score(Y_test,Y_pred3))

from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier();
mlp.fit(X_train,Y_train);
Y_pred4=mlp.predict(X_test)
print(accuracy_score(Y_test,Y_pred4))


from joblib import dump
dump(mlp,'modelimp.joblib')