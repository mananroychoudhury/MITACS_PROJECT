import river
import csv
import numpy as np
from river import preprocessing
from river import linear_model
from river import stream
from river import metrics
import ImbalancedLearningRegression as iblr
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
df=pd.read_csv(r"C:\Users\MAMANROY CHOUDHURY\OneDrive\Desktop\New Microsoft Excel Worksheet.csv")

X=df.drop(['fuelcon'],axis='columns')
Y=df ['fuelcon']

train_x, test_x, train_y, test_y = train_test_split(X,Y.values, test_size=0.8, random_state=42)
scaler=preprocessing.StandardScaler()
knn_model = KNeighborsRegressor(n_neighbors=1)
knn_model.fit(train_x,train_y)
train_preds = knn_model.predict(train_x)
test_preds = knn_model.predict(test_x)
for i in range(len(test_preds)):
    print(test_preds[i])
