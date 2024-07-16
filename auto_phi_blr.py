import river
import csv
import math
import numpy as np
from river import preprocessing
from river import linear_model
from river import stream
from river import metrics
from river import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\MITACS_PROJECT\CPUSum_smote.csv")
X=df.drop(['freeswap'],axis='columns')
Y=df['freeswap']
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.8, random_state=42)
scaler=preprocessing.StandardScaler()
model = river.linear_model.BayesianLinearRegression()


iterator=river.stream.iter_pandas(train_x,train_y)
mae=river.metrics.MAE()
mse=river.metrics.MSE()
for x,y in iterator:

    scaler=scaler.learn_one(x)
    train_x=scaler.transform_one(x)
    model.learn_one(x,y)

iterator=river.stream.iter_pandas(test_x, test_y)

for x,y in iterator:
    scaler=scaler.learn_one(x)
    x=scaler.transform_one(x)
    y_pred=model.predict_one(x)
 
    model.learn_one(x,y) 
    mae.update(y,y_pred)
    mse.update(y,y_pred)

    #print(mae.get())
    print(mse.get())