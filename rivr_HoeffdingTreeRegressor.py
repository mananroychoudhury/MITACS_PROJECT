import river
import csv
import numpy as np
from river import preprocessing
from river import linear_model
from river import stream
from river import metrics
from river import tree
from sklearn.model_selection import train_test_split
import pandas as pd
df=pd.read_csv(r"C:\MITACS_PROJECT\BankFM_smote.csv")
X=df.drop(['rej'],axis='columns')
Y=df['rej']
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.8, random_state=42)
scaler=preprocessing.StandardScaler()
model = river.tree.HoeffdingTreeRegressor()


iterator=river.stream.iter_pandas(train_x,train_y)

for x,y in iterator:

    scaler=scaler.learn_one(x)
    train_x=scaler.transform_one(x)
    model.learn_one(x,y)

iterator=river.stream.iter_pandas(test_x, test_y)

#mae=river.metrics.MAE()
#mse=river.metrics.MSE()


for x,y in iterator:
    scaler=scaler.learn_one(x)
    x=scaler.transform_one(x)
    y_pred=model.predict_one(x)
 
    model.learn_one(x,y)
    #mae.update(y,y_pred)
    #mse.update(y,y_pred)

    #print(mae.get())
    #print(mse.get())
    print(str(y_pred))