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
df=pd.read_csv(r"C:\MITACS_PROJECT\bank8FM.csv")

preprocessed_values = iblr.tomeklinks(
    data = df, 
    y = "rej"
)
print(preprocessed_values)
preprocessed_values.to_csv('BankFM_cnn.csv')