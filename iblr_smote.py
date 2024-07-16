## Synthetic Minority Oversampling Technique (SMOTE) is a statistical technique for increasing the number of
## cases in your dataset in a balanced way
import pandas as pd 
import pandas
import seaborn
import matplotlib.pyplot as plt
import ImbalancedLearningRegression as iblr
housing = pandas.read_csv(
    "C:\MITACS_PROJECT\Bike Sharing Data.csv"
)
## conduct SMOTE
housing_ro = iblr.smote(
    data = housing, 
    y = "cnt"
)
