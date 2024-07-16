## Edited Nearest Neighbor (ENN) is an undersampling method technique that remove the majority class to 
## match the minority class. ENN works by removing samples whose class label differs from the class of the 
## majority of their k nearest neighbors.
import pandas as pd 
import pandas
import seaborn
import matplotlib.pyplot as plt
import ImbalancedLearningRegression as iblr
housing = pandas.read_csv(
    "C:\MITACS_PROJECT\Bike Sharing Data.csv"
)
## conduct ENN
housing_ro = iblr.enn(
    data = housing,   
    y = "cnt", rel_thres = 0.5, rel_method = "manual", rel_ctrl_pts_rg = ([106,0,0],[36,1,0.5],[977,1,0])
)
   
housing.shape
housing_ro.shape
iblr.box_plot_stats(housing['cnt'])['stats']
iblr.box_plot_stats(housing_ro['cnt'])['stats']
seaborn.kdeplot(housing['cnt'], label = "Original")
seaborn.kdeplot(housing_ro['cnt'], label = "Modified")
plt.legend(labels=["Original","Modified"])
plt.show()