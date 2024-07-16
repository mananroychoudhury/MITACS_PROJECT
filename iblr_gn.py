## Gaussian Noise is introduced because Firstly it does accurately reflect many systems. Second, because it
## is very easy to deal with mathematically, making it an attractive model to use.
import pandas as pd 
import pandas
import seaborn
import matplotlib.pyplot as plt
import ImbalancedLearningRegression as iblr
housing = pandas.read_csv(
    "C:\MITACS_PROJECT\Bike Sharing Data.csv"
)
## Introduces Gaussian Noise
housing_ro = iblr.gn(
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